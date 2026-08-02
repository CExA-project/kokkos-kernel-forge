// clang-format off
#include <gcc-plugin.h>
#include <plugin-version.h>
#include <tree.h>
#include <tree-iterator.h>
#include <print-tree.h>
#include <cp/cp-tree.h>
#include <stringpool.h>
#include <iostream>
#include <vector>
// clang-format on

int plugin_is_GPL_compatible;

struct ViewInfo {
  // The declaration of View::data()
  tree func_decl_data = NULL_TREE;
  // The component_ref that references the view on which data() will be called
  tree component_ref = NULL_TREE;
  // The TYPE_DECL for the View's memory space
  tree memory_space = NULL_TREE;
};

void visit(std::vector<ViewInfo>& views, tree root, tree node);

bool visit_view(std::vector<ViewInfo>& views, tree root, tree node) {
  if (DECL_NAME(TYPE_NAME(node)) != get_identifier("View")) {
    return false;
  }

  tree scope = TYPE_CONTEXT(node);
  if (TREE_CODE(scope) != NAMESPACE_DECL ||
      DECL_NAME(scope) != get_identifier("Kokkos")) {
    return false;
  }

  tree data = lookup_qualified_name(node, "data", LOOK_want::NORMAL);
  if (!data) {
    std::cerr << "ERROR: Failed to find the data method of Kokkos::View\n";
    abort();
  }

  tree memory_space =
      lookup_qualified_name(node, "memory_space", LOOK_want::TYPE);
  if (!memory_space) {
    std::cerr << "ERROR: Failed to find the memory_space of Kokkos::View\n";
    abort();
  }

  views.emplace_back(data, root, memory_space);
  return true;
}

void visit_record(std::vector<ViewInfo>& views, tree root, tree record) {
  if (visit_view(views, root, record)) {
    return;
  }

  for (tree member = TYPE_FIELDS(record); member; member = TREE_CHAIN(member)) {
    visit(views, root, member);
  }
}

// We iterate recursively on the functor type to get the view it and its
// subobject contain
// NOTE: we assume that the functor or its subobject do not contain references
// to views, or if they do, those are references to other views contained in the
// functor
void visit(std::vector<ViewInfo>& views, tree root, tree node) {
  switch (TREE_CODE(node)) {
    case RECORD_TYPE: visit_record(views, root, node); break;
    case FIELD_DECL:
      visit(views, build_simple_component_ref(root, node), TREE_TYPE(node));
      break;
    default: break;
  }
}

std::vector<ViewInfo> get_functor_views(tree functor_variable,
                                        tree functor_type) {
  std::vector<ViewInfo> views;
  visit(views, functor_variable, functor_type);
  return views;
}

void ast_callback(void* gcc_data, void* user_data) {
  tree ast = reinterpret_cast<tree>(gcc_data);

  if (TREE_CODE(ast) != FUNCTION_DECL ||
      DECL_NAME(ast) != get_identifier("parallel_for")) {
    return;
  }

  tree scope = DECL_CONTEXT(ast);
  if (TREE_CODE(scope) != NAMESPACE_DECL ||
      DECL_NAME(scope) != get_identifier("krepe")) {
    return;
  }

  tree args = DECL_ARGUMENTS(ast);

  int nb_args = 0;
  while (args && ++nb_args < 3) {
    args = TREE_CHAIN(args);
  }

  if (nb_args != 3) {
    std::cerr
        << "ERROR: krepe::parallel_for is supposed to take 3 arguments, got "
        << nb_args << '\n';
    return;
  }

  tree functor_param = args;
  tree functor_type  = TREE_TYPE(args);

  if (TREE_CODE(functor_type) == REFERENCE_TYPE) {
    functor_type  = TREE_TYPE(functor_type);
    functor_param = convert_from_reference(functor_param);
  }

  if (!functor_type || TREE_CODE(functor_type) != RECORD_TYPE) {
    std::cerr << "ERROR: Third argument of krepe::parallel_for is not a record "
                 "type ("
              << get_tree_code_name(TREE_CODE(functor_type)) << ")\n";
    abort();
    return;
  }

  auto views = get_functor_views(functor_param, functor_type);

  // Logic for modifying the body of krepe::parallel_for. This assumes that the
  // body is an if statement, with its true branch being a statement list
  // (several statements and no variable declaration)
  tree body = DECL_SAVED_TREE(ast);
  if (TREE_CODE(body) != IF_STMT) {
    std::cerr
        << "ERROR: expected an if statement in krepe::parallel_for, but found ("
        << get_tree_code_name(TREE_CODE(body)) << ")\n";
    abort();
  }

  tree if_body = THEN_CLAUSE(body);
  if (TREE_CODE(if_body) != STATEMENT_LIST) {
    std::cerr << "ERROR: expected a statement list in the true branch of "
                 "krepe::parallel_for's if, but found ("
              << get_tree_code_name(TREE_CODE(if_body)) << ")\n";
    abort();
  }

  tree impl_ns = lookup_qualified_name(scope, "impl", LOOK_want::NAMESPACE);
  if (!impl_ns) {
    std::cerr << "ERROR: failed to find the krepe::impl namespace\n";
    abort();
  }
  tree register_view =
      lookup_qualified_name(impl_ns, "register_view", LOOK_want::NORMAL);
  if (!register_view || TREE_CODE(register_view) != FUNCTION_DECL) {
    std::cerr << "ERROR: failed to find krepe::impl::register_view\n";
    abort();
  }

  tree kokkos_ns =
      lookup_qualified_name(global_namespace, "Kokkos", LOOK_want::NAMESPACE);
  if (!kokkos_ns) {
    std::cerr << "ERROR: failed to find the Kokkos namespace\n";
    abort();
  }

  tree host_space =
      lookup_qualified_name(kokkos_ns, "HostSpace", LOOK_want::TYPE);
  if (!host_space) {
    std::cerr << "ERROR: failed to find the Kokkos::HostSpace\n";
    abort();
  }

  tree void_ptr_type = build_pointer_type(void_type_node);

  tree_stmt_iterator it = tsi_start(if_body);

  for (auto [fn, arg, memory_space] : views) {
    tree this_pointer = build_address(arg);
    tree data_expr    = build_call_n(fn, 1, this_pointer);

    tree is_host_pointer =
        build_int_cst(boolean_type_node,
                      static_cast<bool>(same_type_p(TREE_TYPE(host_space),
                                                    TREE_TYPE(memory_space))));

    tree casted_data = build_c_cast(UNKNOWN_LOCATION, void_ptr_type, data_expr);
    tree call_expr =
        build_call_n(register_view, 2, casted_data, is_host_pointer);

    tree cleanup_expr               = build_nt(CLEANUP_POINT_EXPR, call_expr);
    TREE_TYPE(cleanup_expr)         = void_type_node;
    TREE_SIDE_EFFECTS(cleanup_expr) = true;

    tsi_link_before(&it, cleanup_expr, TSI_SAME_STMT);
  }

  return;
}

int plugin_init(plugin_name_args* plugin_info, plugin_gcc_version* version) {
  if (errorcount || sorrycount) {
    // A compile failure already happened earlier
    return 1;
  }

  // FIXME: Use a version variable from CMake or something else
  plugin_info->version = "0.0.1";
  // FIXME: Add an help string
  plugin_info->help = "Some useful help message...";

  if (!plugin_default_version_check(version, &gcc_version)) {
    return 1;
  }

  register_callback(plugin_info->base_name, PLUGIN_PRE_GENERICIZE, ast_callback,
                    nullptr);

  return 0;
}
