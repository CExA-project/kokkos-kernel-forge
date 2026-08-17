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
  // The component_ref that references the view on which data() will be called
  tree component_ref = NULL_TREE;
  // The declaration of View::data()
  tree view_data_func = NULL_TREE;
  // The TYPE_DECL for the View's memory space
  tree memory_space = NULL_TREE;
  // The declaration of memory_space::name()
  tree space_name_func = NULL_TREE;
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

  tree data_function = lookup_qualified_name(node, "data", LOOK_want::NORMAL);
  if (!data_function) {
    std::cerr << "ERROR: Failed to find the data method of Kokkos::View\n";
    abort();
  }

  tree memory_space =
      lookup_qualified_name(node, "memory_space", LOOK_want::TYPE);
  if (!memory_space) {
    std::cerr << "ERROR: Failed to find the memory_space of Kokkos::View\n";
    abort();
  }

  tree name_function =
      lookup_qualified_name(TREE_TYPE(memory_space), "name", LOOK_want::NORMAL);
  if (!name_function) {
    std::cerr << "ERROR: Failed to find the name method of a memory space\n";
    abort();
  }

  views.emplace_back(root, data_function, memory_space, name_function);
  return true;
}

void visit_record(std::vector<ViewInfo>& views, tree root, tree record) {
  if (visit_view(views, root, TYPE_MAIN_VARIANT(record))) {
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
      DECL_NAME(ast) != get_identifier("replay_functor")) {
    return;
  }

  tree scope = DECL_CONTEXT(ast);
  if (TREE_CODE(scope) != NAMESPACE_DECL ||
      DECL_NAME(scope) != get_identifier("krepe")) {
    return;
  }

  tree args = DECL_ARGUMENTS(ast);

  if (!args) {
    std::cerr << "ERROR: krepe::replay_functor is supposed to take 1 "
                 "arguments, got 0\n";
    return;
  }

  tree functor_param = args;
  tree functor_type  = TREE_TYPE(args);

  if (TREE_CODE(functor_type) == REFERENCE_TYPE) {
    functor_type  = TREE_TYPE(functor_type);
    functor_param = convert_from_reference(functor_param);
  }

  if (!functor_type || TREE_CODE(functor_type) != RECORD_TYPE) {
    std::cerr
        << "ERROR: the argument for krepe::replay_functor is not a record "
           "type ("
        << get_tree_code_name(TREE_CODE(functor_type)) << ")\n";
    abort();
    return;
  }

  auto views = get_functor_views(functor_param, functor_type);

  // Logic for modifying the body of krepe::replay_functor. This assumes that
  // the body is a statement list (several statements and no variable
  // declaration)
  tree body = DECL_SAVED_TREE(ast);

  if (TREE_CODE(body) != STATEMENT_LIST) {
    std::cerr << "ERROR: expected a statement list in "
                 "krepe::replay_functor, but found ("
              << get_tree_code_name(TREE_CODE(body)) << ")\n";
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

  tree void_ptr_type = build_pointer_type(void_type_node);

  tree_stmt_iterator it = tsi_start(body);

  bool first_it = true;
  for (auto [view_instance, data_fn, memory_space, space_name_fn] : views) {
    tree data_expr =
        build_new_method_call(view_instance, data_fn, nullptr, NULL_TREE,
                              LOOKUP_NORMAL, nullptr, tf_warning_or_error);

    tree name_expr =
        build_new_method_call(memory_space, space_name_fn, nullptr, NULL_TREE,
                              LOOKUP_NORMAL, nullptr, tf_warning_or_error);

    tree casted_data = build_c_cast(UNKNOWN_LOCATION, void_ptr_type, data_expr);

    tree clear_arg = build_int_cst(boolean_type_node, first_it);

    releasing_vec args;
    vec_safe_push(args, casted_data);
    vec_safe_push(args, name_expr);
    vec_safe_push(args, clear_arg);
    tree call_expr =
        build_new_function_call(register_view, &args, tf_warning_or_error);

    // FIXME: this cleanup expr might not be needed
    tree cleanup_expr               = build_nt(CLEANUP_POINT_EXPR, call_expr);
    TREE_TYPE(cleanup_expr)         = void_type_node;
    TREE_SIDE_EFFECTS(cleanup_expr) = true;

    tsi_link_before(&it, cleanup_expr, TSI_SAME_STMT);

    first_it = false;
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
