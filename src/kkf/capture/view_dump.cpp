#include "view_dump.hpp"

#include "kkf/common/hdf5_utils.hpp"
#include "memory_copy.hpp"

#include <cctype>
#include <cstddef>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace kkf {
namespace {

std::mutex dump_mutex;
using Hdf5Handle = hdf5::ScopedHandle;

std::string pointer_to_string(const void* ptr) {
  std::ostringstream stream;
  stream << ptr;
  return stream.str();
}

std::string sanitize_name(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char c : value) {
    const auto byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte) || c == '_' || c == '-' || c == '.') {
      result.push_back(c);
    } else {
      result.push_back('_');
    }
  }

  if (result.empty()) {
    return "unnamed";
  }
  if (result.size() > 80) {
    result.resize(80);
  }
  return result;
}

std::string dump_filename(std::string_view label, std::uint64_t kernel_id,
                          std::string_view phase) {
  return "kkf_" + sanitize_name(label) + "_" + std::to_string(kernel_id) + "_" +
         std::string(phase) + ".h5";
}

void write_string_attribute(hid_t object, const char* name,
                            std::string_view value) {
  const std::string text(value);
  Hdf5Handle type(CHECK_HDF5_ID(H5Tcopy(H5T_C_S1)), H5Tclose);

  const std::size_t size = text.empty() ? 1 : text.size() + 1;
  CHECK_HDF5_CALL(H5Tset_size(type.get(), size));
  CHECK_HDF5_CALL(H5Tset_strpad(type.get(), H5T_STR_NULLTERM));

  Hdf5Handle space(CHECK_HDF5_ID(H5Screate(H5S_SCALAR)), H5Sclose);
  Hdf5Handle attr(
      CHECK_HDF5_ID(H5Acreate2(object, name, type.get(), space.get(),
                               H5P_DEFAULT, H5P_DEFAULT)),
      H5Aclose);
  CHECK_HDF5_CALL(H5Awrite(attr.get(), type.get(), text.c_str()));

  attr.close_checked();
  space.close_checked();
  type.close_checked();
}

void write_uint64_attribute(hid_t object, const char* name,
                            std::uint64_t value) {
  Hdf5Handle space(CHECK_HDF5_ID(H5Screate(H5S_SCALAR)), H5Sclose);
  Hdf5Handle attr(
      CHECK_HDF5_ID(H5Acreate2(object, name, H5T_NATIVE_UINT64, space.get(),
                               H5P_DEFAULT, H5P_DEFAULT)),
      H5Aclose);

  CHECK_HDF5_CALL(H5Awrite(attr.get(), H5T_NATIVE_UINT64, &value));

  attr.close_checked();
  space.close_checked();
}

void write_int_attribute(hid_t object, const char* name, int value) {
  Hdf5Handle space(CHECK_HDF5_ID(H5Screate(H5S_SCALAR)), H5Sclose);
  Hdf5Handle attr(
      CHECK_HDF5_ID(H5Acreate2(object, name, H5T_NATIVE_INT, space.get(),
                               H5P_DEFAULT, H5P_DEFAULT)),
      H5Aclose);

  CHECK_HDF5_CALL(H5Awrite(attr.get(), H5T_NATIVE_INT, &value));

  attr.close_checked();
  space.close_checked();
}

template <class T>
auto Hdf5Datatype() {
  if constexpr (std::is_same_v<T, unsigned char>) {
    return H5T_NATIVE_UCHAR;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return H5T_NATIVE_INT64;
  } else {
    static_assert(false, "unsupported datatype");
  }
}

template <class T>
void write_dataset(hid_t group, const char* name, const std::vector<T>& data) {
  auto datatype         = Hdf5Datatype<T>();
  const hsize_t dims[1] = {static_cast<hsize_t>(data.size())};
  Hdf5Handle space(CHECK_HDF5_ID(H5Screate_simple(1, dims, nullptr)), H5Sclose);
  Hdf5Handle dataset(
      CHECK_HDF5_ID(H5Dcreate2(group, name, datatype, space.get(), H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT)),
      H5Dclose);

  if (!data.empty()) {
    CHECK_HDF5_CALL(H5Dwrite(dataset.get(), datatype, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data.data()));
  }

  dataset.close_checked();
  space.close_checked();
}

void write_policy(hid_t policy_group, const NoPolicyDesc&) {
  write_string_attribute(policy_group, "type", "none");
}

void write_policy(hid_t policy_group, const ScalarPolicyDesc& policy) {
  write_string_attribute(policy_group, "type", "scalar");
  write_uint64_attribute(policy_group, "end", policy.end);
}

void write_policy(hid_t policy_group, const RangePolicyDesc& policy) {
  write_string_attribute(policy_group, "type", "range");
  write_int_attribute(policy_group, "index_type_size",
                      policy.index_type_desc.size);
  write_int_attribute(policy_group, "index_type_signed",
                      policy.index_type_desc.is_signed);
  write_string_attribute(policy_group, "space", policy.space);
  write_uint64_attribute(policy_group, "begin", policy.begin);
  write_uint64_attribute(policy_group, "end", policy.end);
}

void write_policy(hid_t policy_group, const MDRangePolicyDesc& policy) {
  write_string_attribute(policy_group, "type", "mdrange");
  write_int_attribute(policy_group, "index_type_size",
                      policy.index_type_desc.size);
  write_int_attribute(policy_group, "index_type_signed",
                      policy.index_type_desc.is_signed);
  write_string_attribute(policy_group, "space", policy.space);
  write_int_attribute(policy_group, "rank", policy.rank);
  write_dataset(policy_group, "begin", policy.begin);
  write_dataset(policy_group, "end", policy.end);
  write_dataset(policy_group, "tile", policy.tile);
}

void write_policy(hid_t policy_group, const TeamPolicyDesc& policy) {
  write_string_attribute(policy_group, "type", "team");
  write_int_attribute(policy_group, "index_type_size",
                      policy.index_type_desc.size);
  write_int_attribute(policy_group, "index_type_signed",
                      policy.index_type_desc.is_signed);
  write_string_attribute(policy_group, "space", policy.space);
  write_int_attribute(policy_group, "team_size", policy.team_size);
  write_int_attribute(policy_group, "league_size", policy.league_size);
  write_int_attribute(policy_group, "team_scratch_0", policy.team_scratch_0);
  write_int_attribute(policy_group, "team_scratch_1", policy.team_scratch_1);
  write_int_attribute(policy_group, "thread_scratch_0",
                      policy.thread_scratch_0);
  write_int_attribute(policy_group, "thread_scratch_1",
                      policy.thread_scratch_1);
}

void write_allocation_group(hid_t views_group,
                            const ActiveAllocation& allocation,
                            std::size_t index) {
  const std::string group_name = "view_" + std::to_string(index) + "_" +
                                 sanitize_name(allocation.record.label);
  Hdf5Handle group(
      CHECK_HDF5_ID(H5Gcreate2(views_group, group_name.c_str(), H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT)),
      H5Gclose);

  write_string_attribute(group.get(), "label", allocation.record.label);
  write_string_attribute(group.get(), "space", allocation.record.space);
  write_string_attribute(group.get(), "ptr", pointer_to_string(allocation.ptr));
  write_string_attribute(group.get(), "p_data",
                         pointer_to_string(allocation.record.p_data));
  write_uint64_attribute(group.get(), "size", allocation.record.size);

  std::vector<unsigned char> bytes;
  const std::string skip_reason = copy_allocation_bytes(allocation, bytes);
  write_int_attribute(group.get(), "bytes_dumped", skip_reason.empty() ? 1 : 0);
  if (!skip_reason.empty()) {
    write_string_attribute(group.get(), "skip_reason", skip_reason);
    group.close_checked();
    return;
  }

  write_dataset(group.get(), "bytes", bytes);

  group.close_checked();
}

}  // namespace

ViewDumpResult dump_view_snapshot(
    const AllocationSnapshot& snapshot,
    const std::vector<unsigned char>& functor_data,
    const std::unordered_map<std::string, std::string>& metadata,
    const std::variant<kkf::NoPolicyDesc, kkf::ScalarPolicyDesc,
                       kkf::RangePolicyDesc, kkf::MDRangePolicyDesc,
                       kkf::TeamPolicyDesc>& policy,
    std::string_view phase, std::string_view label, std::uint64_t kernel_id,
    std::uint64_t kernel_invocation) {
  ViewDumpResult result;
  result.filename = dump_filename(label, kernel_id, phase);

  std::lock_guard<std::mutex> lock(dump_mutex);
  try {
    Hdf5Handle file(
        CHECK_HDF5_ID(H5Fcreate(result.filename.c_str(), H5F_ACC_TRUNC,
                                H5P_DEFAULT, H5P_DEFAULT)),
        H5Fclose);

    write_string_attribute(file.get(), "phase", phase);
    write_string_attribute(file.get(), "kernel_label", label);
    write_uint64_attribute(file.get(), "kernel_id", kernel_id);
    write_uint64_attribute(file.get(), "kernel_invocation", kernel_invocation);
    write_uint64_attribute(file.get(), "active_allocations",
                           snapshot.allocations.size());
    write_uint64_attribute(file.get(), "active_bytes", snapshot.active_bytes);

    Hdf5Handle views_group(
        CHECK_HDF5_ID(H5Gcreate2(file.get(), "views", H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT)),
        H5Gclose);
    for (std::size_t i = 0; i < snapshot.allocations.size(); ++i) {
      write_allocation_group(views_group.get(), snapshot.allocations[i], i);
    }
    views_group.close_checked();

    Hdf5Handle metadata_group(
        CHECK_HDF5_ID(H5Gcreate2(file.get(), "metadata", H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT)),
        H5Gclose);
    for (const auto& [key, value] : metadata) {
      write_string_attribute(metadata_group.get(), key.c_str(), value);
    }
    metadata_group.close_checked();

    Hdf5Handle policy_group(
        CHECK_HDF5_ID(H5Gcreate2(file.get(), "policy", H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT)),
        H5Gclose);
    std::visit([&](auto&& arg) { write_policy(policy_group.get(), arg); },
               policy);

    write_dataset(file.get(), "functor", functor_data);

    file.close_checked();
    result.ok = true;
  } catch (const std::exception& error) {
    result.ok    = false;
    result.error = error.what();
  } catch (...) {
    result.ok    = false;
    result.error = "unknown non-standard exception";
  }
  return result;
}

}  // namespace kkf
