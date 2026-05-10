#include "view_dump.hpp"

#include "memory_copy.hpp"

#include <hdf5.h>

#include <cctype>
#include <cstddef>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace kkf {
namespace {

std::mutex dump_mutex;

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
    if (std::isalnum(byte) != 0 || c == '_' || c == '-' || c == '.') {
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
  const hid_t type = H5Tcopy(H5T_C_S1);
  if (type < 0) {
    return;
  }

  const std::size_t size = text.empty() ? 1 : text.size() + 1;
  H5Tset_size(type, size);
  H5Tset_strpad(type, H5T_STR_NULLTERM);

  const hid_t space = H5Screate(H5S_SCALAR);
  if (space < 0) {
    H5Tclose(type);
    return;
  }

  const hid_t attr =
      H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
  if (attr >= 0) {
    H5Awrite(attr, type, text.c_str());
    H5Aclose(attr);
  }

  H5Sclose(space);
  H5Tclose(type);
}

void write_uint64_attribute(hid_t object, const char* name,
                            std::uint64_t value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  if (space < 0) {
    return;
  }

  const hid_t attr = H5Acreate2(object, name, H5T_NATIVE_UINT64, space,
                                H5P_DEFAULT, H5P_DEFAULT);
  if (attr >= 0) {
    H5Awrite(attr, H5T_NATIVE_UINT64, &value);
    H5Aclose(attr);
  }

  H5Sclose(space);
}

void write_int_attribute(hid_t object, const char* name, int value) {
  const hid_t space = H5Screate(H5S_SCALAR);
  if (space < 0) {
    return;
  }

  const hid_t attr =
      H5Acreate2(object, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
  if (attr >= 0) {
    H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
  }

  H5Sclose(space);
}

void write_allocation_group(hid_t views_group,
                            const ActiveAllocation& allocation,
                            std::size_t index) {
  const std::string group_name = "view_" + std::to_string(index) + "_" +
                                 sanitize_name(allocation.record.label);
  const hid_t group = H5Gcreate2(views_group, group_name.c_str(), H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
  if (group < 0) {
    return;
  }

  write_string_attribute(group, "label", allocation.record.label);
  write_string_attribute(group, "space", allocation.record.space);
  write_string_attribute(group, "ptr", pointer_to_string(allocation.ptr));
  write_uint64_attribute(group, "size", allocation.record.size);

  std::vector<unsigned char> bytes;
  const std::string skip_reason = copy_allocation_bytes(allocation, bytes);
  write_int_attribute(group, "bytes_dumped", skip_reason.empty() ? 1 : 0);
  if (!skip_reason.empty()) {
    write_string_attribute(group, "skip_reason", skip_reason);
    H5Gclose(group);
    return;
  }

  const hsize_t dims[1] = {static_cast<hsize_t>(bytes.size())};
  const hid_t space     = H5Screate_simple(1, dims, nullptr);
  if (space < 0) {
    H5Gclose(group);
    return;
  }

  const hid_t dataset = H5Dcreate2(group, "bytes", H5T_NATIVE_UCHAR, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset >= 0) {
    if (!bytes.empty()) {
      H5Dwrite(dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               bytes.data());
    }
    H5Dclose(dataset);
  }

  H5Sclose(space);
  H5Gclose(group);
}

}  // namespace

ViewDumpResult dump_view_snapshot(const AllocationSnapshot& snapshot,
                                  std::string_view phase,
                                  std::string_view label,
                                  std::uint64_t kernel_id) {
  ViewDumpResult result;
  result.filename = dump_filename(label, kernel_id, phase);

  std::lock_guard<std::mutex> lock(dump_mutex);
  const hid_t file = H5Fcreate(result.filename.c_str(), H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    return result;
  }

  write_string_attribute(file, "phase", phase);
  write_string_attribute(file, "kernel_label", label);
  write_uint64_attribute(file, "kernel_id", kernel_id);
  write_uint64_attribute(file, "active_allocations",
                         snapshot.allocations.size());
  write_uint64_attribute(file, "active_bytes", snapshot.active_bytes);

  const hid_t views_group =
      H5Gcreate2(file, "views", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (views_group >= 0) {
    for (std::size_t i = 0; i < snapshot.allocations.size(); ++i) {
      write_allocation_group(views_group, snapshot.allocations[i], i);
    }
    H5Gclose(views_group);
  }

  H5Fclose(file);
  result.ok = true;
  return result;
}

}  // namespace kkf
