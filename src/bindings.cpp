#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Stub function to verify nanobind integration and extension loading
int get_cpp_version() {
    return 1;
}

NB_MODULE(_kamila_cpp, m) {
    m.doc() = "C++ accelerated core routines for scikit-kamila using nanobind";
    m.def("get_cpp_version", &get_cpp_version, "Return the C++ engine version.");
}
