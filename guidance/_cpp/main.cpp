#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/complex.h>
// #include <pybind11/functional.h>
// #include <pybind11/chrono.h>
#include <any>
#include "byte_trie.cpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp, m) {
    m.doc() = "Performance sensitive parts of guidance that have been written in C++.";

    py::class_<ByteTrie, std::shared_ptr<ByteTrie>>(m, "ByteTrie")
        .def(py::init<std::vector<std::string>>())
        .def(py::init<std::vector<std::string>, std::vector<int>>())
        .def("insert", &ByteTrie::insert)
        .def("has_child", &ByteTrie::has_child)
        .def("child", &ByteTrie::child)
        .def("parent", &ByteTrie::parent)
        .def("__len__", &ByteTrie::size) 
        .def("keys", [](const ByteTrie& self) {
            auto byte_strings = self.keys();
            py::list py_byte_strings;
            for (size_t i = 0; i < byte_strings.size(); i++) {
                py_byte_strings.append(py::bytes(&byte_strings[i], 1));
            }
            return py_byte_strings;
        })
        .def("compute_probs", &ByteTrie::compute_probs)
        .def_readwrite("match_version", &ByteTrie::match_version)
        .def_readwrite("match", &ByteTrie::match)
        .def_readwrite("partial_match", &ByteTrie::partial_match)
        .def_readwrite("prob", &ByteTrie::prob)
        .def_readwrite("value", &ByteTrie::value)
        .def_readwrite("children", &ByteTrie::children);
}