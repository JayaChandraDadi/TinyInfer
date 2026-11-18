#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tinyinfer/gemm.h"
#include "tinyinfer/session.h"

namespace py = pybind11;
using namespace tinyinfer;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

Tensor from_numpy(const FloatArray& a) {
    Shape shape(a.shape(), a.shape() + a.ndim());
    std::vector<float> data(a.data(), a.data() + a.size());
    return Tensor(std::move(shape), std::move(data));
}

py::array to_numpy(const Tensor& t) {
    std::vector<py::ssize_t> shape(t.shape.begin(), t.shape.end());
    if (t.is_int) {
        py::array_t<int64_t> out(shape);
        std::copy(t.i64.begin(), t.i64.end(), out.mutable_data());
        return std::move(out);
    }
    py::array_t<float> out(shape);
    std::copy(t.data.begin(), t.data.end(), out.mutable_data());
    return std::move(out);
}

py::list run_outputs(std::vector<Tensor> outs) {
    py::list l;
    for (auto& t : outs) l.append(to_numpy(t));
    return l;
}

}  // namespace

PYBIND11_MODULE(_tinyinfer, m) {
    m.doc() = "TinyInfer: a small C++ inference engine for ONNX CNNs";

    py::class_<Session>(m, "Session")
        .def(py::init<const std::string&>(), py::arg("onnx_path"))
        .def("run",
             [](Session& s, const FloatArray& x) {
                 Tensor t = from_numpy(x);
                 std::vector<Tensor> outs;
                 {
                     py::gil_scoped_release release;
                     outs = s.run(t);
                 }
                 return run_outputs(std::move(outs));
             },
             py::arg("input"), "Run a single-input model; returns a list of numpy arrays.")
        .def("run",
             [](Session& s, const std::map<std::string, FloatArray>& feeds) {
                 std::map<std::string, Tensor> ts;
                 for (auto& kv : feeds) ts[kv.first] = from_numpy(kv.second);
                 std::vector<Tensor> outs;
                 {
                     py::gil_scoped_release release;
                     outs = s.run(ts);
                 }
                 return run_outputs(std::move(outs));
             },
             py::arg("feeds"))
        .def_property_readonly("input_names", &Session::input_names)
        .def_property_readonly("output_names", &Session::output_names)
        .def_property_readonly("input_shapes", &Session::input_shapes)
        .def_property_readonly("op_types", &Session::op_types)
        .def_property_readonly("num_nodes", &Session::num_nodes)
        .def("enable_profiling", &Session::enable_profiling, py::arg("on") = true)
        .def("reset_profile", &Session::reset_profile)
        .def("profile", [](const Session& s) {
            py::list l;
            for (auto& p : s.profile())
                l.append(py::dict(py::arg("op") = p.op_type, py::arg("calls") = p.calls,
                                  py::arg("total_ms") = p.total_ms));
            return l;
        });

    m.def("set_num_threads", &set_num_threads, py::arg("n"));
    m.def("get_num_threads", &get_num_threads);
    m.def("simd_available", &simd_available);
    m.def("supported_ops", &Session::supported_ops);
    m.def("set_conv_algo", [](const std::string& a) {
        if (a == "direct") config().conv_algo = ConvAlgo::Direct;
        else if (a == "im2col") config().conv_algo = ConvAlgo::Im2colGemm;
        else throw std::invalid_argument("conv algo must be 'direct' or 'im2col'");
    });
    m.def("set_gemm_kernel", [](const std::string& k) {
        if (k == "naive") config().gemm_kernel = GemmKernel::Naive;
        else if (k == "blocked") config().gemm_kernel = GemmKernel::Blocked;
        else if (k == "simd") config().gemm_kernel = GemmKernel::Simd;
        else throw std::invalid_argument("gemm kernel must be 'naive', 'blocked' or 'simd'");
    });
    m.def("gemm", [](const FloatArray& a, const FloatArray& b) {
        if (a.ndim() != 2 || b.ndim() != 2 || a.shape(1) != b.shape(0)) throw std::invalid_argument("gemm: bad shapes");
        py::array_t<float> c({a.shape(0), b.shape(1)});
        tinyinfer::gemm(a.shape(0), b.shape(1), a.shape(1), a.data(), b.data(), c.mutable_data());
        return c;
    }, "C = A @ B using the configured GEMM kernel");
}
