#include "pywarpper_image_processor.h"

#include <pybind11/embed.h>
#include <torch/extension.h>

#include "butil/synchronization/lock.h"
#include "core/common/global_flags.h"

namespace py = pybind11;

namespace janus {

class __attribute__((visibility("hidden"))) PyWrapperImpl {
 private:
  PyWrapperImpl() {
    py::gil_scoped_acquire gil;

    mm_ = py::module_::import("pybind.multimodal");
    preprocess_ = mm_.attr("preprocess");
  }
  ~PyWrapperImpl() {}

 public:
  static PyWrapperImpl& instance() {
    static PyWrapperImpl ins;
    return ins;
  }

  bool execute(const MMInput& inputs, MMData& data) {
    butil::AutoLock lock_guard(lock_);
    py::gil_scoped_acquire acquire;

    try {
      py::list py_lst;
      for (const auto& item : inputs.items()) {
        py_lst.append(py::bytes(item.raw_data));
      }

      py::dict res = preprocess_(py_lst, FLAGS_model);
      data = std::move(MMData(MMType::IMAGE, py::cast<MMDict>(res)));

      return true;
    } catch (std::exception& e) {
      LOG(ERROR) << "python call fail, exception is " << e.what();
      return false;
    }
  }

 private:
  py::module_ mm_;
  py::object preprocess_;

  butil::Lock lock_;
};

PyWarpperImageProcessor::PyWarpperImageProcessor(const ModelArgs&) {
  PyWrapperImpl::instance();
}

bool PyWarpperImageProcessor::process(const MMInput& inputs, MMData& datas) {
  return PyWrapperImpl::instance().execute(inputs, datas);
}

}  // namespace janus
