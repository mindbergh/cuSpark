#include "common/context.h"
#include "common/logging.h"


cuspark::Context::Context() {
  DLOG(INFO) << "Initial cuSpark context..." << std::endl;
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    DLOG(INFO) << "No CUDA device found.. Exit..." << std::endl;
    exit(1);
  }

  cudaGetDeviceProperties(&deviceProps, 0);
}

char* cuspark::Context::getDeviceName() {
  return deviceProps.name;
}

int cuspark::Context::getDeviceSM() {
  return deviceProps.multiProcessorCount;
}

size_t cuspark::Context::getTotalGlobalMem() {
  return deviceProps.totalGlobalMem;
}

template <typename BaseType>
cuspark::PipeLine<BaseType> cuspark::Context::textFile(std::string path, uint32_t size) {
  return new cuspark::PipeLine<BaseType>(path, size);
}
