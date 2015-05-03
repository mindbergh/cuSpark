#ifndef CUSPARK_COMMON_CONTEXT_H
#define CUSPARK_COMMON_CONTEXT_H


#include "common/logging.h"
#include <cuda.h>
#include <cuda_runtime.h>


namespace cuspark {

  template <typename BaseType>
    class PipeLine;

  class Context {
    public:
      Context() {
        DLOG(INFO) << "Initial cuSpark context..." << std::endl;
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0) {
          DLOG(INFO) << "No CUDA device found.. Exit..." << std::endl;
          exit(1);
        }
        cudaGetDeviceProperties(&deviceProps, 0);
      }

      char *getDeviceName() {
        return deviceProps.name;
      }

      int getDeviceSM() {
        return deviceProps.multiProcessorCount;
      }

      size_t getTotalGlobalMem() {
        return deviceProps.totalGlobalMem;
      }

      template <typename BaseType>
        BaseType unit(BaseType a) {
          return a;
        }

      template <typename BaseType, typename InputMapOp>
        PipeLine<BaseType> textFile(std::string path, uint32_t size, InputMapOp f) {
          return PipeLine<BaseType>(path, size, f, this); 
        }

      void printDeviceInfo() {
        printf("    Device:\t%s\n", this->getDeviceName());
        printf("       SMs:\t%d\n", this->getDeviceSM());
        printf("Global mem:\t%.0f MB\n",
            static_cast<float>(this->getTotalGlobalMem()) / (1024 * 1024));

      }

    private:
      cudaDeviceProp deviceProps;
  };

}


#endif
