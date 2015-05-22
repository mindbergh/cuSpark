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

        total_memory = this->getTotalGlobalMem();
        usable_memory = 1024 * 1024 * 1024;//total_memory * 0.8;
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

      // Called typically when materializing data to cuda
      int addUsage(uint32_t size){
        DLOG(INFO) << "Adding GPU Global Memory usage by "<< (size / (1024 * 1024)) 
            << "MB, now left: " << (usable_memory / (1024 * 1024));
        if(size > usable_memory) 
          return -1;
        else {
          usable_memory -= size;
          return usable_memory;
        }
      }

      // Called typically when materializing data to None
      int reduceUsage(uint32_t size){
        usable_memory += size;
        DLOG(INFO) << "Reducing GPU Global Memory usage by "<< (size / (1024 * 1024)) 
            << "MB, now left: " << (usable_memory / (1024 * 1024));
        return usable_memory;
      }
  
      uint32_t getUsableMemory(){
        DLOG(INFO) << "Getting Usable GPU Memory: " << (usable_memory / (1024 * 1024));
        return usable_memory;
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
      uint32_t total_memory; 
      uint32_t usable_memory; 
  };

}


#endif
