#ifndef CUSPARK_COMMON_CONTEXT_H
#define CUSPARK_COMMON_CONTEXT_H


#include "common/logging.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "pipeline/pipeline.h"


namespace cuspark {

class Context {
  public:
  Context();
  char *getDeviceName();
  int getDeviceSM();
  size_t getTotalGlobalMem();

  template <typename BaseType>
  PipeLine<BaseType> textFile(std::string path, uint32_t size);

  private:
    cudaDeviceProp deviceProps;
};



}

#endif
