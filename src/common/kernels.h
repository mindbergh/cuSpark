#ifndef CUSPARK_COMMON_KERNELS_H
#define CUSPARK_COMMON_KERNELS_H

#include "common/logging.h"



namespace cuspark {
  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    __global__ void
    global_transform(KeyType* key, ValueType* value, BaseType* base, UnaryOp f, int N) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= N) return;

      auto pair = f(base[idx]);
      key[idx] = pair.first;
      value[idx] = pair.second;
      //printf("This key = %d\n", pair.first);
    }
}


#endif
