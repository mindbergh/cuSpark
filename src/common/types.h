#ifndef CUSPARK_COMMON_TYPES_H
#define CUSPARK_COMMON_TYPES_H

namespace cuspark {

template<typename T, size_t n>
struct Array{

  T data[n];

  __host__ __device__
  void set(int i, T value){
    data[i] = value;
  }

  __host__ __device__
  T get(int i){
    return data[i];
  }

  __host__ __device__
  Array<T,n> operator+(Array<T, n> obj){
    Array<T,n> result;
    for(int i = 0; i < n; i++){
      result.set(i, data[i] + obj.get(i));
    }
    return result;
  }
};

}
#endif

