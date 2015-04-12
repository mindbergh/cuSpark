#ifndef CUSPARK_COMMON_TYPES_H
#define CUSPARK_COMMON_TYPES_H

#include <boost/function.hpp>
#include <boost/function_equal.hpp>

namespace cuspark {
  
  template <typename T, typename U>
  using MapFunction = std::function<U(T)>;

  template <typename T>
  using ReduceFunction = std::function<T (const T& a, const T& b)>;


}



#endif
