#ifndef CUSPARK_COMMON_TYPES_H
#define CUSPARK_COMMON_TYPES_H

#include <boost/function.hpp>
#include <boost/function_equal.hpp>



namespace cuspark {
  
  template <typename T, typename U>
  using MapFunction = boost::function<U (const T& x)>;

  template <typename T>
  using ReduceFunction = boost::function<T (const T& a, const T& b)>;


}



#endif
