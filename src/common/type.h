#ifndef CUSPARK_COMMON_TYPE_H
#define CUSPARK_COMMON_TYPE_H



namespace cuspark {
  
  template<T, U>
  typedef boost::function<U (const T& a)> MapFunction;

  template<T>
  typedef boost::function<T (const T& a, const T& b)> ReduceFunction;

}



#endif
