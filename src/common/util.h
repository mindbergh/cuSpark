#ifndef CUSPARK_COMMON_UTIL_H
#define CUSPARK_COMMON_UTIL_H


namespace cuspark {

  enum MemLevel {Host, Cuda, None};
  
  const char * mlString(MemLevel ml);
}
#endif
