#include "common/util.h" 

namespace cuspark {

  const char * MemLevelStrings[] = { "Host", "Cuda", "None" };
  
  const char * mlString(MemLevel ml) {
    return MemLevelStrings[ml];  
  }
}
