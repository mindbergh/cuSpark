#ifndef CUSPARK_COMMON_TYPES_H
#define CUSPARK_COMMON_TYPES_H

#include <stdio.h>
#include <sstream>
#include <string>
#include <common/logging.h>

namespace cuspark {

  struct point{
    float4 x;
    double y;
  };

  template<typename T>
    struct StringMapFunction {
      point operator()(std::string arg) {
        std::stringstream iss(arg);
        point p;
        iss >> p.x.x  >> p.x.y  >> p.x.z >> p.x.w >> p.y;
        return p;
      }
    };
}



#endif
