#ifndef CUSPARK_COMMON_LOGGING_H
#define CUSPARK_COMMON_LOGGING_H

#include <glog/logging.h>
#include <gflags/gflags.h>

namespace cuspark {

void InitGoogleLoggingSafe(const char* arg);

void ShutdownLogging();

}

#endif
