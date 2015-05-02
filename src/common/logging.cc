#include "common/logging.h"

bool logging_initialized = false;

void cuspark::InitGoogleLoggingSafe(const char* arg) {
  if (logging_initialized) return;

  FLAGS_log_dir = "./log/";
  
  google::InitGoogleLogging(arg);

  logging_initialized = true;
}

void cuspark::ShutdownLogging() {
  google::ShutdownGoogleLogging();
}
