#include <boost/thread/mutex.hpp>
#include "common/logging.h"


bool logging_initialized = false;

using namespace boost;

mutex logging_mutex;

void cuspark::InitGoogleLoggingSafe(const char* arg) {
  mutex::scoped_lock logging_lock(logging_mutex);
  if (logging_initialized) return;

  google::InitGoogleLogging(arg);

  logging_initialized = true;
}

void cuspark::ShutdownLogging() {
  mutex::scoped_lock logging_lock(logging_mutex);
  google::ShutdownGoogleLogging();
}
