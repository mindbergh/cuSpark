#include "common/init.h"
#include "common/logging.h"




void cuspark::InitCommon(int argc, char** argv) {
  cuspark::InitGoogleLoggingSafe(argv[0]);
  FLAGS_log_dir = "./log/";

  LOG(INFO) << "Common Initial Finished." << std::endl;
}
