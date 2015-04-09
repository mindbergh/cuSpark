#include "common/init.h"


#include "common/logging.h"






void impala::InitCommon(int argc, char** argv) {
  cuspark::InitGoogleLoggingSafe(argv[0]);

  LOG(INFO) << "Common Initial Finished."

}
