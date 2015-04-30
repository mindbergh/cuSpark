#include <limits.h>
#include <gtest/gtest.h>
#include "common/init.h"

int main(int argc, char **argv) {
  cuspark::InitCommon(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
