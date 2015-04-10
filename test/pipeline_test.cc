#include <gtest/gtest.h>
#include <glog/logging.h>
#include <pipeline.h>
#include <mappedpipeline.h>

namespace cuspark {

  TEST(PipeLineTest, Basic) {
    uint32_t N = 100;

    int data[N];
    uint32_t i;

    for (i = 0; i < N; ++i) {
      data[i] = i;
    }


    cuspark::PipeLine<int> pl(data, N);

    EXPECT_EQ(pl.GetDataSize(), N);
    EXPECT_EQ(pl.GetData, data);



    for (i = 0; i < N; ++i) {
      EXPECT_EQ(pl.GetElement(i), data[i]);
    }


  }


}
int main(int argc, char **argv) {
  cuspark::InitGoogleLoggingSafe(argv[0]);
  LOG(INFO) << "PipeLine Test: Basic Starts."  << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
