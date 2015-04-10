#include <gtest/gtest.h>
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"

using namespace cuspark;

class PipeLineTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};


TEST_F(PipeLineTest, Basic) {
  uint32_t N = 100;

  int data[N];
  uint32_t i;

  for (i = 0; i < N; ++i) {
    data[i] = i;
  }


  PipeLine<int> pl(data, N);

  //EXPECT_EQ(N, pl.GetDataSize());
  //EXPECT_EQ(data, pl.GetData);



  for (i = 0; i < N; ++i) {
    //EXPECT_EQ(pl.GetElement(i), data[i]);
  }
}

