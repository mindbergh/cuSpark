#include <iostream>
#include <gtest/gtest.h>
#include <common/types.h>
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"

using namespace cuspark;

class PipeLineMapTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

int do_exponential(int a){
  std::cout << "*";
  return a*a;
}
MapFunction<int, int> exponential = do_exponential;
/*
struct do_exponential{
  int operator()(int a) { return a * a; };
};
MapFunction<int, int> exponential = do_exponential();
*/

TEST_F(PipeLineMapTest, Basic) {
  uint32_t N = 100;

  int data[N];
  uint32_t i;

  for (i = 0; i < N; ++i) {
    data[i] = i;
  }

  PipeLine<int> pl(data, N);
  MappedPipeLine<int, int> mpl = pl.Map(exponential);

  EXPECT_EQ(N, mpl.GetDataSize());

  for (i = 0; i < N; ++i) {
    EXPECT_EQ(mpl.GetElement(i), data[i] * data[i]);
  }
}

