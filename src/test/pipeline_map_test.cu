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

/*
struct do_exponential{
  __device__ int operator()(int a){
    return a*a;
  }
};
MapFunction<int, int> exponential = do_exponential();
*/
__host__ __device__ int do_exponential(int i){ return 4* i; }
MapFunction<int, int, int(*)(int)> exponential{ do_exponential };

TEST_F(PipeLineMapTest, Basic) {
  uint32_t N = 5;

  int data[N];
  uint32_t i;

  for (i = 0; i < N; ++i) {
    data[i] = i;
  }

  PipeLine<int> pl(data, N);
  
  MappedPipeLine<int, int> mpl = pl.Map(exponential);

  EXPECT_EQ(N, mpl.GetDataSize());

  int* out = mpl.GetData();
  for (i = 0; i < N; ++i) {
    EXPECT_EQ(out[i], 4 * data[i]);
  }
  
}

