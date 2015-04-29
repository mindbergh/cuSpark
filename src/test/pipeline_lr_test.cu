#include <gtest/gtest.h>
#include "common/logging.h"
#include <common/function.h>
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"

using namespace cuspark;

struct mapfunctor {
  const float4 w_;
  mapfunctor(float4 w) : w_(w) {}
  __host__ __device__ 
    float4 operator()(point arg) { 
      float dotproduct = arg.x.x * w_.x + arg.x.y * w_.y + arg.x.z * w_.z + arg.x.w * w_.w;
      dotproduct = (1/(1+exp(-arg.y * dotproduct)) - 1) * arg.y;
      return make_float4(arg.x.x*dotproduct, arg.x.y*dotproduct, arg.x.z*dotproduct, arg.x.w*dotproduct);
    }
};


struct reducefunctor { 
  __host__ __device__ 
    float4 operator() (float4 arg1, float4 arg2){
      return make_float4(arg1.x+arg2.x, arg1.y+arg2.y, arg1.z+arg2.z, arg1.w+arg2.w);
    }
};


class PipeLineLogisticRegressionTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(PipeLineLogisticRegressionTest, Basic) {

  float eta = 0.01;

  StringMapFunction<point> f1;
  PipeLine<point> pl("./data/lrdatasmall.txt", 32928, f1);
  pl.Cache();
  point* out = pl.GetData();

  float4 w = make_float4(1,1,1,1);
  for(int i = 0; i < 1000; i++){
    
    MappedPipeLine<float4, point, mapfunctor> mpl = pl.Map<float4>(mapfunctor(w));
    
    float4 wdiff = mpl.Reduce(reducefunctor());
    
    w = make_float4(w.x+eta*wdiff.x, w.y+eta*wdiff.y, w.z+eta*wdiff.z, w.w+eta*wdiff.w);
    DLOG(INFO) << "iteration: #"<< i << ", wdiff:" <<wdiff.x<<", "<<wdiff.y<<", "<<wdiff.z<<", "<<wdiff.w<<std::endl;
  }
}

