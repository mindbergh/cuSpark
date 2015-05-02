#include <gtest/gtest.h>
#include "common/logging.h"
#include <common/function.h>
#include "pipeline/textpipeline.h"
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


typedef point (*InputMapOp)(const std::string&);

class TextPipeLineLogisticRegressionTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(TextPipeLineLogisticRegressionTest, Basic) {
  float eta = 0.01;

  Context context;
  InputMapOp func = [] (const std::string& line) -> point {
    std::stringstream iss(line);
    point p;
    iss >> p.x.x  >> p.x.y  >> p.x.z >> p.x.w >> p.y;
    return p;
  };
  TextPipeLine<point> * tpl = context.textFile<point>("./data/lrdatasmall.txt", 32928, func);
  //tpl->Materialize(Cuda);

  float4 w = make_float4(1,1,1,1);
  float4 identity = make_float4(0,0,0,0);
  for (int i = 0; i < 100; i++){

    MappedPipeLine<float4, point, mapfunctor> mpl = tpl->Map<float4>(mapfunctor(w));
    
    mpl.Materialize(Cuda);
    
    EXPECT_EQ(tpl->size_, tpl->GetDataSize());
    EXPECT_EQ(tpl->size_, mpl.size_);
    
    //float4 wdiff = mpl.Reduce(reducefunctor(), identity);

    //thrust::device_ptr dptr(mpl.data_);
    
    //float4 *mem_data = new float4[mpl.size_];



    //cudaMemcpy(mem_data, mpl.data_, mpl.size_ * sizeof(float4), cudaMemcpyDeviceToHost);

    //for (int i = 0; i < mpl.size_; i++) {
    //  DLOG(INFO) << mem_data[i].x << ", " << mem_data[i].y << ", " <<  mem_data[i].z << ", " << mem_data[i].w << std::endl;
    //}
  

    //w = make_float4(w.x+eta*wdiff.x, w.y+eta*wdiff.y, w.z+eta*wdiff.z, w.w+eta*wdiff.w);
    //DLOG(INFO) << "iteration: #"<< i << ", wdiff:" <<wdiff.x<<", "<<wdiff.y<<", "<<wdiff.z<<", "<<wdiff.w<<std::endl;
  }
}

