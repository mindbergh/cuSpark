#include <gtest/gtest.h>
#include "common/logging.h"
#include "common/context.h"
#include "common/types.h"
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"

using namespace cuspark;

typedef Array<double, 18> double18;

struct point{
  double18 x;
  double y;
};

struct mapfunctor {
  double18 w_;
  mapfunctor(double18 w) : w_(w) {}
  __host__ __device__ 
  double18 operator()(point arg) { 
    float dotproduct = 0;
    for(int i = 0; i < 18; i++)
      dotproduct += arg.x.get(i) * w_.get(i);
    dotproduct = (1/(1+exp(-arg.y * dotproduct)) - 1) * arg.y;
    double18 result;
    for(int i = 0; i < 18; i++)
      result.set(i, arg.x.get(i) * dotproduct);
    return result;
  }
};

struct reducefunctor { 
  __host__ __device__ 
  double18 operator() (double18 arg1, double18 arg2){
    return arg1 + arg2;
  }
};

typedef point (*InputMapOp)(const std::string&);

class PipeLineLogisticRegressionTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(PipeLineLogisticRegressionTest, Basic) {
  DLOG(INFO) << "******************Running Logistic Regression Test******************";
  float eta = 0.01;
  Context context;
  InputMapOp func = [] (const std::string& line) -> point {
    std::stringstream iss(line);
    point p;
    iss >> p.y;
    for(int i = 0; i < 18; i++){
      iss >> p.x.data[i];
    }
    return p;
  };
  //PipeLine<point> pl = context.textFile<point>("/tmp/muyangya/SUSY.csv", 5000000, func);
  PipeLine<point> pl = context.textFile<point>("/tmp/muyangya/SUSY.csv", 1000, func);
  //pl.Materialize(Host);

  double18 w;
  
  for (int i = 0; i < 100; i++){

    MappedPipeLine<double18, point, mapfunctor> mpl = pl.Map<double18>(mapfunctor(w));
    
    EXPECT_EQ(pl.size_, pl.GetDataSize());
    EXPECT_EQ(pl.size_, mpl.size_);
    
    double18 wdiff = mpl.Reduce(reducefunctor());

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

