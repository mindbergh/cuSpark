#include <gtest/gtest.h>
#include "common/logging.h"
#include "common/context.h"
#include "common/types.h"
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"

using namespace cuspark;

typedef Array<float, 18> double18;

struct point{
  double18 x;
  float y;
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
  InputMapOp func = [] (const std::string& line) -> point {
    std::stringstream iss(line);
    point p;
    iss >> p.y;
    for(int i = 0; i < 18; i++){
      iss >> p.x.data[i];
    }
    return p;
  };
class PipeLineLogisticRegressionTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(PipeLineLogisticRegressionTest, Basic) {
  DLOG(INFO) << "******************Running Logistic Regression Test******************";
  Context context;
  context.printDeviceInfo();
  /*
  InputMapOp func = [] (const std::string& line) -> point {
    std::stringstream iss(line);
    point p;
    iss >> p.y;
    for(int i = 0; i < 18; i++){
      iss >> p.x.data[i];
    }
    return p;
  };
  */

  PipeLine<point> pl = context.textFile<point>("/tmp/muyangya/SUSY.csv", 10000000, func);
  pl.Materialize(Host);

  double18 w;
  
  for (int i = 0; i < 100; i++){

    //MappedPipeLine<double18, point, mapfunctor> mpl = pl.Map<double18>(mapfunctor(w));
    //double18 wdiff = mpl.Reduce(reducefunctor());
    //double18 wdiff = pl.Map<double18>(mapfunctor(w)).Reduce(reducefunctor());
    
    auto mpl = pl.Map<double18>(mapfunctor(w));
    double18 wdiff = mpl.Reduce(reducefunctor());

    w = w + wdiff;
    DLOG(INFO) << "##################iteration: #"<< i ;
  }
  
}

