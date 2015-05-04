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
  point operator()(point arg) { 
    float dotproduct = 0;
    for(int i = 0; i < 18; i++)
      dotproduct += arg.x.get(i) * w_.get(i);
    dotproduct = (1/(1+exp(-arg.y * dotproduct)) - 1) * arg.y;
    point result;
    for(int i = 0; i < 18; i++)
      result.x.set(i, arg.x.get(i) * dotproduct);
    result.y = arg.y;
    return result;
  }
};

struct reducefunctor { 
  __host__ __device__ 
  point operator() (point arg1, point arg2){
    point p;
    p.x = arg1.x + arg2.x;
    p.y = arg1.y + arg2.y;
    return p;
  }
};

typedef point (*InputMapOp)(const std::string&);

class PipeLineMapTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(PipeLineMapTest, Basic) {
  DLOG(INFO) << "******************Running Map Test******************";
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
  PipeLine<point> pl = context.textFile<point>("/tmp/muyangya/SUSY.csv", 1000000, func);
  pl.Materialize(Cuda);

  double18 w;
  for(int i = 0; i < 18; i++){
    w.set(i, 0);
  }
  MappedPipeLine<point, point, mapfunctor> mpl = pl.Map<point>(mapfunctor(w));
  //point res = mpl.Reduce(reducefunctor()); 
  point res;
  for(int i = 0; i < 100; i++){
    res = mpl.Reduce(reducefunctor()); 
  }
  EXPECT_EQ(res.y, 55);

}

