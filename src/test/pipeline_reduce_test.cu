#include <gtest/gtest.h>
#include "pipeline/pipeline.h"
#include "common/logging.h"
#include "common/context.h"
#include "common/types.h"
#include <boost/lexical_cast.hpp>

using namespace cuspark;

class PipeLineReduceTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

typedef Array<double, 18> double18;

struct point{
  double18 x;
  double y;
};
  
typedef point (*InputMapOp)(const std::string&);

struct reducefunctor {
  __host__ __device__
  point operator() (point arg1, point arg2){
    point p;
    p.x = arg1.x + arg2.x;
    p.y = arg1.y + arg2.y;
    return p;
  }
};

TEST_F(PipeLineReduceTest, Basic) {
  DLOG(INFO) << "******************Running Reduce Test******************";
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
  PipeLine<point> pl = context.textFile<point>("/tmp/muyangya/SUSY.csv", 5000000, func);
 
  point res = pl.Reduce(reducefunctor());

  EXPECT_EQ(res.y, 55);

}
