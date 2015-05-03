#include <gtest/gtest.h>
#include "pipeline/mappedpipeline.h"
#include "pipeline/pipeline.h"
#include "common/logging.h"
#include "common/context.h"
#include <boost/lexical_cast.hpp>


using namespace cuspark;

typedef int (*InputMapOp)(const std::string&);


class PipeLineReduceTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

template <typename T, typename Func>
void loadToHost(std::string path, uint32_t size, T* t, Func f) {
  uint32_t count = 0;
  std::ifstream infile;
  infile.open(path);
  std::string line;

  while (std::getline(infile, line)) {
    t[count++] = f(line);
  }

  EXPECT_EQ(count, size);
  return;
};

struct mapfunctor {
  const int factor;
  mapfunctor(int factor) : factor(factor) {}
  __host__ __device__
  int operator()(int a) {
    return a * factor;
  }
};

struct reducefunctor {
  __host__ __device__
  int operator() (int a, int b) {
    return a + b;
  }
};

TEST_F(PipeLineReduceTest, Basic) {
  DLOG(INFO) << "******************Running Reduce Test******************";
  std::string path = "./data/testInts.txt";
  uint32_t size = 10;
  Context context;
  InputMapOp func = [] (const std::string& line) -> int {
    int res = boost::lexical_cast<int>(line);
    return res;
  };
  PipeLine<int> pl = context.textFile<int>(path, size, func);
  
  pl.Materialize(Host);
 
  int factor = 3;
  MappedPipeLine<int, int, mapfunctor> mpl = pl.Map<int>(mapfunctor(factor));
  
  mpl.Materialize(Host);
  int res = mpl.Reduce(reducefunctor());

  EXPECT_EQ(res, 55 * 3);

}