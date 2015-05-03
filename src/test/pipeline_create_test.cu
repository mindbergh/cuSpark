#include <gtest/gtest.h>
#include "pipeline/pipeline.h"
#include "common/logging.h"
#include "common/context.h"
#include "common/types.h"

using namespace cuspark;

struct point{
  Array<float, 4> x;
  double y;
};

struct Inputfunctor {
  point operator()(std::string arg) {
    std::stringstream iss(arg);
    point p;
    iss >> p.x.data[0]  >> p.x.data[1]  >> p.x.data[2] >> p.x.data[3] >> p.y;
    return p;
  }
};

typedef point (*InputMapOp)(const std::string&);

class PipeLineCreateTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

bool isPointEqual(const point& p1, const point& p2) {
  if (p1.x.data[0] != p2.x.data[0]) return false;
  if (p1.x.data[1] != p2.x.data[1]) return false;
  if (p1.x.data[2] != p2.x.data[2]) return false;
  if (p1.x.data[3] != p2.x.data[3]) return false;
  if (p1.y != p2.y) return false;
  return true;
}

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
}

TEST_F(PipeLineCreateTest, Basic) {
  DLOG(INFO) << "******************Running Create Test******************";
  std::string path = "./data/lrdatasmall.txt";
  uint32_t size = 32928;
  Context context;
  context.printDeviceInfo();
  InputMapOp func = [] (const std::string& line) -> point {
    std::stringstream iss(line);
    point p;
    iss >> p.x.data[0]  >> p.x.data[1]  >> p.x.data[2] >> p.x.data[3] >> p.y;
    return p;
  };
  PipeLine<point> pl = context.textFile<point>(path, size, func);
  pl.Materialize(Host);
  
  point points[pl.size_];
  loadToHost<point>(path, size, points, func);
  for (int i = 0; i < pl.GetDataSize(); i++) {
    EXPECT_TRUE(isPointEqual(points[i], pl.data_[i]));
  }
  
  pl.Materialize(None);
  EXPECT_EQ(pl.data_, nullptr);
}
