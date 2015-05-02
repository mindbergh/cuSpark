#include <gtest/gtest.h>
#include "pipeline/pipeline.h"
#include "pipeline/textpipeline.h"
#include "common/logging.h"
#include "common/context.h"


using namespace cuspark;

struct Inputfunctor {
  point operator()(std::string arg) {
    std::stringstream iss(arg);
    point p;
    iss >> p.x.x  >> p.x.y  >> p.x.z >> p.x.w >> p.y;
    return p;
  }
};

typedef point (*InputMapOp)(const std::string&);


class TextPipeLineCreateTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};


bool isPointEqual(const point& p1, const point& p2) {
  if (p1.x.x != p2.x.x) return false;
  if (p1.x.y != p2.x.y) return false;
  if (p1.x.z != p2.x.z) return false;
  if (p1.x.w != p2.x.w) return false;
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



TEST_F(TextPipeLineCreateTest, Basic) {
  std::string path = "./data/lrdatasmall.txt";
  uint32_t size = 32928;
  Context context;
  context.printDeviceInfo();
  InputMapOp func = [] (const std::string& line) -> point {
    std::stringstream iss(line);
    point p;
    iss >> p.x.x  >> p.x.y  >> p.x.z >> p.x.w >> p.y;
    return p;
  };
  TextPipeLine<point> * tpl = context.textFile<point>(path, size, func);
  tpl->Materialize(Host);
  

  point points[tpl->size_];
  
  loadToHost<point>(path, size, points, func);

  for (int i = 0; i < tpl->GetDataSize(); i++) {

    EXPECT_TRUE(isPointEqual(points[i], tpl->data_[i]));
  }
  
  tpl->Materialize(None);

  EXPECT_EQ(tpl->data_, nullptr);
  
  delete tpl;
}
