#include <gtest/gtest.h>
#include "common/logging.h"
#include "common/context.h"
#include "common/types.h"
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"
#include "pipeline/pairedpipeline.h"

#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <math_constants.h>

#include <tuple>

using namespace cuspark;

typedef Tuple<float, 18> point;
/*
   struct point {
   double18 x;
   };
 */

struct mapfunctor {
  int n = 5;
  point centroids[5];
  mapfunctor(point* c) { 
    memcpy(centroids, c, n * sizeof(point)); 
  }

  __device__ 
    thrust::pair<int, point> operator()(point arg) { 
      int id = -1;
      float dist = CUDART_INF;
      for (int i = 0; i < n; i++) {
        float thisdist = arg.distTo(centroids[i]);
        if (thisdist < dist) {
          dist = thisdist;
          id = i;
        }
      }
      thrust::pair<int, point> res(id, arg);
      return res;
    }
};

struct reducefunctor { 
  __host__ __device__ 
    point operator() (point arg1, point arg2) {
      point p;
      p = arg1 + arg2;
      return p;
    }
};

typedef point (*InputMapOp)(const std::string&);

class PipeLinePairTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(PipeLinePairTest, Basic) {
  DLOG(INFO) << "******************Running Paring Test******************";

  int N = 5;

  Context context;
  //context.printDeviceInfo();
  InputMapOp func = [] (const std::string& line) -> point {
    std::stringstream iss(line);
    point p;
    float tmp;
    iss >> tmp;
    for (int i = 0; i < 18; i++) {
      iss >> p.data[i];
    }
    return p;
  };

  PipeLine<point> points = context.textFile<point>("/tmp/mingf/SUSY.txt", 1000000, func);
  points.Materialize(Host);
  point *old_cen = points.Take(N);
  int* id = nullptr;
  int* cnt = nullptr;

  point *new_cen = nullptr;
  int iter = 0;
  float diff = 0.0;
  size_t size; 

  for (int i = 0; i < 100; i ++) {
    //DLOG(INFO) << "K mean iteration: " << i;
    /*
       DLOG(INFO) << "Centroids: ";
       for (int j = 0; j < N; j++) {
       DLOG(INFO) << j << ": " << centroids[j].toString();
       }
     */
    PairedPipeLine<int, point, point, mapfunctor> assignedPairs = points.Map<int, point>(mapfunctor(old_cen));

    std::tie(id, new_cen, cnt, size) = assignedPairs.ReduceByKey(reducefunctor());
    /*
    id = std::get<0>(res);
    new_cen = std::get<1>(res);
    cnt = std::get<2>(res);
    size_t size = std::get<3>(res);
    */

    float diff = 0;

    //DLOG(INFO) << "The resuling size: " << size;
    //DLOG(INFO) << "New Centroids: ";
    for (int j = 0; j < size; j++) {
      new_cen[j] = new_cen[j] / cnt[j];
      //DLOG(INFO) << id[j] << ":" << new_cen[j].toString();
      diff += new_cen[j].distTo(old_cen[j]);
    }
    diff /= N;
    DLOG(INFO) << "Diff = " << diff;
    old_cen = new_cen;
  }
  //EXPECT_EQ(res.size(), 1);
  //auto a = res.begin();
  //int b = a->first;
  //EXPECT_EQ(b, 1);
}

