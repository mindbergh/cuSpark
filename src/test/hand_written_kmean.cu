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
    int operator()(point arg) { 
      int id = -1;
      float dist = CUDART_INF;
      for (int i = 0; i < n; i++) {
        float thisdist = arg.distTo(centroids[i]);
        if (thisdist < dist) {
          dist = thisdist;
          id = i;
        }
      }
      return id;
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

class HandWrittenKMeanTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(HandWrittenKMeanTest, Basic) {
  DLOG(INFO) << "******************Running HandWritten KMean Test******************";

  int K = 5;
  int N = 1000000;
  point base[N];
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

  std::string line;
  std::ifstream infile;
  infile.open("/tmp/mingf/SUSY.txt");
  int count = 0;
  while (std::getline(infile, line) && count < N) {
    base[count++] = func(line);
  }
  
  DLOG(INFO) << "Reading finished.";

  point *old_cen = (point*)malloc(K * sizeof(point));
  point *new_cen;
  int* id = (int*)malloc(K * sizeof(int));
  int* cnt = (int*)malloc(K * sizeof(int));

  point* cuda_base;
  point* cuda_cen;

  thrust::device_ptr<point> cuda_cen_ptr = thrust::device_malloc<point>(K);
  thrust::device_ptr<point> cuda_base_ptr = thrust::device_malloc<point>(N);
  thrust::device_ptr<point> cuda_point_ptr = thrust::device_malloc<point>(N);

  thrust::device_ptr<int> cuda_id_ptr = thrust::device_malloc<int>(N);

  thrust::device_ptr<int> cuda_ided_ptr = thrust::device_malloc<int>(N);
  thrust::device_ptr<point> cuda_reduced_ptr = thrust::device_malloc<point>(N);

  cuda_cen = thrust::raw_pointer_cast(cuda_cen_ptr);
  thrust::copy(base, base+N, cuda_base_ptr);
  thrust::equal_to<int> pred;

  float diff;

  memcpy(old_cen, base, K * sizeof(point));
  int iter = 0;
  for (int i = 0; i < 100; i ++) {
    thrust::copy(cuda_base_ptr, cuda_base_ptr+N, cuda_point_ptr);


    thrust::transform(cuda_point_ptr, cuda_point_ptr+N, cuda_id_ptr, mapfunctor(old_cen));
    thrust::sort_by_key(cuda_id_ptr, cuda_id_ptr+N, cuda_point_ptr);
    auto new_end = thrust::reduce_by_key(cuda_id_ptr, cuda_id_ptr+N, cuda_point_ptr, cuda_ided_ptr, cuda_reduced_ptr, pred, reducefunctor());

    int num_group = new_end.first - cuda_ided_ptr;

    thrust::copy(cuda_ided_ptr, cuda_ided_ptr+num_group, id);
    point *new_cen = (point*)malloc(K * sizeof(point));
    thrust::copy(cuda_reduced_ptr, cuda_reduced_ptr+num_group, new_cen);


    for (int j = 0; j < num_group; j++) {
      cnt[j] = thrust::count(cuda_id_ptr, cuda_id_ptr+N, id[j]);
    }

    diff = 0;

    //DLOG(INFO) << "The resuling size: " << num_group;
    //DLOG(INFO) << "New Centroids: ";
    for (int j = 0; j < num_group; j++) {
      new_cen[j] = new_cen[j] / cnt[j];
      //DLOG(INFO) << id[j] << ":" << new_cen[j].toString();
      diff += new_cen[j].distTo(old_cen[j]);
    }
    diff /= K;
    //DLOG(INFO) << "Diff = " << diff;
    free(old_cen);
    old_cen = new_cen;
    iter++;
  }
  

  free(id);
  free(new_cen);
  thrust::device_free(cuda_cen_ptr);
  thrust::device_free(cuda_base_ptr);
  thrust::device_free(cuda_point_ptr);

  thrust::device_free(cuda_id_ptr);

  thrust::device_free(cuda_ided_ptr);
  thrust::device_free(cuda_reduced_ptr);

  DLOG(INFO) << "Converged after iter: " << iter;
  /*



     point *new_cen = nullptr;

     for (int i = 0; i < 100; i++) { 
     DLOG(INFO) << "K mean iteration: " << i;

     thrust::transform()
     PairedPipeLine<int, point, point, mapfunctor> assignedPairs = points.Map<int, point>(mapfunctor(old_cen));

     auto res = assignedPairs.ReduceByKey(reducefunctor());

     id = std::get<0>(res);
     new_cen = std::get<1>(res);
     cnt = std::get<2>(res);
     size_t size = std::get<3>(res);
     float diff = 0;

     DLOG(INFO) << "The resuling size: " << size;
     DLOG(INFO) << "New Centroids: ";
     for (int j = 0; j < size; j++) {
     new_cen[j] = new_cen[j] / cnt[j];
     DLOG(INFO) << id[j] << ":" << new_cen[j].toString();
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
   */
}

