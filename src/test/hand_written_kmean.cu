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

#define K 5
#define N 5000000


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
  Context context;
  uint32_t maxMem = context.getUsableMemory();
  uint32_t unitMem = 2 * sizeof(point) + 2 * sizeof(int);
  uint32_t size_ = N;
  uint32_t n = std::min(maxMem / unitMem, size_);  // partition size
  int num_partition = (N + n - 1) / n;

  double start = CycleTimer::currentSeconds();

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

  double end = CycleTimer::currentSeconds();

  DLOG(INFO) << "Reading finished. time tood: " << (end - start) * 1000 << " ms";

  point *old_cen = (point*)malloc(K * sizeof(point));
  point *new_cen;
  int* id = (int*)malloc(K * sizeof(int));
  int* cnt = (int*)malloc(K * sizeof(int));

  point* cuda_base;
  point* cuda_cen;

  thrust::device_ptr<point> cuda_cen_ptr = thrust::device_malloc<point>(K);
  thrust::device_ptr<point> cuda_point_ptr = thrust::device_malloc<point>(n);

  thrust::device_ptr<int> cuda_id_ptr = thrust::device_malloc<int>(n);

  thrust::device_ptr<int> cuda_ided_ptr = thrust::device_malloc<int>(n);
  thrust::device_ptr<point> cuda_reduced_ptr = thrust::device_malloc<point>(n);

  cuda_cen = thrust::raw_pointer_cast(cuda_cen_ptr);
  
  thrust::equal_to<int> pred;
  auto reducer = reducefunctor();

  float diff;

  std::map<int, point> cen_map;
  std::map<int, int> cnt_map;

  memcpy(old_cen, base, K * sizeof(point));
  int iter = 0;

  for (int i = 0; i < 100; i ++) {
    cen_map.clear();
    cnt_map.clear();

    point *new_cen = (point*)malloc(K * sizeof(point));
    for (int k = 0; k < num_partition; k++) {
      uint32_t partition_start = k * n;
      uint32_t myn = std::min(n, N - k * n);
      DLOG(INFO) << "About to processe (" << partition_start << ", " << partition_start << "+" << myn << ") in " << N;
      point* mybase = base + partition_start;
      thrust::copy(mybase, mybase+myn, cuda_point_ptr);
      thrust::transform(cuda_point_ptr, cuda_point_ptr+myn, cuda_id_ptr, mapfunctor(old_cen));
      thrust::sort_by_key(cuda_id_ptr, cuda_id_ptr+myn, cuda_point_ptr);
      auto new_end = thrust::reduce_by_key(cuda_id_ptr, cuda_id_ptr+myn, cuda_point_ptr, cuda_ided_ptr, cuda_reduced_ptr, pred, reducer);

      int num_group = new_end.first - cuda_ided_ptr;
      assert(num_group == K);
      thrust::copy(cuda_ided_ptr, cuda_ided_ptr+num_group, id);
      thrust::copy(cuda_reduced_ptr, cuda_reduced_ptr+num_group, new_cen);


      for (int j = 0; j < num_group; j++) {
        cnt[j] = thrust::count(cuda_id_ptr, cuda_id_ptr+myn, id[j]);
        auto got_cen = cen_map.find(id[j]);
        if (got_cen == cen_map.end()) {
          cen_map[id[j]] = new_cen[j];
        } else {
          cen_map[id[j]] = reducer(got_cen->second, new_cen[j]);
        }

        auto got_cnt = cnt_map.find(id[j]);
        if (got_cnt == cnt_map.end()) {
          cnt_map[id[j]] = cnt[j];
        } else {
          cnt_map[id[j]] = got_cnt->second + cnt[j];
        }
      }
    }

    diff = 0;
    int k = 0;
    for (auto it = cen_map.begin(); it != cen_map.end(); ++it) {
      new_cen[k] = it->second / cnt_map[it->first];
      diff += new_cen[k].distTo(old_cen[k]);
      k++;
    }



    diff /= K;
    free(old_cen);
    old_cen = new_cen;
    DLOG(INFO) << "Diff = " << diff;
  }
  iter++;

  free(id);
  free(new_cen);
  thrust::device_free(cuda_cen_ptr);
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

