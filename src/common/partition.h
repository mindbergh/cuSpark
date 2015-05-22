#ifndef CUSPARK_COMMON_PARTITION_H
#define CUSPARK_COMMON_PARTITION_H

#include "common/logging.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <queue>

#define THREAD_POOL_SIZE 1

namespace cuspark {

  struct PartitionInfo {

    cudaStream_t stream_;
 
    uint32_t partition_start_;
    uint32_t partition_end_;
    uint32_t this_partition_size_;

    uint32_t partition_id_;
    uint32_t num_partitions_;

    void* data_;
    uint32_t used_data_size_ = 0;
    std::mutex* memcpy_lock_;
  };

  class PartitionWorkQueue{

    public:
      PartitionWorkQueue(uint32_t partition_size, uint32_t total_size, uint32_t unit_size, uint32_t action = 0)
      : action_(action),
      partition_size_(partition_size),
      total_size_(total_size),
      unit_size_(unit_size){
        std::mutex memcpy_lock;
        num_partitions_ = (total_size + partition_size - 1) / partition_size;
        for(uint32_t i = 0; i < num_partitions_; i++){
          PartitionInfo* partition = new PartitionInfo[1];
          partition->partition_start_ = i * partition_size;
          partition->partition_end_ = std::min(total_size, (i+1) * partition_size);
          partition->this_partition_size_ = partition->partition_end_ - partition->partition_start_;
          partition->partition_id_ = i;
          partition->num_partitions_ = num_partitions_;
          partition->memcpy_lock_ = &memcpy_lock;
          queue_.push(partition);
        }
      }

      PartitionInfo* pop(){
        queue_lock_.lock();
        if(!queue_.empty()){
          PartitionInfo* partition = queue_.front();
          queue_.pop();
          queue_lock_.unlock();
          return partition;
        }else{
          queue_lock_.unlock();
          return NULL;
        }
      }
 
      uint32_t num_partitions_;
      uint32_t unit_size_;
      uint32_t partition_size_;
      uint32_t total_size_;
      std::queue<PartitionInfo*> queue_;
      std::mutex queue_lock_;
      uint32_t action_;
  };

}

#endif
