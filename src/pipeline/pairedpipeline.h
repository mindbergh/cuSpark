#ifndef CUSPARK_PIPELINE_PAIREDPIPELINE_H_
#define CUSPARK_PIPELINE_PAIREDPIPELINE_H_

#include <stdio.h>
#include <map>
#include <tuple>

#include "common/logging.h"
#include "common/util.h"
#include "pipeline/pipeline.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/count.h>


#include "common/CycleTimer.h"

namespace cuspark {

  /*
   * Mapped from type BaseType to type AfterType
   */
  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    class PairedPipeLine {

      //typedef std::tuple<KeyType*, ValueType*, int*, size_t> std::tuple<KeyType*, ValueType*, int*, size_t>;

      public:
        PairedPipeLine(PipeLine<BaseType> *parent, UnaryOp f);

        template <typename BinaryOp> 
          std::tuple<KeyType*, ValueType*, int*, size_t> ReduceByKey(BinaryOp f);

        template <typename BinaryPred, typename BinaryOp> 
          std::tuple<KeyType*, ValueType*, int*, size_t> ReduceByKey(BinaryPred pred, BinaryOp f);

        template <typename BinaryPred, typename BinaryOp> 
          std::tuple<KeyType*, ValueType*, int*, size_t> ReduceByKey_(BinaryPred pred, BinaryOp f);

        // return the data in memory
        //AfterType *GetData();

        Context *GetContext();

        uint32_t GetDataSize();

        void Materialize(MemLevel ml, bool hard_materialized = true);

        void Materialize_(MemLevel ml);

        //this function return the max unit memory that is used in a chain
        // used when counting the size of a partition when allocating data
        uint32_t GetMaxUnitMemory_();

        void Map_Partition_(KeyType* cuda_key, 
            ValueType* cuda_value,
            uint32_t partition_start, 
            uint32_t this_partition_size);

        // functor of the map operation
        UnaryOp f_;

        PipeLine<BaseType> *parent_;
        uint32_t size_;
        Context *context_;
        MemLevel memLevel_;


      private:
        template <typename BinaryOp>
          void merge(KeyType *cuda_key, 
              ValueType* cuda_value, 
              KeyType *base_key,
              int partition_size,
              int N, 
              BinaryOp f);

        std::tuple<KeyType*, ValueType*, int*, size_t> make_reduce_result();

        std::map<KeyType, ValueType> result;
        std::map<KeyType, int> count;
        
        KeyType* res_key = (KeyType*)malloc(sizeof(KeyType));
        ValueType* res_val = (ValueType*)malloc(sizeof(ValueType));
        int* res_cnt = (int*)malloc(sizeof(int));
        //AfterType* GetPartition_(uint32_t partition_start, uint32_t this_partition_size);
    };

  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::
    PairedPipeLine(PipeLine<BaseType> *parent, UnaryOp f) :
      memLevel_(None),
      size_(parent->GetDataSize()),
      context_(parent->GetContext()),
      parent_(parent),
      f_(f) {}

  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    template <typename BinaryOp>
    std::tuple<KeyType*, ValueType*, int*, size_t> PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::ReduceByKey(BinaryOp f) {
      return this->ReduceByKey(thrust::equal_to<KeyType>(), f);
    }


  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    template <typename BinaryPred, typename BinaryOp>
    std::tuple<KeyType*, ValueType*, int*, size_t> PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::ReduceByKey_(BinaryPred pred, BinaryOp f) {

      result.clear();
      count.clear();
      //std::map<KeyType, ValueType> &result = this->result_;
      uint32_t partition_size = std::min((this->context_->getUsableMemory() 
            / this->GetMaxUnitMemory_()), this->size_);
      uint32_t num_partitions = (this->size_ + partition_size - 1)/partition_size;
      DLOG(INFO) << "Reducing by Key from paired pipeline, with " << num_partitions << " partitions, each dealing with " << partition_size << " size of data";

      // Allocating the space for a single partition to hold
      KeyType* cuda_key;
      ValueType* cuda_value;
      KeyType* cuda_key_reduced;
      ValueType* cuda_value_reduced;

      cudaMalloc((void**)&cuda_key, partition_size * sizeof(KeyType));
      cudaMalloc((void**)&cuda_value, partition_size * sizeof(ValueType));
      cudaMalloc((void**)&cuda_key_reduced, partition_size * sizeof(KeyType));
      cudaMalloc((void**)&cuda_value_reduced, partition_size * sizeof(ValueType));

      // do this on each of the partitions
      for (uint32_t i = 0; i < num_partitions; i++) {
        uint32_t partition_start = i * partition_size;
        uint32_t partition_end = std::min(this->size_, (i+1) * partition_size);
        uint32_t this_partition_size = partition_end - partition_start;
        DLOG(INFO) << "Reducing, partition #" << i << ", size: "<< this_partition_size;
        this->Map_Partition_(cuda_key, cuda_value, partition_start, this_partition_size);    


        thrust::device_ptr<KeyType> key_ptr = thrust::device_pointer_cast(cuda_key);
        thrust::device_ptr<ValueType> value_ptr = thrust::device_pointer_cast(cuda_value);
        thrust::device_ptr<KeyType> keyed_ptr = thrust::device_pointer_cast(cuda_key_reduced);
        thrust::device_ptr<ValueType> valued_ptr = thrust::device_pointer_cast(cuda_value_reduced);

        thrust::sort_by_key(key_ptr, key_ptr+this_partition_size, value_ptr);

        auto new_end = thrust::reduce_by_key(key_ptr, 
            key_ptr+this_partition_size, 
            value_ptr, 
            keyed_ptr,
            valued_ptr,
            pred,
            f);

        int N = new_end.first - keyed_ptr;
        DLOG(INFO) << "Size of key group after RedcueByKey: " << N;
        DLOG(INFO) << "Size of map before merge: " << result.size();
        merge(cuda_key_reduced, 
            cuda_value_reduced, 
            cuda_key,
            this_partition_size,
            N, 
            f);
        DLOG(INFO) << "Size of map after merge: " << result.size();
      }
      cudaFree(cuda_key);
      cudaFree(cuda_value);
      cudaFree(cuda_key_reduced);
      cudaFree(cuda_value_reduced);
      return make_reduce_result();
    }

  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    std::tuple<KeyType*, ValueType*, int*, size_t> PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::make_reduce_result() {
      size_t n = result.size();
      
      res_key = (KeyType*)realloc(res_key, n * sizeof(KeyType));
      res_val = (ValueType*)realloc(res_val, n * sizeof(ValueType));
      res_cnt = (int*)realloc(res_cnt, n * sizeof(int));
      int i = 0;
      /*
      for (auto it = count.begin(); it != count.end(); it++) {
        DLOG(INFO) << "(" << it->first << ", " << it->second << ")";
      }
      */


      for (auto it = result.begin(); it != result.end(); ++it) {
        res_key[i] = it->first;
        res_val[i] = it->second;
        int cnt = count[it->first];
        res_cnt[i] = cnt; 
        //DLOG(INFO) << "Key: " << it->first << ", cnt: " << res_cnt[i] << "|" << cnt<< "|" << count[it->first];
        i++;
      }
      return std::make_tuple(res_key, res_val, res_cnt, n);
    }

  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    template <typename BinaryOp>
    void PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::merge(
        KeyType *cuda_key, 
        ValueType* cuda_value, 
        KeyType *base_key,
        int partition_size,
        int N, 
        BinaryOp f) {
      thrust::device_ptr<KeyType> key_ptr = thrust::device_pointer_cast<KeyType>(base_key);
      KeyType key[N];
      ValueType value[N];
      cudaMemcpy(key, cuda_key, N * sizeof(KeyType), cudaMemcpyDeviceToHost);
      cudaMemcpy(value, cuda_value, N * sizeof(ValueType), cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; i++) {
        KeyType thiskey = key[i];
        ValueType thisvalue = value[i];

        auto got = result.find(thiskey);
        auto got_cnt = count.find(thiskey);
        int c = thrust::count(key_ptr, key_ptr + partition_size, thiskey);
        if (got == result.end()) {
          result[thiskey] = thisvalue;
        } else {
          result[thiskey] = f(got->second, thisvalue); 
        }

        if (got_cnt == count.end()) {
          count[thiskey] = c;
        } else {
          count[thiskey] = got_cnt->second + c;
        }

      }

    }

  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    template <typename BinaryPred, typename BinaryOp>
    std::tuple<KeyType*, ValueType*, int*, size_t> PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::ReduceByKey(BinaryPred pred, BinaryOp f) {
      switch (this->memLevel_) {
        case Host:
        case Cuda:
          //return PipeLine<AfterType>::Reduce(f);
          DLOG(INFO) << "Undefined behavior!!\n";
          exit(1);
        case None:
          return this->ReduceByKey_(pred, f);
        default:
          DLOG(INFO) << "Undefined Mem level!!\n";
          exit(1);
      }
    }


  /*
     template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
     AfterType * 
     PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::GetData() {
     return PipeLine<AfterType>::GetData();
     }
     */

  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    Context *
    PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::GetContext() {
      return context_;
    }

  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    uint32_t PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::GetDataSize() {
      return size_;
    }


  /*
     template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
     void PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::Materialize(MemLevel ml, bool hard_materialized) {
     uint32_t total_materialized_size = this->size_ * sizeof(BaseType);      
     switch(ml) {
     case Host: {
     this->hard_materialized_ = hard_materialized;
     DLOG(INFO) << "Calling to materialize **mapped** pipeline to host " << ml << ", using data "
     << (total_materialized_size / (1024 * 1024)) << "MB";
     this->materialized_data_ = new AfterType[this->size_];
     this->Materialize_(ml);
     break;
     } case Cuda: {
     this->hard_materialized_ = hard_materialized;
     DLOG(INFO) << "Calling to materialize **mapped** pipeline to cuda " << ml << ", using data "
     << (total_materialized_size / (1024 * 1024)) << "MB";
     if(this->context_->addUsage(total_materialized_size) < 0){
     DLOG(FATAL) << "Over allocating GPU Memory, Program terminated";
     exit(1);
     }
     cudaMalloc((void**)&(this->materialized_data_), total_materialized_size);
     this->Materialize_(ml);
     break;
     } case None: {
     this->hard_materialized_ = false;
     PipeLine<AfterType>::Materialize(None, hard_materialized);
     break;
     }
     }
     this->memLevel_ = ml;
     }

     template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
     void PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::Materialize_(MemLevel ml) {
     uint32_t partition_size = std::min((this->context_->getUsableMemory() 
     / this->GetMaxUnitMemory_()), this->size_);
     uint32_t num_partitions = (this->size_ + partition_size - 1)/partition_size;
     DLOG(INFO) << "Materializing Map, with " << num_partitions 
     << " partitions, each dealing with " << partition_size << " size of data";

// Allocating the space for a single partition to hold
AfterType* cuda_after;
cudaMalloc((void**)&cuda_after, partition_size * sizeof(AfterType));
// do this on each fo the iterations
for(uint32_t i = 0; i < num_partitions; i++){
uint32_t partition_start = i * partition_size;
uint32_t partition_end = std::min(this->size_, (i+1) * partition_size);
uint32_t this_partition_size = partition_end - partition_start;
DLOG(INFO) << "Mapping, partition #" << i << ", size: "<< this_partition_size;
this->Map_Partition_(cuda_after, partition_start, this_partition_size);

// Materialize this chunk of data according to the MemLevel
switch(ml) {
case Host: {
cudaMemcpy(this->materialized_data_ + partition_start, cuda_after, 
this_partition_size * sizeof(AfterType), cudaMemcpyDeviceToHost);
} case Cuda: {
cudaMemcpy(this->materialized_data_ + partition_start, cuda_after, 
this_partition_size * sizeof(AfterType), cudaMemcpyDeviceToDevice);
}
}
}
cudaFree(cuda_after);
}
*/

// Map the partiton to a cuda memory address
template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
void PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::Map_Partition_(KeyType* cuda_key, ValueType* cuda_value, uint32_t partition_start, uint32_t this_partition_size) {
  BaseType* cuda_base = parent_->GetPartition_(partition_start, this_partition_size);
  //thrust::device_ptr<BaseType> base_ptr = thrust::device_pointer_cast(cuda_base);
  //thrust::device_ptr<KeyType> key_ptr = thrust::device_pointer_cast(cuda_key);
  //thrust::device_ptr<ValueType> value_ptr = thrust::device_pointer_cast(cuda_value);

  this->context_->pairTransform(cuda_base, cuda_base + this_partition_size, cuda_key, cuda_value, f_);

  parent_->DisposePartition_(cuda_base);
}

template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
uint32_t PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::GetMaxUnitMemory_() {
  return std::max(parent_->GetMaxUnitMemory_(), 2 * (uint32_t)(sizeof(KeyType) + sizeof(ValueType)));
}

/*
   template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
   AfterType* PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp>::GetPartition_(uint32_t partition_start, uint32_t this_partition_size){
//Retrieve the partition according to the memLevel 
switch(this->memLevel_){
case Host:
case Cuda: 
return PipeLine<AfterType>::GetPartition_();
case None: {
AfterType* partition_data;
cudaMalloc((void**)&partition_data, this_partition_size * sizeof(AfterType));
this->Map_Partition_(partition_data, partition_start, this_partition_size);
return partition_data;
}
}
}
*/
}

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
