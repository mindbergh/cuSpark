#ifndef CUSPARK_PIPELINE_MAPPEDPIPELINE_H_
#define CUSPARK_PIPELINE_MAPPEDPIPELINE_H_

#include <stdio.h>
#include "common/logging.h"
#include "common/util.h"
#include "pipeline/pipeline.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include "common/CycleTimer.h"

namespace cuspark {

  /*
   * Mapped from type BaseType to type AfterType
   */
  template <typename AfterType, typename BaseType, typename UnaryOp>
    class MappedPipeLine : public PipeLine<AfterType> {

      public:
        MappedPipeLine(PipeLine<BaseType> *parent, UnaryOp f);

        //template <typename W>
        //MappedPipeLine<W, AfterType> Map(unaryOp f);

        template <typename BinaryOp> AfterType Reduce(BinaryOp f);

        template <typename BinaryOp> AfterType Reduce_(BinaryOp f);

        // return the data in memory
        AfterType *GetData();

        Context *GetContext();

        uint32_t GetDataSize();

        void Materialize(MemLevel ml, bool hard_materialized = true);

        void Materialize_(MemLevel ml);

        //this function return the max unit memory that is used in a chain
        // used when counting the size of a partition when allocating data
        uint32_t GetMaxUnitMemory_();

        void Map_Partition_(AfterType* cuda_after, uint32_t partition_start, uint32_t this_partition_size);

        // functor of the map operation
        UnaryOp f_;

        PipeLine<BaseType> *parent_;

        AfterType* GetPartition_(uint32_t partition_start, uint32_t this_partition_size);
    };

  template <typename AfterType, typename BaseType, typename UnaryOp>
    MappedPipeLine<AfterType, BaseType, UnaryOp>::
    MappedPipeLine(PipeLine<BaseType> *parent, UnaryOp f)
    : PipeLine<AfterType>(parent->GetDataSize(), parent->GetContext()),
    parent_(parent),
    f_(f) {}

  template <typename AfterType, typename BaseType, typename UnaryOp>
    template <typename BinaryOp>
    AfterType MappedPipeLine<AfterType, BaseType, UnaryOp>::Reduce_(BinaryOp f) {

      AfterType result;
      uint32_t partition_size = std::min((this->context_->getUsableMemory() 
          / this->GetMaxUnitMemory_()), this->size_);
      uint32_t num_partitions = (this->size_ + partition_size - 1)/partition_size;
      DLOG(INFO) << "Reducing from mapped pipeline, with " << num_partitions << " partitions, each dealing with " << partition_size << " size of data";

      // Allocating the space for a single partition to hold
      AfterType* cuda_after;
      cudaMalloc((void**)&cuda_after, partition_size * sizeof(BaseType));

      // do this on each of the partitions
      for(uint32_t i = 0; i < num_partitions; i++){
        uint32_t partition_start = i * partition_size;
        uint32_t partition_end = std::min(this->size_, (i+1) * partition_size);
        uint32_t this_partition_size = partition_end - partition_start;
        DLOG(INFO) << "Reducing, partition #" << i << ", size: "<< this_partition_size;
        this->Map_Partition_(cuda_after, partition_start, this_partition_size);    
        AfterType partition_result = this->Reduce_Partition_(cuda_after, this_partition_size, f);

        //update the result according to which partition this is handling
        if(i == 0){
          result = partition_result;
        } else {
          result = f(result, partition_result);
        }
      }
      cudaFree(cuda_after);
      return result;
    }
  
  template <typename AfterType, typename BaseType, typename UnaryOp>
    template <typename BinaryOp>
    AfterType MappedPipeLine<AfterType, BaseType, UnaryOp>::Reduce(BinaryOp f) {
      switch (this->memLevel_) {
        case Host:
        case Cuda:
          return PipeLine<AfterType>::Reduce(f);
        case None:
          return this->Reduce_(f);
      }
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    AfterType * 
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetData() {
      return PipeLine<AfterType>::GetData();
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    Context *
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetContext() {
      return PipeLine<AfterType>::GetContext();
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    uint32_t MappedPipeLine<AfterType, BaseType, UnaryOp>::GetDataSize() {
      return PipeLine<AfterType>::GetDataSize();
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    void MappedPipeLine<AfterType, BaseType, UnaryOp>::Materialize(MemLevel ml, bool hard_materialized) {
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

  template <typename AfterType, typename BaseType, typename UnaryOp>
    void MappedPipeLine<AfterType, BaseType, UnaryOp>::Materialize_(MemLevel ml) {
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

  // Map the partiton to a cuda memory address
  template <typename AfterType, typename BaseType, typename UnaryOp>
    void MappedPipeLine<AfterType, BaseType, UnaryOp>::Map_Partition_(AfterType* cuda_after, uint32_t partition_start, uint32_t this_partition_size) {
      BaseType* cuda_base = parent_->GetPartition_(partition_start, this_partition_size);
      thrust::device_ptr<BaseType> base_ptr = thrust::device_pointer_cast(cuda_base);
      thrust::device_ptr<AfterType> after_ptr = thrust::device_pointer_cast(cuda_after);
      thrust::transform(base_ptr, base_ptr + this_partition_size, after_ptr, f_);
      parent_->DisposePartition_(cuda_base);
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    uint32_t MappedPipeLine<AfterType, BaseType, UnaryOp>::GetMaxUnitMemory_() {
      return std::max(parent_->GetMaxUnitMemory_(), (uint32_t)(sizeof(AfterType) + sizeof(BaseType)));
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    AfterType* MappedPipeLine<AfterType, BaseType, UnaryOp>::GetPartition_(uint32_t partition_start, uint32_t this_partition_size){
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
}

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
