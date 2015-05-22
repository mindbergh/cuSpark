#ifndef CUSPARK_PIPELINE_MAPPEDPIPELINE_H_
#define CUSPARK_PIPELINE_MAPPEDPIPELINE_H_

#include <stdio.h>
#include "common/logging.h"
#include "common/util.h"
#include "pipeline/pipeline.h"
#include <queue>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include "common/CycleTimer.h"
#include "common/partition.h"

namespace cuspark {

  // The arguements of the thread to do reduce over mapped pipeline
  template<typename AfterType, typename BaseType, typename BinaryOp, typename UnaryOp>
  struct MappedReduceWorkerArgs{
    AfterType* reduce_partition_result_;
    PartitionWorkQueue* queue_;
    BinaryOp f_;
    MappedPipeLine<AfterType, BaseType, UnaryOp>* mpl_;
  }; 

  // The thread to do reduce over mapped pipeline
  template <typename AfterType, typename BaseType, typename BinaryOp, typename UnaryOp>
  void* MappedReducePartitionWorker(void* threadArgs){

    MappedReduceWorkerArgs<AfterType, BaseType, BinaryOp, UnaryOp> args = *static_cast<MappedReduceWorkerArgs<AfterType, BaseType, BinaryOp, UnaryOp>*>(threadArgs);

    // create the cuda stream for this worker
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // generate the cuda data for this partition
    void* thread_data;
    cudaMalloc(&thread_data, args.queue_->partition_size_ * args.queue_->unit_size_);

    // Work on the partitions until everything is finished
    uint32_t finished_job = 0;
    PartitionInfo* partition = args.queue_->pop();
    while(partition != NULL){
      DLOG(INFO) << "Partition #" << partition->partition_id_ << " starts executing, size: "<< partition->this_partition_size_;
      partition->stream_ = stream;
      partition->data_ = thread_data;

      // do map and reduce over the partition
      void* mapped_result = (void*)args.mpl_->Map_Partition_(partition);
      AfterType temp_result = args.mpl_->Reduce_Partition_(mapped_result, partition, args.f_);

      // update the partition result
      if(finished_job == 0){
        *(args.reduce_partition_result_) = temp_result;
      }else{
        *(args.reduce_partition_result_) = args.f_(temp_result, *(args.reduce_partition_result_));
      }

      // get the next partition
      finished_job++;
      partition = args.queue_->pop();
    }

    // Clean the data we used
    cudaFree(thread_data);
    return NULL;
  }
      

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

        //this function return the unit memory that is used in a chain
        // used when counting the size of a partition when allocating data
        uint32_t GetUnitMemory_();

        void* Map_Partition_(PartitionInfo* partition);

        // functor of the map operation
        UnaryOp f_;

        PipeLine<BaseType> *parent_;

        AfterType* GetPartition_(PartitionInfo* partition);
  
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

      // determine the partition size
      uint32_t unit_memory = this->GetUnitMemory_();
      uint32_t partition_size = std::min(
          (this->context_->getUsableMemory() / unit_memory / THREAD_POOL_SIZE) , this->size_);
      //partition_size = std::min(partition_size, (uint32_t)(1000 * 1024));
      PartitionWorkQueue queue(partition_size, this->size_, unit_memory);
      DLOG(INFO) << "Reducing from mapped pipeline, with " << queue.num_partitions_ << " partitions, each dealing with " << partition_size << " size of data";

      // initiate the threads to deal with each partition
      AfterType reduce_partition_result[THREAD_POOL_SIZE];
      pthread_t threads[THREAD_POOL_SIZE];
      MappedReduceWorkerArgs<AfterType, BaseType, BinaryOp, UnaryOp> threadArgs[THREAD_POOL_SIZE];

      // initate and launch the thread pool
      for(uint32_t i = 0; i < THREAD_POOL_SIZE; i++){
        threadArgs[i].queue_ = &queue;
        threadArgs[i].f_ = f;
        threadArgs[i].mpl_ = this;
        threadArgs[i].reduce_partition_result_ = &reduce_partition_result[i];
        pthread_create(&threads[i], NULL, MappedReducePartitionWorker<AfterType, BaseType, BinaryOp, UnaryOp>, &threadArgs[i]);
      }

      // wait for the threads to return
      for(uint32_t i = 0; i < THREAD_POOL_SIZE; i++){
        pthread_join(threads[i], NULL);
      }

      //update the result according to which partition this is handling
      AfterType result = reduce_partition_result[0];
      for(uint32_t i = 1; i < THREAD_POOL_SIZE; i++){
        result = f(result, reduce_partition_result[i]);
      }
      return result;
    }
  
  template <typename AfterType, typename BaseType, typename UnaryOp>
    template <typename BinaryOp>
    AfterType MappedPipeLine<AfterType, BaseType, UnaryOp>::Reduce(BinaryOp f) {
      switch (this->memLevel_) {
        case Host:
        case Cuda:
        /*to do!!!! we have to fix this*/
          //return PipeLine<AfterType>::Reduce(f);
          AfterType result;
          return result ;
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
          //this->materialized_data_ = new AfterType[this->size_];
          cudaMallocHost((void**)&(this->materialized_data_), sizeof(AfterType) * this->size_);
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

  /*
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
    */

  // Map the partiton to a cuda memory address
  template <typename AfterType, typename BaseType, typename UnaryOp>
    void* MappedPipeLine<AfterType, BaseType, UnaryOp>::Map_Partition_(PartitionInfo* partition) {
      void* cuda_base = parent_->GetPartition_(partition);
      void* cuda_after = (char*)partition->data_ + partition->used_data_size_;
      thrust::device_ptr<BaseType> base_ptr = thrust::device_pointer_cast((BaseType*)cuda_base);
      thrust::device_ptr<AfterType> after_ptr = thrust::device_pointer_cast((AfterType*)cuda_after);

      DLOG(INFO) << "Map starts, partition #"<< partition->partition_id_ << ", size: "<<partition->this_partition_size_;
      thrust::transform(thrust::cuda::par.on(partition->stream_), base_ptr, base_ptr + partition->this_partition_size_, after_ptr, f_);
      DLOG(INFO) << "Map finishes, partition #"<< partition->partition_id_ << ", size: "<<partition->this_partition_size_;
      return cuda_after;
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    uint32_t MappedPipeLine<AfterType, BaseType, UnaryOp>::GetUnitMemory_() {
      if(this->memLevel_ != Cuda){
        return (uint32_t)sizeof(AfterType) + parent_->GetUnitMemory_();
      }else{
        return 0;
      }
    }

  /*
  template <typename AfterType, typename BaseType, typename UnaryOp>
    AfterType* MappedPipeLine<AfterType, BaseType, UnaryOp>::GetPartition_(PartitionWorkerArgs* partition){
      //Retrieve the partition according to the memLevel 
      switch(this->memLevel_){
        case Host:
        case Cuda: 
          return PipeLine<AfterType>::GetPartition_();
        case None: {
          AfterType* partition_data;
          cudaMalloc((void**)&partition_data, partition->this_partition_size_ * sizeof(AfterType));
          this->Map_Partition_(partition_data, partition->partition_start, partition->this_partition_size);
          return partition_data;
        }
      }
    }
    */
}

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
