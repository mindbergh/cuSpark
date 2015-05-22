#ifndef CUSPARK_TEXTPIPELINE_PIPELINE_H_
#define CUSPARK_TEXTPIPELINE_PIPELINE_H_

#include <iostream>
#include <fstream>
#include <string>
#include "common/logging.h"
#include "common/util.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include "common/CycleTimer.h"
#include "common/context.h"
#include "common/partition.h"
#include <mutex>

namespace cuspark {

  template <typename AfterType, typename BaseType, typename UnaryOp>
    class MappedPipeLine;

  template <typename BaseType, typename BinaryOp>
  struct ReduceWorkerArgs{
    BaseType* reduce_result_;
    PartitionWorkQueue* queue_;
    BinaryOp f_;
    PipeLine<BaseType>* pl_;
  };

  template <typename BaseType, typename BinaryOp>
    void* ReducePartitionWorker(void* threadArgs) {
      ReduceWorkerArgs<BaseType, BinaryOp>* args = static_cast<ReduceWorkerArgs<BaseType, BinaryOp>*>(threadArgs);

      //create the cuda stream for this worker
      cudaStream_t stream;
      cudaStreamCreate(&stream);
  
      // generate the cuda data for this partition
      void* thread_data;
      cudaMalloc(&thread_data, args->queue_->partition_size_ * args->queue_->unit_size_);

      // Work on the partitions until everything is finished
      uint32_t finished_job = 0;
      PartitionInfo* partition = args->queue_->pop();
      while(partition != NULL){
        DLOG(INFO) << "Partition #" << partition->partition_id_ << " starts executing [Reduce] , size: "<< partition->this_partition_size_;
        partition->stream_ = stream;
        partition->data_ = thread_data;
 
        // Get raw data and do reduce
        void* cuda_data = args->pl_->GetPartition_(partition);
        BaseType temp_result = args->pl_->Reduce_Partition_(cuda_data, partition, args->f_);

        //update the partition result
        if(finished_job == 0){
          *(args->reduce_partition_result_) = temp_result;
        }else{
          *(args->reduce_partition_result_) = f_(temp_result, *(args->reduce_partition_result));
        }
    
        finished_job++;
        partition = args->queue_->pop();
      }
    
      cudaFree(thread_data);
      return NULL;
    }

  /*
   * Basic PipeLine class, which we generate from file or array
   * 
   */
  template <typename BaseType>
    class PipeLine {
      typedef BaseType (*InputMapOp)(const std::string&);

      public:
      PipeLine(std::string filename, 
          uint32_t size, InputMapOp f, Context *context);

      PipeLine(uint32_t size, Context *context);

      // Interface for calling map function
      template <typename AfterType, typename UnaryOp>
        MappedPipeLine<AfterType, BaseType, UnaryOp> Map(UnaryOp f);

      // Interface for calling reduce function
      template <typename BinaryOp>
        BaseType Reduce(BinaryOp f);

      // Inner handler of reduce function
      template <typename BinaryOp>
        BaseType Reduce_Partition_(void* cuda_data, PartitionInfo* partition, BinaryOp f);

      uint32_t GetDataSize();

      BaseType *GetData();

      Context *GetContext();
      
      // This is a function that may be used by both the user and the inner functions
      // So there's a bool value to determine if it is from user(in that case we'll perform more cautious eviction)
      void Materialize(MemLevel ml, bool hard_materialized = true);

      void ReadFile_(BaseType* mem_data);

      // this function return the max unit memory that is used in a chain
      // used when counting the size of a partition when allocating data
      uint32_t GetUnitMemory_();

      // total length of the data array
      uint32_t size_;

      // function to map from string(input file) to the data type
      InputMapOp f_;

      // Input file path
      std::string filename_;

      // Cuda context
      Context *context_;
  
      // Overall setting of the level of materialization
      // (1) None to be default
      // (2) Host so that everything is materialized in memory
      // (3) Cuda so that everything is materialized in cuda global memory
      MemLevel memLevel_;
   
      // pointer to the data array, in host or in cuda
      // It will only be set set if we materialied the data in cuda or host
      BaseType* materialized_data_;
   
      // This is an indicator whether this data is materialized upon user request
      bool hard_materialized_ = false;
    
      // This is an important inner function
      // Its primary functionality is to retrieve a pointer in cuda global memory
      // But it also addresses useful functionality in lazy execution
      void* GetPartition_(PartitionInfo* partition);

      void DisposePartition_(BaseType* partition_data){
        if(!(this->memLevel_ == Cuda)){
          cudaFree(partition_data);
        }
      }

      ///std::mutex temp_lock;
    };

  template <typename BaseType>
    PipeLine<BaseType>::PipeLine(std::string filename, 
        uint32_t size, 
        InputMapOp f,
        Context *context)
    : size_(size),
    f_(f),
    filename_(filename),
    context_(context),
    memLevel_(None) {
      DLOG(INFO) << "Construct PipeLine from file: " << size << " * " << sizeof(BaseType);
    }

  template <typename BaseType>
    PipeLine<BaseType>::PipeLine(uint32_t size, Context *context)
    : size_(size),
    context_(context),
    memLevel_(None) {
      DLOG(INFO) << "Construct PipeLine by size and context: " << size << " * " << sizeof(BaseType);
    }

  template <typename BaseType>
    template <typename AfterType, typename UnaryOp>
    MappedPipeLine<AfterType, BaseType, UnaryOp> PipeLine<BaseType>::Map(UnaryOp f) {
      MappedPipeLine<AfterType, BaseType, UnaryOp> a(this, f);
      return a;
    }

  template <typename BaseType>
    template <typename BinaryOp>
    BaseType PipeLine<BaseType>::Reduce_Partition_(void* cuda_data, PartitionInfo* partition, BinaryOp f) {
      thrust::device_ptr<BaseType> dptr = thrust::device_pointer_cast((BaseType*)cuda_data);
      DLOG(INFO) << "Reduce Pointer cast finish, partition #" << partition->partition_id_ << ", size: "<< partition->this_partition_size_ << ", start: " << partition->partition_start_;
      // Use the first element as the inital value
      BaseType initvalue;
      cudaMemcpyAsync(&initvalue, cuda_data, sizeof(BaseType), cudaMemcpyDeviceToHost, partition->stream_);
      // Execute reduce on this chunk using thrust
      
      DLOG(INFO) << "Reduce Starts, partition #" << partition->partition_id_ << ", size: "<< partition->this_partition_size_ << ", start: " << partition->partition_start_;
      BaseType thisres = thrust::reduce(thrust::cuda::par.on(partition->stream_), dptr+1, dptr + partition->this_partition_size_, initvalue, f);
      DLOG(INFO) << "Reduce Ends, partition #" << partition->partition_id_ << ", size: "<< partition->this_partition_size_ << ", start: " << partition->partition_start_;
      return thisres;
    }

  template <typename BaseType>
    template <typename BinaryOp>
    BaseType PipeLine<BaseType>::Reduce(BinaryOp f) {
      BaseType result;

      switch (this->memLevel_) {
        case Host: {
    
          // determine the partition size
          uint32_t unit_memory = this->GetUnitMemory_();
          uint32_t partition_size = std::min(
              (this->context_->getUsableMemory() / unit_memory / THREAD_POOL_SIZE) , this->size_);
          PartitionWorkQueue queue(partition_size, this->size_, unit_memory);
          DLOG(INFO) << "Executing Reduce From Host Memory, with " << queue.num_partitions_ << " partitions, each dealing with " << partition_size << " size of data";
 
          // initiate the threads to deal with each partition
          BaseType reduce_partition_result[THREAD_POOL_SIZE];
          pthread_t threads[THREAD_POOL_SIZE];
          ReduceWorkerArgs<BaseType, BinaryOp> threadArgs[THREAD_POOL_SIZE];

          // initiate and launch the thread pool
          for(uint32_t i = 0; i < THREAD_POOL_SIZE; i++){
            threadArgs[i].queue_ = &queue;
            threadArgs[i].f_ = f;
            threadArgs[i].pl_ = this;
            threadArgs[i].reduce_partition_result_ = &reduce_partition_result[i];
            pthread_create(&threads[i], NULL, ReducePartitionWorker<BaseType, BinaryOp>, &threadArgs[i]);
          }

          // wait for the threads to return
          for(uint32_t i = 0; i < THREAD_POOL_SIZE; i++) {
            pthread_join(threads[i], NULL);
          }

          // update the result according to which partition this is handling
          result = reduce_partition_result[0];
          for(uint32_t i = 1; i < THREAD_POOL_SIZE; i++){
            result = f(result, reduce_partition_result[i]);
          }
          break; 
        } case Cuda: {
          DLOG(INFO) << "Executing Reduce From Cuda Memory\n";
          result = this->Reduce_Partition_(this->materialized_data_, NULL, f);
          break;
        } case None: {
          this->Materialize(Host, false);
          result = this->Reduce(f);
          this->Materialize(None);
          break;
        }
      }
      return result;
    }

  template <typename BaseType>
    void PipeLine<BaseType>::ReadFile_(BaseType* mem_data){
      DLOG(INFO) << "Reading File to memory: " << filename_;
      std::ifstream infile;
      infile.open(filename_);
      std::string line;
      int count = 0;
      while (std::getline(infile, line) && count < size_) {
        mem_data[count++] = f_(line);
      }
    }

  template <typename BaseType>
    void PipeLine<BaseType>::Materialize(MemLevel ml, bool hard_materialized) {
      uint32_t total_materialized_size = this->size_ * sizeof(BaseType);
      switch(ml){
        case Host: {
          this->hard_materialized_ = hard_materialized;
          DLOG(INFO) << "Calling to materialize pipeline to host " << ml << ", using data "
              << (total_materialized_size / (1024 * 1024)) << "MB";
          //this->materialized_data_ = new BaseType[this->size_];
          DLOG(INFO) << sizeof(BaseType) << ", "<< this->size_;
          cudaError_t status = cudaMallocHost((void**)&(this->materialized_data_), sizeof(BaseType) * this->size_);
          DLOG(INFO) << "last CUDA error: " << cudaGetErrorString(status);
          //this->materialized_data_ = new BaseType[sizeof(BaseType) * this->size_];
          ReadFile_(materialized_data_);
          break;
        } case Cuda: {
          this->hard_materialized_ = hard_materialized;
          DLOG(INFO) << "Calling to materialize pipeline to cuda " << ml << ", using data "
              << (total_materialized_size / (1024 * 1024)) << "MB";
          int left_memory = this->context_->addUsage(total_materialized_size); 
          if(left_memory < 0){
            DLOG(FATAL) << "Over allocating GPU Memory, now left " << left_memory << ", Program terminated";
            exit(1);
          }
          BaseType* mem_data; // = new BaseType[this->size_];
          cudaMallocHost((void**)&mem_data, sizeof(BaseType) * this->size_);
          ReadFile_(mem_data);
          cudaMalloc((void**)&materialized_data_, total_materialized_size);
          cudaMemcpy(materialized_data_, mem_data, total_materialized_size, cudaMemcpyHostToDevice);
          cudaFreeHost(mem_data);
          break;
        } case None: {
          this->hard_materialized_ = false;
          DLOG(INFO) << "Calling to freeing materialized pipeline  from " << ml << ", releasing data "
              << (total_materialized_size / (1024 * 1024)) << "MB";
          switch (this->memLevel_){
            case Host:
              cudaFreeHost(this->materialized_data_);
              this->materialized_data_ = nullptr;
              break;
            case Cuda:
              if(hard_materialized) cudaFree(this->materialized_data_);
              this->materialized_data_ = nullptr;
              this->context_->reduceUsage(total_materialized_size);
              break;
          }
        }
      }
      this->memLevel_ = ml;
    }
  
  template <typename BaseType>
    void* PipeLine<BaseType>::GetPartition_(PartitionInfo* partition){
      // Retrieve the partition according to the memLevel
      switch(this->memLevel_){
        case Host: {
          // copy the partition to the data chunk
          //temp_lock.lock();
          cudaMemcpyAsync(partition->data_, this->materialized_data_ + partition->partition_start_, partition->this_partition_size_ * sizeof(BaseType), cudaMemcpyHostToDevice, partition->stream_);
          //temp_lock.unlock();
          partition->used_data_size_ += partition->this_partition_size_ * sizeof(BaseType);

          // If we've got to the last partition, just clean the mess we just made
          if(partition->partition_start_ + partition->this_partition_size_ == this->size_ && this->hard_materialized_ == false){
            this->Materialize(None);
          }
          return partition->data_;
        } case Cuda: {
          cudaMemcpyAsync(partition->data_, this->materialized_data_ + partition->partition_start_, partition->this_partition_size_ * sizeof(BaseType), cudaMemcpyDeviceToDevice, partition->stream_);
          partition->used_data_size_ += partition->this_partition_size_ * sizeof(BaseType);
          return partition->data_;
        } case None: {
          void* partition_data;
          if(partition->partition_start_ == 0 && partition->partition_start_ + partition->this_partition_size_ == this->size_){
            // it fits in cuda global memory, just materialize it into cuda
            this->Materialize(Cuda, false);
            partition_data = this->GetPartition_(partition);
            this->Materialize(None, false); 
          } else {
            this->Materialize(Host, false);
            partition_data = this->GetPartition_(partition);
          }
          return partition_data;
        } default: {
          DLOG(FATAL) << "memLevel type not correct, exiting";
          exit(1);
        }
      }
    }

  template <typename BaseType>
    uint32_t PipeLine<BaseType>::GetDataSize(){
      return this->size_;
    }

  // Get the data array for use
  // In case it interferes with the normal execution
  // I would simply malloc a new array and copy the data out
  template <typename BaseType>
    BaseType * PipeLine<BaseType>::GetData(){
      // Initiate the array for the returned data, this should fits in memory 
      BaseType* returned_data; // = new BaseType[this->size_];
      cudaMallocHost((void**)&returned_data, sizeof(BaseType) * this->size_);
      switch (this->memLevel_) {
        case Host:
          memcpy(returned_data, this->materialized_data_, this->size_ * sizeof(BaseType));
          break;
        case None:
          this->Materialize(Host, false);
          memcpy(returned_data, this->materialized_data_, this->size_ * sizeof(BaseType));
          this->Materialize(None);
          break;
        case Cuda:
          cudaMemcpy(returned_data, this->materialized_data_, this->size_ * sizeof(BaseType), cudaMemcpyDeviceToHost);
          break;
        default:
          DLOG(FATAL) << "memLevel type not correct";
          exit(1);
      }
      return returned_data;
    }

  template <typename BaseType>
    Context * PipeLine<BaseType>::GetContext() {
      return this->context_;
    }

  template <typename BaseType>
    uint32_t PipeLine<BaseType>::GetUnitMemory_() {
      if(this->memLevel_ == Cuda){
        return 0;
      }else{
        return sizeof(BaseType);
      }
    }
}

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
