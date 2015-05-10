#ifndef CUSPARK_TEXTPIPELINE_PIPELINE_H_
#define CUSPARK_TEXTPIPELINE_PIPELINE_H_

#include <iostream>
#include <fstream>
#include <string>
#include "common/logging.h"
#include "common/util.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include "common/CycleTimer.h"
#include "common/context.h"
#include "common/event_timer.h"

namespace cuspark {

  template <typename AfterType, typename BaseType, typename UnaryOp>
    class MappedPipeLine;

  template <typename KeyType, typename ValueType, typename BaseType, typename UnaryOp>
    class PairedPipeLine;

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

      template <typename KeyType, typename ValueType, typename UnaryOp>
        PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp> Map(UnaryOp f);

      // Interface for calling reduce function
      template <typename BinaryOp>
        BaseType Reduce(BinaryOp f);

      // Inner handler of reduce function
      template <typename BinaryOp>
        BaseType Reduce_Partition_(BaseType* cuda_data, uint32_t partition_size, BinaryOp f);

      BaseType *Take(uint32_t N);

      uint32_t GetDataSize();

      BaseType *GetData();

      Context *GetContext();

      // This is a function that may be used by both the user and the inner functions
      // So there's a bool value to determine if it is from user(in that case we'll perform more cautious eviction)
      void Materialize(MemLevel ml, bool hard_materialized = true);

      void ReadFile_(BaseType* mem_data);

      // this function return the max unit memory that is used in a chain
      // used when counting the size of a partition when allocating data
      uint32_t GetMaxUnitMemory_();

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
      BaseType* GetPartition_(uint32_t partition_start, uint32_t this_partition_size);

      void DisposePartition_(BaseType* partition_data){
        if(!(this->memLevel_ == Cuda)){
          cudaFree(partition_data);
        }
      }

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
    template <typename KeyType, typename ValueType, typename UnaryOp>
    PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp> PipeLine<BaseType>::Map(UnaryOp f) {
      PairedPipeLine<KeyType, ValueType, BaseType, UnaryOp> a(this, f);
      return a;
    }

  template <typename BaseType>
    template <typename BinaryOp>
    BaseType PipeLine<BaseType>::Reduce_Partition_(BaseType* cuda_data, uint32_t partition_size, BinaryOp f) {
      thrust::device_ptr<BaseType> dptr = thrust::device_pointer_cast(cuda_data);
      // Use the first element as the inital value
      BaseType initvalue;
      cudaMemcpy(&initvalue, cuda_data, sizeof(BaseType), cudaMemcpyDeviceToHost);
      // Execute reduce on this chunk using thrust
      BaseType thisres = thrust::reduce(dptr + 1, dptr + partition_size, initvalue, f);
      return thisres;
    }

  template <typename BaseType>
    template <typename BinaryOp>
    BaseType PipeLine<BaseType>::Reduce(BinaryOp f) {
      BaseType result;
      uint32_t partition_start;
      uint32_t partition_end;
      uint32_t this_partition_size;
      uint32_t partition_size;
      uint32_t num_partitions;
      switch (this->memLevel_) {
        case Host: 
          partition_size = std::min((context_->getUsableMemory() 
                / this->GetMaxUnitMemory_()), size_);
          num_partitions = (size_ + partition_size - 1)/partition_size;
          DLOG(INFO) << "Executing Reduce From Host Memory, with " << num_partitions << " partitions, each dealing with " << partition_size << " size of data";

          // Allocate the space for a single partition to hold
          BaseType *cuda_data;
          cudaMalloc((void**)&cuda_data, partition_size * sizeof(BaseType));

          // do this on each of the partitions
          for (int i = 0; i < num_partitions; i++) {
            partition_start = i * partition_size;
            partition_end = std::min(size_, (i+1) * partition_size);
            this_partition_size = partition_end - partition_start;
            DLOG(INFO) << "Reducing, partition #" << i << ", size: "<< this_partition_size;
            cudaMemcpy(cuda_data, this->materialized_data_ + partition_start, this_partition_size * sizeof(BaseType), cudaMemcpyHostToDevice);
            BaseType partition_result = this->Reduce_Partition_(cuda_data, this_partition_size, f);
            // update the result according to which batch this is handling
            if (i == 0) {
              result = partition_result;
            } else {
              result = f(result, partition_result);
            }
          }
          cudaFree(cuda_data);
          break; 
        case Cuda: 
          DLOG(INFO) << "Executing Reduce From Cuda Memory\n";
          result = this->Reduce_Partition_(this->materialized_data_, this->size_, f);
          break;
        case None:
          this->Materialize(Host, false);
          result = this->Reduce(f);
          this->Materialize(None);
          break;
      }
      return result;
    }

  template <typename BaseType>
    void PipeLine<BaseType>::ReadFile_(BaseType* mem_data){
      DLOG(INFO) << "Reading File to memory: " << filename_;
      double start = CycleTimer::currentSeconds();
      std::ifstream infile;
      infile.open(filename_);
      std::string line;
      int count = 0;
      while (std::getline(infile, line) && count < size_) {
        mem_data[count++] = f_(line);
      }
      double end = CycleTimer::currentSeconds();
      DLOG(INFO) << "Reading File finished. time took: " << (end - start) * 1000 << " ms";
    }

  template <typename BaseType>
    void PipeLine<BaseType>::Materialize(MemLevel ml, bool hard_materialized) {
      uint32_t total_materialized_size = this->size_ * sizeof(BaseType);
      BaseType* mem_data;
      switch (ml) {
        case Host: 
          this->hard_materialized_ = hard_materialized;
          DLOG(INFO) << "Calling to materialize pipeline to host " << ml << ", using data "
            << (total_materialized_size / (1024 * 1024)) << "MB";
          this->materialized_data_ = new BaseType[this->size_];
          ReadFile_(materialized_data_);
          break;
        case Cuda: 
          this->hard_materialized_ = hard_materialized;
          DLOG(INFO) << "Calling to materialize pipeline to cuda " << ml << ", using data "
            << (total_materialized_size / (1024 * 1024)) << "MB";
          if(this->context_->addUsage(total_materialized_size) < 0){
            DLOG(FATAL) << "Over allocating GPU Memory, Program terminated";
            exit(1);
          }
          mem_data = new BaseType[this->size_];
          ReadFile_(mem_data);
          cudaMalloc((void**)&materialized_data_, total_materialized_size);
          cudaMemcpy(materialized_data_, mem_data, total_materialized_size, cudaMemcpyHostToDevice);
          delete mem_data;
          break;
        case None: 
          this->hard_materialized_ = false;
          DLOG(INFO) << "Calling to freeing materialized pipeline  from " << ml << ", releasing data "
            << (total_materialized_size / (1024 * 1024)) << "MB";
          switch (this->memLevel_) {
            case Host:
              delete this->materialized_data_;
              this->materialized_data_ = nullptr;
              break;
            case Cuda:
              if(hard_materialized) cudaFree(this->materialized_data_);
              this->materialized_data_ = nullptr;
              this->context_->reduceUsage(total_materialized_size);
              break;
          }
      }
      this->memLevel_ = ml;
    }

  template <typename BaseType>
    BaseType* PipeLine<BaseType>::GetPartition_(uint32_t partition_start, uint32_t this_partition_size){
      BaseType* partition_data;
      EventTimer et;
      // Retrieve the partition according to the memLevel
      double start = CycleTimer::currentSeconds();
      et.start();
      switch (this->memLevel_) {
        case Host: 
          cudaMalloc((void**)&partition_data, this_partition_size * sizeof(BaseType));
          cudaMemcpy(partition_data, this->materialized_data_ + partition_start, this_partition_size * sizeof(BaseType), cudaMemcpyHostToDevice);
          // If we've got to the last partition, just clean the mess we just made
          if(partition_start + this_partition_size == this->size_ && this->hard_materialized_ == false){
            this->Materialize(None);
          }
          break;
        case Cuda: 
          partition_data = this->materialized_data_ + partition_start;
          break;
        case None: 
          if (partition_start == 0 && partition_start + this_partition_size == this->size_) {
            // it fits in cuda global memory, just materialize it into cuda
            this->Materialize(Cuda, false);
            partition_data = this->GetPartition_(partition_start, this_partition_size);
            this->Materialize(None, false); 
          } else {
            this->Materialize(Host, false);
            partition_data = this->GetPartition_(partition_start, this_partition_size);
          }
      }
      et.stop();
      double end = CycleTimer::currentSeconds();
      DLOG(INFO) << "GetPartion of size " << this_partition_size << ", time took " << et.elapsed() << "|" << (end-start) * 1000 << "ms.";
      return partition_data;
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
      BaseType* returned_data = new BaseType[this->size_];
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
    BaseType *PipeLine<BaseType>::Take(uint32_t N) {
      BaseType* returned_data = new BaseType[N];
      switch (this->memLevel_) {
        case Host:
          memcpy(returned_data, this->materialized_data_, N * sizeof(BaseType));
          break;
        case None:
          this->Materialize(Host, false);
          memcpy(returned_data, this->materialized_data_, N * sizeof(BaseType));
          this->Materialize(None);
          break;
        case Cuda:
          cudaMemcpy(returned_data, this->materialized_data_, N * sizeof(BaseType), cudaMemcpyDeviceToHost);
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
    uint32_t PipeLine<BaseType>::GetMaxUnitMemory_() {
      return sizeof(BaseType);
    }
}

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
