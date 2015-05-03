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

namespace cuspark {

  template <typename AfterType, typename BaseType, typename UnaryOp>
    class MappedPipeLine;

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

      template <typename AfterType, typename UnaryOp>
        MappedPipeLine<AfterType, BaseType, UnaryOp> Map(UnaryOp f);

      // Interface for calling reduce function
      template <typename BinaryOp>
        BaseType Reduce(BinaryOp f);

      // Inner handler of reduce function
      template <typename BinaryOp>
        BaseType Reduce_(BaseType* cuda_data, int batch_size, BinaryOp f);

      uint32_t GetDataSize();

      BaseType *GetData();

      BaseType GetElement(uint32_t index);

      BaseType GetElement_(uint32_t index);

      Context *GetContext();

      void MallocCudaData();

      void FreeCudaData();

      void ReadFile_(BaseType* mem_data);

      void Materialize(MemLevel ml);

      BaseType* data_; //pointer to the array, raw ptr in CUDA
      uint32_t size_; //the length of the data array
      InputMapOp f_;
      std::string filename_;
      Context *context_;
      MemLevel memLevel;
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
    memLevel(None) {}

  template <typename BaseType>
    PipeLine<BaseType>::PipeLine(uint32_t size, Context *context)
    : size_(size),
    context_(context),
    memLevel(None) {
      DLOG(INFO) << "Construct PipeLine by size and context: " << size;
    }

  template <typename BaseType>
    template <typename AfterType, typename UnaryOp>
    MappedPipeLine<AfterType, BaseType, UnaryOp> PipeLine<BaseType>::Map(UnaryOp f) {
      MappedPipeLine<AfterType, BaseType, UnaryOp> a(this, f);
      return a;
    }

  template <typename BaseType>
    template <typename BinaryOp>
    BaseType PipeLine<BaseType>::Reduce_(BaseType* cuda_data, int batch_size, BinaryOp f) {
      thrust::device_ptr<BaseType> dptr = thrust::device_pointer_cast(cuda_data);
      // Use the first element as the inital value
      BaseType initvalue;
      cudaMemcpy(&initvalue, cuda_data, sizeof(BaseType), cudaMemcpyDeviceToHost);
      // Execute reduce on this chunk using thrust
      BaseType thisres = thrust::reduce(dptr + 1, dptr + batch_size, initvalue, f);
      return thisres;
    }

  template <typename BaseType>
    template <typename BinaryOp>
    BaseType PipeLine<BaseType>::Reduce(BinaryOp f) {
      DLOG(INFO) << "Executing Reduce\n";
      BaseType result, batch_result;
      size_t processed = 0;
      // to do: need current available mem
      size_t maxBatch = context_->getTotalGlobalMem() / sizeof(BaseType);

      switch (this->memLevel) {
        case Host:
          BaseType *cuda_data;
          if (size_ > maxBatch)
            cudaMalloc((void**)&cuda_data, maxBatch * sizeof(BaseType));
          else
            cudaMalloc((void**)&cuda_data, size_ * sizeof(BaseType));
          while (processed < size_) {
            size_t thisBatch = std::min(maxBatch, size_ - processed);
            DLOG(INFO) << "Reducing, This batch: " << thisBatch;
            cudaMemcpy(cuda_data, this->data_ + processed, thisBatch * sizeof(BaseType), cudaMemcpyHostToDevice);
            batch_result = this->Reduce_(cuda_data, thisBatch, f);
            // update the result according to which batch this is handling
            if(processed == 0)
              result = batch_result;
            else
              result = f(result, batch_result);

            processed += thisBatch;
          }
          return result;
        case Cuda:
          this->Reduce_(this->data_, this->size_, f);
        case None:
          this->Materialize(Host);
          return this->Reduce(f);
        default:
          DLOG(FATAL) << "memLevel type not correct";
          exit(1);
      }
    }

  template <typename BaseType>
    void PipeLine<BaseType>::ReadFile_(BaseType* mem_data){
      std::ifstream infile;
      infile.open(filename_);
      std::string line;
      int count = 0;
      while (std::getline(infile, line) && count < size_) {
        data_[count++] = f_(line);
      }
    }

  template <typename BaseType>
    void PipeLine<BaseType>::Materialize(MemLevel ml) {
      BaseType *mem_data;
      switch (ml) {
        case Host:
          this->memLevel = Host;
          data_ = new BaseType[size_];
          ReadFile_(data_);
          return;
        case Cuda: 
          this->memLevel = Cuda;
          mem_data = new BaseType[size_];
          ReadFile_(mem_data);
          cudaMalloc((void**)&data_, size_ * sizeof(BaseType));
          cudaMemcpy(data_, mem_data, size_ * sizeof(BaseType), cudaMemcpyHostToDevice);
          delete mem_data;
          return;
        case None:
          switch (this->memLevel) {
            case Host:
              delete this->data_;
              this->data_ = nullptr;
              break;
            case Cuda:
              cudaFree(this->data_);
              this->data_ = nullptr;
              break;
            case None:
              break;
            default:
              break;
          }
        default:
          return; 
      }
    }


  template <typename BaseType>
    uint32_t PipeLine<BaseType>::GetDataSize(){
      DLOG(INFO) << "Get Datasize";
      return this->size_;
    }
  template <typename BaseType>
    BaseType * PipeLine<BaseType>::GetData(){
      DLOG(INFO) << "Getting data from address: " << data_;
      return this->data_;
    }

  template <typename BaseType>
    Context * PipeLine<BaseType>::GetContext() {
      DLOG(INFO) << "Get Context";
      return this->context_;
    }

  template <typename BaseType>
    BaseType PipeLine<BaseType>::GetElement(uint32_t index){
      return GetElement_(index);
    }

  template <typename BaseType>
    BaseType PipeLine<BaseType>::GetElement_(uint32_t index){
      BaseType element;
      cudaMemcpy(&element, this->data_ + index, sizeof(BaseType), cudaMemcpyDeviceToHost);
      return element;
    }

  template <typename BaseType>
    void PipeLine<BaseType>::MallocCudaData(){
      DLOG(INFO) << "malloc GPU memory for data with size :" << sizeof(BaseType) << " * " << size_;
      cudaMalloc((void**)&data_, size_ * sizeof(BaseType));
    }

  template <typename BaseType>
    void PipeLine<BaseType>::FreeCudaData(){
      DLOG(INFO) << "freeing GPU memory for data with size :" << sizeof(BaseType) << " * " << size_;
      cudaFree(data_);
      data_ = NULL;
    }
}






#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
