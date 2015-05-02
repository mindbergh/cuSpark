#ifndef CUSPARK_TEXTPIPELINE_PIPELINE_H_
#define CUSPARK_TEXTPIPELINE_PIPELINE_H_

#include <iostream>
#include <fstream>
#include <string>
#include "common/function.h"
#include "common/logging.h"
#include "common/util.h"
#include "cuda/cuda-basics.h"
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
    class TextPipeLine {
      typedef BaseType (*InputMapOp)(const std::string&);

      public:
      TextPipeLine(std::string filename, 
          uint32_t size, InputMapOp f, Context *context);

      TextPipeLine(uint32_t size, Context *context);

      template <typename AfterType, typename UnaryOp>
        MappedPipeLine<AfterType, BaseType, UnaryOp> Map(UnaryOp f);

      template <typename BinaryOp>
        BaseType Reduce(BinaryOp f, BaseType identity);

      uint32_t GetDataSize();

      BaseType *GetData();

      BaseType GetElement(uint32_t index);

      BaseType GetElement_(uint32_t index);

      Context *GetContext();

      void Cache();

      void MallocCudaData();

      void FreeCudaData();

      void Execute();

      void Materialize(MemLevel ml);

      BaseType* data_; //pointer to the array, raw ptr in CUDA
      bool cached = false;
      uint32_t size_; //the length of the data array
      InputMapOp f_;
      std::string filename_;
      Context *context_;
      MemLevel memLevel;
    };

  template <typename BaseType>
    TextPipeLine<BaseType>::TextPipeLine(std::string filename, 
        uint32_t size, 
        InputMapOp f,
        Context *context)
    : size_(size),
    f_(f),
    filename_(filename),
    context_(context),
    memLevel(None) {}

  template <typename BaseType>
    TextPipeLine<BaseType>::TextPipeLine(uint32_t size, Context *context)
    : size_(size),
    context_(context),
    memLevel(None) {
      DLOG(INFO) << "Construct TextPipeLine by size and context: " << size;
    }

  template <typename BaseType>
    template <typename AfterType, typename UnaryOp>
    MappedPipeLine<AfterType, BaseType, UnaryOp> TextPipeLine<BaseType>::Map(UnaryOp f) {
      MappedPipeLine<AfterType, BaseType, UnaryOp> a(this, f);
      return a;
    }

  template <typename BaseType>
    template <typename BinaryOp>
    BaseType TextPipeLine<BaseType>::Reduce(BinaryOp f, BaseType identity) {
      DLOG(INFO) << "Executing Reduce\n";
      BaseType result = identity;
      size_t processed = 0;
      // to do: need current available mem
      size_t maxBatch = context_->getTotalGlobalMem() / sizeof(BaseType);
      thrust::device_ptr<BaseType> dptr;
      BaseType *data_ = this->GetData();

      switch (this->memLevel) {
        case Host:
          BaseType *cuda_data;
          if (size_ > maxBatch)
            cudaMalloc((void**)&cuda_data, maxBatch * sizeof(BaseType));
          else
            cudaMalloc((void**)&cuda_data, size_ * sizeof(BaseType));
          while (processed < size_) {
            size_t thisBatch = std::min(maxBatch, size_ - processed);
            DLOG(INFO) << "This batch: " << thisBatch;
            cudaMemcpy(cuda_data, data_ + processed, thisBatch * sizeof(BaseType), cudaMemcpyHostToDevice);
            dptr = thrust::device_pointer_cast(cuda_data);
            BaseType thisres = thrust::reduce(dptr, dptr+thisBatch, identity, f);
            result = f(result, thisres);
            processed += thisBatch;
          }
          return result;
        case Cuda:
          dptr = thrust::device_pointer_cast(data_);
          return thrust::reduce(dptr, dptr + size_, identity, f);
        case None:
          this->Materialize(Host);
          return this->Reduce(f, identity);
        default:
          return identity;
      }
    }

  template <typename BaseType>
    void TextPipeLine<BaseType>::Materialize(MemLevel ml) {
      std::ifstream infile;
      infile.open(filename_);
      std::string line;
      int count = 0;
      BaseType *mem_data;
      switch (ml) {
        case Host:
          this->memLevel = Host;
          data_ = new BaseType[size_];
          while (std::getline(infile, line)) {
            data_[count++] = f_(line);
          }
          return;
        case Cuda: 
          this->memLevel = Cuda;
          mem_data = new BaseType[size_];
          infile.open(filename_);
          while (std::getline(infile, line)) {
            mem_data[count++] = f_(line);
          }
          cudaMalloc((void**)&data_, count * sizeof(BaseType));
          cudaMemcpy(data_, mem_data, count * sizeof(BaseType), cudaMemcpyHostToDevice);
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
    uint32_t TextPipeLine<BaseType>::GetDataSize(){
      DLOG(INFO) << "Get Datasize";
      return this->size_;
    }
  template <typename BaseType>
    BaseType * TextPipeLine<BaseType>::GetData(){
      DLOG(INFO) << "Getting data from address: " << data_;
      return this->data_;
    }

  template <typename BaseType>
    Context * TextPipeLine<BaseType>::GetContext() {
      DLOG(INFO) << "Get Context";
      return this->context_;
    }

  template <typename BaseType>
    BaseType TextPipeLine<BaseType>::GetElement(uint32_t index){
      Execute();
      return GetElement_(index);
    }

  template <typename BaseType>
    BaseType TextPipeLine<BaseType>::GetElement_(uint32_t index){
      BaseType element;
      cudaMemcpy(&element, this->data_ + index, sizeof(BaseType), cudaMemcpyDeviceToHost);
      return element;
    }

  template <typename BaseType>
    void TextPipeLine<BaseType>::Cache(){
      cached = true;
    }

  template <typename BaseType>
    void TextPipeLine<BaseType>::MallocCudaData(){
      DLOG(INFO) << "malloc GPU memory for data with size :" << sizeof(BaseType) << " * " << size_;
      cudaMalloc((void**)&data_, size_ * sizeof(BaseType));
    }

  template <typename BaseType>
    void TextPipeLine<BaseType>::FreeCudaData(){
      if(!cached){
        DLOG(INFO) << "freeing GPU memory for data with size :" << sizeof(BaseType) << " * " << size_;
        cudaFree(data_);
        data_ = NULL;
      }
    }

  template <typename BaseType>
    void TextPipeLine<BaseType>::Execute(){
      DLOG(INFO) << "Executing PipeLine";
    }

}






#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
