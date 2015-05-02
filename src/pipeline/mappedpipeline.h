#ifndef CUSPARK_PIPELINE_MAPPEDPIPELINE_H_
#define CUSPARK_PIPELINE_MAPPEDPIPELINE_H_

#include <stdio.h>
#include "common/function.h"
#include "common/logging.h"
#include "common/util.h"
#include "cuda/cuda-basics.h"
#include "pipeline/pipeline.h"
#include "pipeline/textpipeline.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include "common/CycleTimer.h"

namespace cuspark {

  /*
   * Mapped from type BaseType to type AfterType
   */
  template <typename AfterType, typename BaseType, typename UnaryOp>
    class MappedPipeLine : public TextPipeLine<AfterType> {

      public:
        MappedPipeLine(TextPipeLine<BaseType> *parent, UnaryOp f);

        //template <typename W>
        //MappedPipeLine<W, AfterType> Map(unaryOp f);

        template <typename BinaryOp>
          AfterType Reduce(BinaryOp f, AfterType identity);

        AfterType *GetData();

        //AfterType GetElement(uint32_t index);

        Context *GetContext();

        uint32_t GetDataSize();

        void Materialize(MemLevel ml);

        UnaryOp f_;
        TextPipeLine<BaseType> *parent_;
      
      private:
        typedef TextPipeLine<AfterType> inherited;
    };

  template <typename AfterType, typename BaseType, typename UnaryOp>
    MappedPipeLine<AfterType, BaseType, UnaryOp>::
    MappedPipeLine(TextPipeLine<BaseType> *parent, UnaryOp f)
    : TextPipeLine<AfterType>(parent->GetDataSize(), parent->GetContext()),
    parent_(parent),
    f_(f) {}


  template <typename AfterType, typename BaseType, typename UnaryOp>
    template <typename BinaryOp>
    AfterType
    MappedPipeLine<AfterType, BaseType, UnaryOp>::Reduce(BinaryOp f, AfterType identity) {
      DLOG(INFO) << "Mapped Reduce from " << mlString(this->memLevel);
      DLOG(INFO) << "data_ addr: " << inherited::data_;
      switch (this->memLevel) {
        case Host:
        case Cuda:
          return TextPipeLine<AfterType>::Reduce(f, identity);
        case None:
          this->Materialize(Host);
          return TextPipeLine<AfterType>::Reduce(f, identity);
        default:
          return identity;
      }
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    AfterType * 
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetData() {
      return inherited::GetData();
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    Context *
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetContext() {
      return TextPipeLine<AfterType>::GetContext();
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    uint32_t
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetDataSize() {
      return TextPipeLine<AfterType>::GetDataSize();
    }


  /*
  template <typename AfterType, typename BaseType, typename UnaryOp>
    AfterType 
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetElement(uint32_t index) {
      return PipeLine<AfterType>::GetElement(index);
    }

    */
  template <typename AfterType, typename BaseType, typename UnaryOp>
    void
    MappedPipeLine<AfterType, BaseType, UnaryOp>::Materialize(MemLevel ml) {
      DLOG(INFO) << "Materialize from " << mlString(this->memLevel) << " to " << mlString(ml);
      BaseType *cuda_base;
      AfterType *cuda_after;
      thrust::device_ptr<BaseType> base_ptr;
      thrust::device_ptr<AfterType> after_ptr;
      size_t processed = 0;
      size_t maxBatch = 0;
      size_t size_ = this->GetDataSize();
      AfterType *data_;
      switch (ml) {
        case Host:
          inherited::data_ = new AfterType[this->GetDataSize()];
          switch (parent_->memLevel) {
            case Host:
              // to do: actuall need current available memory
              maxBatch = this->GetContext()->getTotalGlobalMem() / sizeof(BaseType);
              cudaMalloc((void**)&cuda_base, size_ * sizeof(BaseType));
              cudaMalloc((void**)&cuda_after, size_ * sizeof(AfterType));
              base_ptr = thrust::device_pointer_cast(cuda_base);
              after_ptr = thrust::device_pointer_cast(cuda_after);
              while (processed < size_) {
                size_t thisBatch = std::min(maxBatch, size_ - processed);
                cudaMemcpy(cuda_base, parent_->GetData() + processed, thisBatch * sizeof(BaseType), cudaMemcpyHostToDevice);
                thrust::transform(base_ptr, base_ptr + thisBatch, after_ptr, f_);
                cudaMemcpy(this->GetData() + processed, cuda_after, thisBatch * sizeof(AfterType), cudaMemcpyDeviceToHost);
                processed += thisBatch;
              }
              cudaFree(cuda_base);
              cudaFree(cuda_after);
              break;
            case Cuda:
              cudaMalloc((void**)&cuda_after, size_ * sizeof(AfterType));
              base_ptr = thrust::device_pointer_cast(parent_->GetData());
              after_ptr= thrust::device_pointer_cast(cuda_after);
              thrust::transform(base_ptr, base_ptr + size_, after_ptr, f_);
              cudaMemcpy(this->GetData(), cuda_after, size_ * sizeof(AfterType), cudaMemcpyDeviceToHost);
              cudaFree(cuda_after);
              break;
            case None:
              parent_->Materialize(Host);
              this->Materialize(Host);
          }
          this->memLevel = Host;
          break;
        case Cuda:
          cudaMalloc((void**)&(data_), size_ * sizeof(AfterType));
          inherited::data_ = data_;
          switch (parent_->memLevel) {
            case Host:
              cudaMalloc((void**)&cuda_base, size_ * sizeof(BaseType));
              cudaMemcpy(cuda_base, parent_->GetData(), size_ * sizeof(BaseType), cudaMemcpyHostToDevice);
              base_ptr = thrust::device_pointer_cast(cuda_base);
              after_ptr = thrust::device_pointer_cast(this->GetData());
              thrust::transform(base_ptr, base_ptr + size_, after_ptr, f_);
              cudaFree(cuda_base);
              break;
            case Cuda:
              base_ptr = thrust::device_pointer_cast(parent_->GetData());
              after_ptr = thrust::device_pointer_cast(this->GetData());
              thrust::transform(base_ptr, base_ptr + size_, after_ptr, f_);
              break;
            case None:
              parent_->Materialize(Cuda);
              this->Materialize(Cuda);
              break;
          }
          this->memLevel = Cuda;
          break;
        case None:
          this->memLevel = None;
          break;
        default:
          break;
      }
      return;
    }
}

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
