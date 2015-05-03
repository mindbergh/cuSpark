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

        template <typename BinaryOp>
          AfterType Reduce(BinaryOp f);

        // return the data in memory
        AfterType *GetData();

        Context *GetContext();

        uint32_t GetDataSize();

        void Materialize(MemLevel ml);

        // functor of the map operation
        UnaryOp f_;

        PipeLine<BaseType> *parent_;
    };

  template <typename AfterType, typename BaseType, typename UnaryOp>
    MappedPipeLine<AfterType, BaseType, UnaryOp>::
    MappedPipeLine(PipeLine<BaseType> *parent, UnaryOp f)
    : PipeLine<AfterType>(parent->GetDataSize(), parent->GetContext()),
    parent_(parent),
    f_(f) {}


  template <typename AfterType, typename BaseType, typename UnaryOp>
    template <typename BinaryOp>
    AfterType
    MappedPipeLine<AfterType, BaseType, UnaryOp>::Reduce(BinaryOp f) {
      DLOG(INFO) << "Mapped Reduce from " << mlString(this->memLevel_);
      DLOG(INFO) << "data_ addr: " << PipeLine<AfterType>::data_;
      switch (this->memLevel_) {
        case Host:
        case Cuda:
          return PipeLine<AfterType>::Reduce(f);
        case None:
          this->Materialize(Host);
          return PipeLine<AfterType>::Reduce(f);
        default:
          DLOG(FATAL) << "memLevel type not correct";
          exit(1);
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
    uint32_t
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetDataSize() {
      return PipeLine<AfterType>::GetDataSize();
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    void
    MappedPipeLine<AfterType, BaseType, UnaryOp>::Materialize(MemLevel ml) {
      DLOG(INFO) << "Materialize from " << mlString(this->memLevel_) << " to " << mlString(ml);
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
          PipeLine<AfterType>::data_ = new AfterType[this->GetDataSize()];
          switch (parent_->memLevel_) {
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
          this->memLevel_ = Host;
          break;
        case Cuda:
          cudaMalloc((void**)&(data_), size_ * sizeof(AfterType));
          PipeLine<AfterType>::data_ = data_;
          switch (parent_->memLevel_) {
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
          this->memLevel_ = Cuda;
          break;
        case None:
          this->memLevel_ = None;
          break;
        default:
          break;
      }
      return;
    }
}

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
