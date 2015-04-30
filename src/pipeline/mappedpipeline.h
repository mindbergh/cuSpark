#ifndef CUSPARK_PIPELINE_MAPPEDPIPELINE_H_
#define CUSPARK_PIPELINE_MAPPEDPIPELINE_H_

#include <stdio.h>
#include "common/function.h"
#include "common/logging.h"
#include "cuda/cuda-basics.h"
#include "pipeline/pipeline.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>

namespace cuspark {

  /*
   * Mapped from type BaseType to type AfterType
   */
  template <typename AfterType, typename BaseType, typename UnaryOp>
    class MappedPipeLine : public virtual PipeLine<AfterType> {
      public:
        MappedPipeLine(PipeLine<BaseType> *parent, UnaryOp f);

        //template <typename W>
        //MappedPipeLine<W, AfterType> Map(unaryOp f);

        template <typename BinaryOp>
          AfterType Reduce(BinaryOp f);

        AfterType *GetData();

        AfterType GetElement(uint32_t index);

        void Execute();
        
        UnaryOp f_;
        PipeLine<BaseType> *parent_;


    };

  template <typename AfterType, typename BaseType, typename UnaryOp>
    MappedPipeLine<AfterType, BaseType, UnaryOp>::
    MappedPipeLine(PipeLine<BaseType> *parent, UnaryOp f)
    : PipeLine<AfterType>(parent->GetDataSize()),
    parent_(parent),
    f_(f) {}


  template <typename AfterType, typename BaseType, typename UnaryOp>
    template <typename BinaryOp>
    AfterType
    MappedPipeLine<AfterType, BaseType, UnaryOp>::Reduce(BinaryOp f) {
      Execute();
      return PipeLine<AfterType>::Reduce(f);
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    AfterType * 
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetData() {
      Execute();
      return PipeLine<AfterType>::GetData_(); 
    }

  template <typename AfterType, typename BaseType, typename UnaryOp>
    AfterType 
    MappedPipeLine<AfterType, BaseType, UnaryOp>::GetElement(uint32_t index) {
      Execute();
      return PipeLine<AfterType>::GetElement(index);
    }


  template <typename AfterType, typename BaseType, typename UnaryOp>
    void 
    MappedPipeLine<AfterType, BaseType, UnaryOp>::Execute() {
      parent_ -> Execute();
      DLOG(INFO) << "Executing MappedPipeLine";
      PipeLine<AfterType>::MallocCudaData();
      thrust::device_ptr<BaseType> parent_data(parent_ -> data_);
      thrust::device_ptr<AfterType> child_data(this -> data_);
      thrust::transform(parent_data, 
          parent_data + this->size_, 
          child_data, f_);
      parent_ -> FreeCudaData();
    }


}

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
