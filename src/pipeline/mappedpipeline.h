#ifndef CUSPARK_PIPELINE_MAPPEDPIPELINE_H_
#define CUSPARK_PIPELINE_MAPPEDPIPELINE_H_

#include "common/types.h"
#include "pipeline/pipeline.h"

namespace cuspark {

/**
 * Mapped from type U to type T
 *
 */
template <typename T, typename U>
class MappedPipeLine : public PipeLine<T> {
  public:
    MappedPipeLine(PipeLine<U> *parent, MapFunction<U, T, U(*)(T)> f)
        : PipeLine<T>(NULL, parent->GetDataSize()),
	  parent_(parent),
	  f_(f) {}

    template <typename W>
    MappedPipeLine<T, W> Map(MapFunction<T, W, U(*)(T)> f);

    void Execute();

    T Reduce(ReduceFunction<T> f);
    
    T *GetData() {
      Execute();
      return PipeLine<T>::GetData(); 
    }

    T GetElement(uint32_t index) {
      Execute();
      return PipeLine<T>::GetElement(index);
    }

  protected:

    MapFunction<U, T, U(*)(T)> f_;
    PipeLine<U> *parent_;

};

}
#ifdef INNERFLAG
#include "pipeline/mappedpipeline.cu"
#endif

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
