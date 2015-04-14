#ifndef CUSPARK_PIPELINE_PIPELINE_H_
#define CUSPARK_PIPELINE_PIPELINE_H_

#include <common/types.h>

namespace cuspark {

template <typename T, typename U>
class MappedPipeLine;

template <typename T>
class PipeLine {
  public:
    PipeLine(T *data, uint32_t size);
   
    template <typename U>
    MappedPipeLine<T, U> Map(MapFunction<T, U, U(*)(T)> f){
      MappedPipeLine<T, U> a(this, f);
      return a;
    }
    
    T Reduce(ReduceFunction<T> f);
  
    uint32_t GetDataSize(){
	return size_;
    }
    
    T *GetData();

    T GetElement(uint32_t index);

    void Execute();

    uint32_t size_; //the length of the data array
    T* data_; //pointer to the array

};

}

#ifdef INNERFLAG
#include "pipeline/pipeline.cu"
#endif

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
