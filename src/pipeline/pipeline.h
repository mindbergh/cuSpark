#ifndef CUSPARK_PIPELINE_PIPELINE_H_
#define CUSPARK_PIPELINE_PIPELINE_H_

#include <common/types.h>
#include <boost/function.hpp>
#include <boost/function_equal.hpp>

namespace cuspark {

template <typename T, typename U>
class MappedPipeLine;

template <typename T>
class PipeLine {
  public:
    PipeLine(T *data, uint32_t size)
      : data_(data),
        size_(size) {}
   
    template <typename U>
    MappedPipeLine<T, U> Map(MapFunction<T, U> f){
      T* data = (T*) malloc (sizeof(T) * size_);
      MappedPipeLine<T, U> a(this, f, data);
      return a;
    }
    
    T Reduce(ReduceFunction<T> f);
  
    uint32_t GetDataSize() const{
	return size_;
    }
    
    T *GetData() const {
      return data_;
    }

    T GetElement(uint32_t index) {
      return data_[index];
    }

  private:

    uint32_t size_; //the length of the data array
    T *data_; //pointer to the array

};

}

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
