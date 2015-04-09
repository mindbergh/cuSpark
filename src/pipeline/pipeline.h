#ifndef CUSPARK_SRC_PIPELINE_PIPELINE_H_
#define CUSPARK_SRC_PIPELINE_PIPELINE_H_


namespace cuspark {

template <typename T>
class PipeLine {
  public:
    PipeLine(T *data, uint32_t size)
      : data_(data),
        size_(size) {}
    
    
    template <typename U>
    MappedPipeLine<U> map(MapFunction<T, U> f);
    
    T reduce(ReduceFunction<T> f);
  
    uint32_t GetDataSize() const {
      return size_;
    }
    
    T *GetData() const {
      return data_;
    }

    T GetElement(uint32_t index) {
      return data_[index];
    }

  private:
  

    uint32_t size_;
    T *data_;




}

}

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
