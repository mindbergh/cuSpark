#ifndef CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
#define CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_

namespace cuspark {

/**
 * Mapped from type U to type T
 *
 */
template <typename T, typename U>
class MappedPipeLine : public PipeLine<T> {
  public:
    MappedPipeLine(PipeLine<U> *parent, MapFunction<U, T> f)
      : parent_(parent),
        f_(f) {}

    template <typename U>
    MappedPipeLine<U> Map(MapFuncion<T, U> f);

    T Reduce(ReduceFunction<T> f);

    uint32_t GetDataSize() const {
      return parent_->GetDataSize();
    }

    T *GetData() const {
      return 
    }

    T GetElement(uint32_t index) {
      return f_(parent_->GetElement(index));
    }

  private:

    MapFunction<U, T> f_;
    PipeLine<U> parent_;

}


}



#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_

