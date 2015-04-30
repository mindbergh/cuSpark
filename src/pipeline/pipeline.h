#ifndef CUSPARK_PIPELINE_PIPELINE_H_
#define CUSPARK_PIPELINE_PIPELINE_H_

#include <iostream>
#include <fstream>
#include <string>
#include "common/function.h"
#include "common/logging.h"
#include "cuda/cuda-basics.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>

namespace cuspark {

  template <typename AfterType, typename BaseType, typename UnaryOp>
    class MappedPipeLine;

  /*
   * Basic PipeLine class, which we generate from file or array
   * 
   */
  template <typename BaseType>
    class PipeLine {
      public:
        PipeLine(std::string path, uint32_t size);
        PipeLine(BaseType *data, uint32_t size);
        PipeLine(std::string filename, 
            uint32_t size, StringMapFunction<BaseType> f);
        PipeLine(uint32_t size);

        template <typename AfterType, typename UnaryOp>
          MappedPipeLine<AfterType, BaseType, UnaryOp> Map(UnaryOp f);

        template <typename BinaryOp>
          BaseType Reduce(BinaryOp f);

        uint32_t GetDataSize();

        BaseType *GetData();

        BaseType *GetData_();

        BaseType GetElement(uint32_t index);

        BaseType GetElement_(uint32_t index);

        void Cache();

        void MallocCudaData();

        void FreeCudaData();

        void Execute();
 
        BaseType* data_; //pointer to the array, raw ptr in CUDA
        bool cached = false;
        uint32_t size_; //the length of the data array

   };
  

  // size can be larger than cuda global mem
  template <typename BaseType>
    PipeLine<BaseType>::PipeLine(std::string path, uint32_t size)
    : size_(size) {
      DLOG(INFO) << "Initialing pipeline from textfile lazily" << std::endl;
      
    }

  template <typename BaseType>
    PipeLine<BaseType>::PipeLine(BaseType *data, uint32_t size) 
    : size_(size) {
      DLOG(INFO) << "initiating pipeline from array";
      MallocCudaData();
      cudaMemcpy(data_, data, size_ * sizeof(BaseType), cudaMemcpyHostToDevice);
    }

  template <typename BaseType>
    PipeLine<BaseType>::PipeLine(std::string filename, 
        uint32_t size, 
        StringMapFunction<BaseType> f)
    : size_(size) {
      DLOG(INFO) << "initiating pipeline from file: " << size_ << std::endl;
      MallocCudaData();
      BaseType* cache = new BaseType[size_];

      std::ifstream infile;
      infile.open(filename);
      std::string line;
      int line_number = 0;
      while(std::getline(infile, line)){
        cache[line_number++] = f(line);
      }
      cudaMemcpy(data_, cache, 
          size_ * sizeof(BaseType), cudaMemcpyHostToDevice);
      free(cache);
    }

  template <typename BaseType>
    PipeLine<BaseType>::PipeLine(uint32_t size)
    : size_(size) {}

  template <typename BaseType>
  template <typename AfterType, typename UnaryOp>
    MappedPipeLine<AfterType, BaseType, UnaryOp> PipeLine<BaseType>::Map(UnaryOp f) {
      MappedPipeLine<AfterType, BaseType, UnaryOp> a(this, f);
      return a;
    }

  template <typename BaseType>
  template <typename BinaryOp>
    BaseType PipeLine<BaseType>::Reduce(BinaryOp f){
      DLOG(INFO) << "Executing Reduce";
      thrust::device_ptr<BaseType> self_data(data_);
      BaseType init = GetElement_(0);
      BaseType result = thrust::reduce(self_data + 1, 
          self_data + size_, init, f);
      FreeCudaData();
      return result;
    }

  template <typename BaseType>
    uint32_t PipeLine<BaseType>::GetDataSize(){
      return size_;
    }
  template <typename BaseType>
    BaseType * PipeLine<BaseType>::GetData(){
      Execute();
      return GetData_();
    }

  template <typename BaseType>
    BaseType * PipeLine<BaseType>::GetData_(){
      DLOG(INFO) << "Getting data from address: " << data_;
      BaseType* data = (BaseType*)malloc(size_ * sizeof(BaseType));
      cudaMemcpy(data, this->data_, size_ * sizeof(BaseType), cudaMemcpyDeviceToHost);
      return data;
    }

  template <typename BaseType>
    BaseType PipeLine<BaseType>::GetElement(uint32_t index){
      Execute();
      return GetElement_(index);
    }

  template <typename BaseType>
    BaseType PipeLine<BaseType>::GetElement_(uint32_t index){
      BaseType element;
      cudaMemcpy(&element, this->data_ + index, sizeof(BaseType), cudaMemcpyDeviceToHost);
      return element;
    }

  template <typename BaseType>
    void PipeLine<BaseType>::Cache(){
      cached = true;
    }

  template <typename BaseType>
    void PipeLine<BaseType>::MallocCudaData(){
      DLOG(INFO) << "malloc GPU memory for data with size :" << sizeof(BaseType) << " * " << size_;
      cudaMalloc((void**)&data_, size_ * sizeof(BaseType));
    }

  template <typename BaseType>
    void PipeLine<BaseType>::FreeCudaData(){
      if(!cached){
        DLOG(INFO) << "freeing GPU memory for data with size :" << sizeof(BaseType) << " * " << size_;
        cudaFree(data_);
        data_ = NULL;
      }
    }

  template <typename BaseType>
    void PipeLine<BaseType>::Execute(){
      DLOG(INFO) << "Executing PipeLine";
    }





}






#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
