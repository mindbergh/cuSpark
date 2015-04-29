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
        PipeLine(BaseType *data, uint32_t size):size_(size){
          DLOG(INFO) << "initiating pipeline from array";
          MallocCudaData();
          cudaMemcpy(data_, data, size_ * sizeof(BaseType), cudaMemcpyHostToDevice);
        }

        PipeLine(std::string filename, uint32_t size, StringMapFunction<BaseType> f):size_(size){
          DLOG(INFO) << "initiating pipeline from file: "<<size_;
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

        PipeLine(uint32_t size):size_(size){}

        template <typename AfterType, typename UnaryOp>
          MappedPipeLine<AfterType, BaseType, UnaryOp> Map(UnaryOp f) {
            MappedPipeLine<AfterType, BaseType, UnaryOp> a(this, f);
            return a;
          }

        template <typename BinaryOp>
          BaseType Reduce(BinaryOp f){
            DLOG(INFO) << "Executing Reduce";
            thrust::device_ptr<BaseType> self_data(data_);
            BaseType init = GetElement_(0);
            BaseType result = thrust::reduce(self_data + 1, 
                                             self_data + size_, init, f);
            FreeCudaData();
            return result;
          }

        uint32_t GetDataSize(){
          return size_;
        }

        BaseType *GetData(){
          Execute();
          return GetData_();
        }

        BaseType *GetData_(){
          DLOG(INFO) << "Getting data from address: " << data_;
          BaseType* data = (BaseType*)malloc(size_ * sizeof(BaseType));
          cudaMemcpy(data, this->data_, size_ * sizeof(BaseType), cudaMemcpyDeviceToHost);
          return data;
        }

        BaseType GetElement(uint32_t index){
          Execute();
          return GetElement_(index);
        }

        BaseType GetElement_(uint32_t index){
          BaseType element;
          cudaMemcpy(&element, this->data_ + index, sizeof(BaseType), cudaMemcpyDeviceToHost);
          return element;
        }

        void Cache(){
          cached = true;
        }

        //protected:

        BaseType* data_; //pointer to the array, raw ptr in CUDA
        bool cached = false;
        uint32_t size_; //the length of the data array

        void MallocCudaData(){
          DLOG(INFO) << "malloc GPU memory for data with size :" << sizeof(BaseType) << " * " << size_;
          cudaMalloc((void**)&data_, size_ * sizeof(BaseType));
        }

        void FreeCudaData(){
          if(!cached){
            DLOG(INFO) << "freeing GPU memory for data with size :" << sizeof(BaseType) << " * " << size_;
            cudaFree(data_);
            data_ = NULL;
          }
        }

        void Execute(){
          DLOG(INFO) << "Executing PipeLine";
        }

    };

}

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
