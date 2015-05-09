#ifndef CUSPARK_COMMON_TYPES_H
#define CUSPARK_COMMON_TYPES_H

namespace cuspark {

  template<typename T, size_t n>
    struct Tuple{

      T data[n];

      __host__ __device__ inline
        void set(int i, T value){
          data[i] = value;
        }

      __host__ __device__ inline
        T get(int i){
          return data[i];
        }

      __host__ __device__ 
        Tuple<T, n> operator+(Tuple<T, n>& obj){
          Tuple<T,n> result;
          for (int i = 0; i < n; i++) {
            result.set(i, data[i] + obj.get(i));
          }
          return result;
        }
      
      __host__  
        Tuple<T, n> operator/(int& cnt){
          Tuple<T,n> result;
          for (int i = 0; i < n; i++) {
            result.set(i, data[i] / cnt);
          }
          return result;
        }


        void divide(const int cnt) {
          std::cout << this->toString();
          for (int i = 0; i < n; i++) {
            //DLOG(INFO) << data[i] << ", " << data[i] / cnt;
            data[i] = data[i] / cnt;
          }
          std::cout << this->toString();
        }

      std::string toString() {
        std::stringstream ss;
        for (int i = 0; i < n; i++) {
          ss << data[i];
          ss << "\t";
        }
        return ss.str();
      }

      /*
         __host__ __device__
         Tuple<T, n>& operator<<(Tuple<T, n>& other) {
         if (&other == this)
         return *this;
         for (int i = 0; i < n; i++) {
         data[i] = other.get(i);
         }
         return *this;
         }

*/

      __host__ __device__ inline 
        T distTo(Tuple<T, n>& obj) {
          T dist = 0.0;
          for (int i = 0; i < n; i++) {
            T thisdist = data[i] - obj.get(i);
            dist += thisdist * thisdist;
          }
          return sqrt(dist / n);
        }

    };

}
#endif

