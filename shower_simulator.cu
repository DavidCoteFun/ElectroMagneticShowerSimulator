#include <stdio.h>
#include <stdlib.h>

#include "Master.cuh"

//to activate DEBUG, compile like this:
// nvcc -DDEBUG debug_errors.cu 

inline void check_cuda_errors(const char *filename, const int line_number)
{
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}

__global__ void kernel(int *input, int *output)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  input[index] = 7;
}



int main(void)
{
  Master* mgr = new Master();
  //float init_energy=mgr->m_threshold*1.1;
  float init_energy=1000000;
  printf("Input electron energy: %f \n",init_energy);
  mgr->add_electron(init_energy,0,0,1,0);
  mgr->start();
  mgr->print();

  check_cuda_errors(__FILE__, __LINE__);
  return 0;
}
