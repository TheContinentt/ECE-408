// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float ShareStore[BLOCK_SIZE * 2];
  int tx = threadIdx.x;
  int distance = 2 * blockIdx.x * blockDim.x;
  int first = tx + distance;
  int second = first + blockDim.x;
  if (first < len)
    ShareStore[tx] = input[first];
  else
    ShareStore[tx] = 0;
  if (second < len)
    ShareStore[second - distance] = input[second];
  else
    ShareStore[second - distance] = 0;
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE)
  {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if ((index < 2 * BLOCK_SIZE) && (index - stride >= 0))
      ShareStore[index] += ShareStore[index - stride];
    stride *= 2;
  }
  __syncthreads();
  stride = BLOCK_SIZE / 2;
  while (stride > 0)
  {
    int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE)
      ShareStore[index + stride] += ShareStore[index];
    stride = stride / 2;
    __syncthreads();
  }
  if (first < len)
    output[first] = ShareStore[tx];
  if (second < len)
    output[second] = ShareStore[second - distance];
}

__global__ void Saved(float * input, float * output, int len) {
  int tx = threadIdx.x;
  int index = (tx + 1) * BLOCK_SIZE * 2 - 1;
  __syncthreads();
  if (index < len)
    output[tx] = input[index];
}

__global__ void Added(float * input, float * output, int len) {
  int tx = threadIdx.x;
  int index = tx + blockIdx.x * blockDim.x;
  int final_idx = index + blockDim.x;
  __syncthreads();
  if (final_idx < len)
    output[final_idx] += input[blockIdx.x];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *temp;
  float *ttemp;
  float* store;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&temp, (ceil(numElements / float(BLOCK_SIZE * 2))) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&store, (ceil(numElements / float(BLOCK_SIZE * 2))) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int Block_num = ceil(numElements / float(BLOCK_SIZE * 2));

  dim3 DimGrid(Block_num, 1.0, 1.0);
  dim3 DimBlock(BLOCK_SIZE, 1.0, 1.0);

  dim3 GridSave(1.0, 1.0, 1.0);
  dim3 BlockSave(Block_num, 1.0, 1.0);

  dim3 GridScan(1.0, 1.0, 1.0);
  dim3 BlockScan(Block_num, 1.0, 1.0);

  dim3 GridAdd(Block_num, 1.0, 1.0);
  dim3 BlockAdd(2 * BLOCK_SIZE, 1.0, 1.0);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);

  Saved<<<GridSave, BlockSave>>>(deviceOutput, temp, numElements);

  scan<<<GridScan, BlockScan>>>(temp, store, Block_num);

  Added<<<GridAdd, BlockAdd>>>(store, deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(temp);
  cudaFree(store);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
