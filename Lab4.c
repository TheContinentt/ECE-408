#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define MASK_WIDTH 3
#define MASK_RADIUS 1
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float  N_ds[TILE_WIDTH + MASK_WIDTH - MASK_RADIUS][TILE_WIDTH + MASK_WIDTH - MASK_RADIUS][TILE_WIDTH + MASK_WIDTH - MASK_RADIUS];

  int temp;
  float result = 0.0;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int hei_o = blockIdx.z * TILE_WIDTH;
  int row_o = blockIdx.x * TILE_WIDTH;
  int col_o = blockIdx.y * TILE_WIDTH;

  int row_i = row_o - MASK_RADIUS;
  int col_i = col_o - MASK_RADIUS;
  int hei_i = hei_o - MASK_RADIUS;

  int index = tz * MASK_WIDTH * MASK_WIDTH + ty * MASK_WIDTH + tx;
  int indexX = index % (TILE_WIDTH + MASK_WIDTH - MASK_RADIUS);
  temp = index / (TILE_WIDTH + MASK_WIDTH - MASK_RADIUS);
  int indexY = temp % (TILE_WIDTH + MASK_WIDTH - MASK_RADIUS);

  if (index < (TILE_WIDTH + MASK_WIDTH - MASK_RADIUS) * (TILE_WIDTH + MASK_WIDTH - MASK_RADIUS))
  {
    row_i = row_i + indexX;
    col_i = col_i + indexY;
    int temphei_i = hei_i;

    for (int i = 0; i < TILE_WIDTH + MASK_WIDTH - MASK_RADIUS; i++)
    {
      hei_i = hei_i + i;
      if ((row_i >= 0) && (row_i < x_size) && (col_i >= 0) && (col_i < y_size) && (hei_i >= 0) && (hei_i < z_size))
      {
        N_ds[indexX][indexY][i] = input[hei_i * x_size * y_size + col_i * x_size + row_i];
      }
      else
      {
        N_ds[indexX][indexY][i] = 0.0;
      }
      hei_i = temphei_i;
    }
  }
  __syncthreads();

  row_i = row_o + tx;
  col_i = col_o + ty;
  hei_i = hei_o + tz;
  if ((row_i >= 0) && (row_i < x_size) && (col_i >= 0) && (col_i < y_size) && (hei_i >= 0) && (hei_i < z_size))
  {
   for (int i = 0; i < MASK_WIDTH; i++)
   {
     for (int j = 0; j < MASK_WIDTH; j++)
     {
       for (int k = 0; k < MASK_WIDTH; k++)
       {
         result = result + N_ds[tx + i][ty + j][tz + k] * deviceKernel[k][j][i];
       }
     }
   }
    output[(hei_o + tz) * x_size * y_size + (col_o + ty) * x_size + row_o + tx] = result;
  }

}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(ceil(x_size / (1.0 * TILE_WIDTH)),ceil(y_size / (1.0 * TILE_WIDTH)), ceil(z_size / (1.0 * TILE_WIDTH)));


  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
