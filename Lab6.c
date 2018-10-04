// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

__global__ void floatToUchar(float *inputImage, unsigned char *ucharImage, int width, int height)
{
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int current;
  if (w < width && h < height)
  {
    current = blockIdx.z * width * height + h * width + w;
    ucharImage[current] = (unsigned char) (255 * inputImage[current]);
  }
}

__global__ void RGBtoGrayScale(unsigned char *inputImage, unsigned char *grayImage, int width, int height)
{
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int current;
  if (w < width && h < height)
  {
    current = h * width + w;
    unsigned char r = inputImage[3 * current];
    unsigned char g = inputImage[3 * current + 1];
    unsigned char b = inputImage[3 * current + 2];
    grayImage[current] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void histo_kernel(unsigned char *input, unsigned int *output, int width, int height)
{
  __shared__ unsigned int histogram[256];

  int tx = threadIdx.y * blockDim.x + threadIdx.x;
  if (tx < 256)
    histogram[tx] = 0;
  __syncthreads();
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (w < width && h < height)
  {
    int current = h * width + w;
    unsigned char value = input[current];
    atomicAdd(&(histogram[value]), 1);
  }
  __syncthreads();
  if (tx < 256)
    atomicAdd(&(output[tx]), histogram[tx]);
}

__global__ void Histo_to_CDF(unsigned int *histo, float *CDF, int width, int height)
{
  __shared__ unsigned int cdf_shared[256];
  int tx = threadIdx.x;
  cdf_shared[tx] = histo[tx];
  int stride = 1;
  while (stride < 256)
  {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if ((index < 256) && (index - stride >= 0))
      cdf_shared[index] += cdf_shared[index - stride];
    stride *= 2;
  }
  __syncthreads();
  stride = 256 / 4;
  while (stride > 0)
  {
    int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < 256)
      cdf_shared[index + stride] += cdf_shared[index];
    stride = stride / 2;
    __syncthreads();
  }
  CDF[tx] = cdf_shared[tx] / float((width * height));
}

__global__ void CDF_equalization(float *CDF, unsigned char *output, int width, int height)
{
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int current;
  if (w < width && h < height)
  {
    current = blockIdx.z * width * height + h * width + w;
    unsigned char value = output[current];
    float correct_value = max(255*(CDF[value] - CDF[0])/(1.0 - CDF[0]), 0.0);
    float retval = min(correct_value, 255.0);
    output[current] = (unsigned char) retval;
  }
}

__global__ void Uchartofloat(unsigned char *inputImage, float *floatImage, int width, int height)
{
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int current;
  if (w < width && h < height)
  {
    current = blockIdx.z * width * height + h * width + w;
    floatImage[current] = (float) (inputImage[current] / 255.0);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  float *deviceFloat;
  float *deviceCDF;
  unsigned char *deviceUChar;
  unsigned char *deviceGrayScale;
  unsigned int *deviceHisto;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");


  cudaMalloc((void **)&deviceFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceUChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGrayScale, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHisto, 256 * sizeof(unsigned int));
  cudaMalloc((void**) &deviceCDF, 256 * sizeof(float));

  cudaMemset((void *) deviceHisto, 0, 256 * sizeof(unsigned int));

  cudaMemcpy(deviceFloat, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid  = dim3(ceil(imageWidth / 32.0), ceil(imageHeight / 32.0), imageChannels);
  dim3 dimBlock = dim3(32.0, 32.0, 1.0);

  floatToUchar<<<dimGrid, dimBlock>>>(deviceFloat, deviceUChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth / 32.0), ceil(imageHeight / 32.0), 1.0);
  dimBlock = dim3(32.0, 32.0, 1.0);

  RGBtoGrayScale<<<dimGrid, dimBlock>>>(deviceUChar, deviceGrayScale, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth / 32.0), ceil(imageHeight / 32.0), 1.0);
  dimBlock = dim3(32.0, 32.0, 1.0);

  histo_kernel<<<dimGrid, dimBlock>>>(deviceGrayScale, deviceHisto, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(1.0, 1.0, 1.0);
  dimBlock = dim3(256.0, 1.0, 1.0);

  Histo_to_CDF<<<dimGrid, dimBlock>>>(deviceHisto, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth / 32.0), ceil(imageHeight / 32.0), imageChannels);
  dimBlock = dim3(32.0, 32.0, 1.0);

  CDF_equalization<<<dimGrid, dimBlock>>>(deviceCDF, deviceUChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth / 32.0), ceil(imageHeight / 32.0), imageChannels);
  dimBlock = dim3(32.0, 32.0, 1.0);

  Uchartofloat<<<dimGrid, dimBlock>>>(deviceUChar, deviceFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceFloat, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  cudaFree(deviceFloat);
  cudaFree(deviceUChar);
  cudaFree(deviceGrayScale);
  cudaFree(deviceHisto);
  cudaFree(deviceCDF);

  return 0;
}
