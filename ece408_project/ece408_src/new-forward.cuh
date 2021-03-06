
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 1024
#define Mask_width  5
#define clamp(x) min(max((x), 0.0), 1.0)
#define Mask_radius 2
#define ww 20

namespace mxnet
{
namespace op
{


__constant__ float subtileW[3000];

__global__ void forward_kernel(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    //__shared__ float acc;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
    int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));

    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int h = (blockIdx.z / W_grid)*blockDim.y + threadIdx.y;
    const int w = (blockIdx.z % W_grid)*blockDim.x + threadIdx.x;

float acc=0.0;
if(n < B && m < M && h < H_out && w < W_out){
#pragma unroll
for(int c = 0;  c < C; c++)         // sum over all input feature maps
  for(int p = 0; p < K; p++)	       // KxK  filter
    for(int q = 0; q < K; q++)
        acc+= x4d(n, c, (h + p), (w + q)) * subtileW[m * (C * K * K) + c * (K * K) + p * (K) + q];
        //atomicAdd(&acc, x4d(n, c, (h + p), (w + q)) * subtileW[m * (C * K * K) + c * (K * K) + p * (K) + q]);
//printf("REach here.\n");
y4d(n, m, h, w) = acc;
//atomicAdd(&y4d(n, m, h, w), acc);
}

#undef y4d
#undef x4d
#undef k4d
}


/*__global__ void unroll_kernel(int C, int H, int W, int K, const float* X, float* X_unroll)
{
    //#define x4d(i3,i2,i1,i0) X[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]

    int c, s, p, q, h_out, w_out, h_unroll,w_unroll;
    int H_out, W_out, W_unroll;

    int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;

    H_out = H - K + 1;
    W_out = W - K + 1;
    W_unroll = H_out * W_out;

    int h_base;

    if (t < C * W_unroll)
    {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;

        w_unroll = h_out * W_out + w_out;
        h_base = c * K * K;

        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++)
            {
                h_unroll = h_base + p * K + q;
                X_unroll[h_unroll * W_unroll + w_unroll] = X[c * H * W +(h_out + p) * W +(w_out + q)];
            }
        }
    }

}*/


/*__global__ void matrixMultiplykernel(float* W_dot, float* X_unrolled, float* X, float* Y, int M, int H_unroll, int W_unroll, int C, int H, int W, int K)
{
  __shared__ float subTileW[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileX[TILE_WIDTH][TILE_WIDTH];*/
  /*int c, s, p, q, h_out, w_out, h_unroll,w_unroll;
  int H_out, W_out;
  int t = blockIdx.z * CUDA_MAX_NUM_THREADS * blockIdx.y * blockIdx.x + threadIdx.z;

  H_out = H - K + 1;
  W_out = W - K + 1;
  //W_unroll = H_out * W_out;

  int h_base;

  if (t < C * W_unroll)
  {
      c = t / W_unroll;
      s = t % W_unroll;
      h_out = s / W_out;
      w_out = s % W_out;

      w_unroll = h_out * W_out + w_out;
      h_base = c * K * K;

      for(p = 0; p < K; p++){
          for(q = 0; q < K; q++)
          {
              h_unroll = h_base + p * K + q;
              X_unrolled[h_unroll * W_unroll + w_unroll] = X[c * H * W +(h_out + p) * W +(w_out + q)];
          }
      }
  }
  __syncthreads();*/
/*  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by*blockDim.y+ty;
  int Col = bx*blockDim.x+tx;
  float Pvalue = 0;
  for (int m=0; m<(H_unroll- 1)/TILE_WIDTH + 1; m++){
    if(m*TILE_WIDTH + tx < H_unroll && Row < M){
        subTileW[ty][tx] = W_dot[Row*H_unroll+ m*TILE_WIDTH+tx];
      }
      else{
        subTileW[ty][tx] = 0.0f;
      }
      if(m*TILE_WIDTH + ty < H_unroll && Col < W_unroll){
        subTileX[ty][tx] = X[(m*TILE_WIDTH+ty)*W_unroll+ Col];
      }
      else{
        subTileX[ty][tx] = 0.0f;
      }
      __syncthreads();
      for(int k = 0; k < TILE_WIDTH; ++k){
        Pvalue += subTileW[ty][k] * subTileX[k][tx];
      }
      __syncthreads();
  }
  if(Row < M && Col < W_unroll){
    Y[Row * W_unroll + Col] = Pvalue;
  }
}*/


/*__global__ void MatMulOPT(float* A, float* B, float* C, int A_col, int B_col, int C_col)
{
    __shared__ float result[CUDA_MAX_NUM_THREADS];
    int e = threadIdx.x;
    int row = blockIdx.y;
    int col = blockIdx.x;

    result[e] = A[row*A_col + 2] * B[2*B_col + col];
    __syncthreads();
	for(int size=blockDim.x/2; size>0; size/=2)
  	{
		if (e < size)
		{
			result[e] += result[e + size];
		}
	}
	if (e == 0)C[row*C_col + 2] = result[0];
}*/

__global__ void convolution(float *I, float *M, float *P, int channels, int width, int height, int bbb) 
{
   __shared__ float N_ds[ww][ww];
   int k;
   for (k = 0; k < channels; k++) {
      // First batch loading
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,// + bbb * TILE_WIDTH * TILE_WIDTH,
         destY = dest / ww, destX = dest % ww,
         srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius,
         srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
         src = (srcY * width + srcX) * channels + k;
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         N_ds[destY][destX] = I[src];
      else
         N_ds[destY][destX] = 0;

      __syncthreads();

      float accum = 0;
      int y, x;
      for (y = 0; y < Mask_width; y++)
         for (x = 0; x < Mask_width; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
      y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         P[(y * width + x) * channels + k] = clamp(accum);
      __syncthreads();
   }
}


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
    const int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));

    printf("W = %d\n", W);
    printf("C = %d\n", C);
    printf("B = %d\n", B);
    printf("M = %d\n", M);
    printf("H = %d\n", H);
    printf("K = %d\n", K);


    if(B<200)
    {
    cudaMemcpyToSymbol(subtileW, w.dptr_, M *C*K*K * sizeof(float));

    const int Z = H_grid * W_grid;
    // Set the kernel dimensions
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);



    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    }
    else{


    //printf("wsize = %d\n", sizeof(w));
    ///////////////////////////////////////////////////////////////////////////////

    float* X_unrolled;
    //int num_thread = C * H_out * W_out;
    //int num_blocks_unroll = (num_thread - 1) / ((CUDA_MAX_NUM_THREADS + 1) * (H_out*W_out/(TILE_WIDTH*1.0)) * ((M - 1)/TILE_WIDTH + 1));
    //int num_blocks_unroll = (num_thread - 1) / TILE_WIDTH + 1;
    const int H_unroll = C*K*K;
    const int W_unroll = W_out * H_out;

    cudaMalloc((void**) &X_unrolled, W_unroll * H_unroll * sizeof(float));

    //dim3 grid_Unroll(num_blocks_unroll, 1, 1);
    //dim3 block_Unroll(CUDA_MAX_NUM_THREADS, 1, 1);

    dim3 grid_Multiple(ceil(H_out*W_out/(TILE_WIDTH*1.0)), (M - 1)/TILE_WIDTH + 1, 1);
    dim3 block_Multiple(TILE_WIDTH, TILE_WIDTH, 1);

	dim3 optDimGrid(W_out, H_out, 1);
	dim3 optDimBlock(1024, 1, 1);

    dim3 dimGridd(ceil((float)W_out/TILE_WIDTH), ceil((float)H_out/TILE_WIDTH), 1.0);
    dim3 dimBlockk(TILE_WIDTH, TILE_WIDTH, 1);

    for (int n=0; n < B; n++)
    {
      //unroll_kernel<<<grid_Unroll, block_Unroll>>>(C, H, W, K, x.dptr_+C*H*W*n, X_unrolled);
      //matrixMultiplykernel<<<grid_Multiple, block_Multiple>>>(w.dptr_, X_unrolled, x.dptr_+C*H*W*n, y.dptr_+M*H_out*W_out*n,M,H_unroll,W_unroll, C, H, W, K);
	//MatMulOPT <<<optDimGrid, optDimBlock >>> (x.dptr_, w.dptr_, y.dptr_, W, K, W_out);
	convolution <<<dimGridd, dimBlockk>>>(x.dptr_, w.dptr_, y.dptr_, C, W_out, H_out, n);
    }

     //Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    cudaFree(X_unrolled);


    cudaDeviceSynchronize();
  }


}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
