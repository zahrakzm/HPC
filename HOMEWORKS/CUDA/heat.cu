#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

/*
 * `step_kernel_mod` is currently a direct copy of the CPU reference solution
 * `step_kernel_ref` below. Accelerate it to run as a CUDA kernel.
 */

__global__ void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  int indexWithinTheGrid = blockIdx.x* blockDim.x+ threadIdx.x;
  int gridStride = gridDim.x * blockDim.x;
  int N = (ni-2)*(nj-2);

  for(int k=indexWithinTheGrid; k<N; k+=gridStride){
    int i = k % (ni-2) + 1;
    int j = k / (nj-2) + 1;

    i00 = I2D(ni, i, j);
    im10 = I2D(ni, i-1, j);
    ip10 = I2D(ni, i+1, j);
    i0m1 = I2D(ni, i, j-1);
    i0p1 = I2D(ni, i, j+1);
    
    // evaluate derivatives
    d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
    d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];
  
    // update temperatures
    temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
  }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;


  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
  }
}

int main(int argc, char* argv[]){
  if(argc != 3){
    printf("Usage: %s <N> <num_threads>\n", argv[0]);
    return 1;
  }

  int istep;
  int nstep = 200; // number of time steps
  int num_threads = atoi(argv[2]);

  // Specify our 2D dimensions
  const int ni = atoi(argv[1]);
  const int nj = atoi(argv[1]);
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp_tmp_ref, *temp1_init, *temp2_init;

  const int size = ni * nj * sizeof(float);

  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);
  temp1_init = (float*)malloc(size);
  temp2_init = (float*)malloc(size);

  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp2_init[i] = temp2_init[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }

  clock_t start, end;
  start = clock();
  // Execute the CPU-only reference version
  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers
    temp_tmp_ref = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp_ref;
  }
  end = clock();
  printf("CPU-only execution time: %f seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);

  cudaEvent_t start_malloc, start_gpu, end_gpu, end_malloc;
  cudaEventCreate(&start_malloc);
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&end_gpu);
  cudaEventCreate(&end_malloc);

  cudaEventRecord(start_malloc, 0);
  float *temp1, *temp2, *temp_tmp;
  cudaMalloc((void **) &temp1, size);
  cudaMalloc((void **) &temp2, size);
  cudaMalloc((void **) &temp_tmp, size);

  cudaMemcpy(temp1, temp1_init, size, cudaMemcpyHostToDevice);
  cudaMemcpy(temp2, temp2_init, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(num_threads);
  dim3 blocksPerGrid(((ni/2) * (nj/2) + threadsPerBlock.x - 1) / threadsPerBlock.x);

  cudaEventRecord(start_gpu, 0);
  // Execute the modified version using same data
  for (istep=0; istep < nstep; istep++) {
    step_kernel_mod<<<blocksPerGrid, threadsPerBlock>>>(ni, nj, tfac, temp1, temp2);
    cudaDeviceSynchronize();
    // swap the temperature pointers
    temp_tmp = temp1;
    temp1 = temp2;
    temp2= temp_tmp;
  }
  cudaEventRecord(end_gpu, 0);

  cudaMemcpy(temp1_init, temp1, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(temp2_init, temp2, size, cudaMemcpyDeviceToHost);

  float malloc_tot_time, gpu_tot_time;
  cudaEventElapsedTime(&malloc_tot_time, start_malloc, end_malloc);
  cudaEventElapsedTime(&gpu_tot_time, start_gpu, end_gpu);
  printf("GPU execution time: %f ms\n", gpu_tot_time);
  printf("GPU + Memory allocation execution time: %f ms\n", malloc_tot_time);

  cudaFree(temp1);
  cudaFree(temp2);

  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1_init[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1_init[i]-temp1_ref[i]); }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  free( temp1_ref );
  free( temp2_ref );
  free( temp1 );
  free( temp2 );

  return 0;
}
