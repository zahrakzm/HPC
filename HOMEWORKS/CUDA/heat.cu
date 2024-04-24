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

// define index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for ( int index = i; index<(nj-2)*(ni-2);index+=blockDim.x*gridDim.x){

int ix=index%(ni-2)+1;
int jx=index/(nj-2)+1;
//printf("j = %d\n", jx);
//printf("i = %d\n", ix);


i00 = I2D(ni, ix, jx);
im10 = I2D(ni, ix-1, jx);
ip10 = I2D(ni, ix+1, jx);
i0m1 = I2D(ni, ix, jx-1);
i0p1 = I2D(ni, ix, jx+1);

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
  cudaEvent_t start_malloc, start_gpu, end_gpu, end_malloc;
  float malloc_tot_time, gpu_tot_time;

  // Specify our 2D dimensions
  const int ni = atoi(argv[1]);
  const int nj = atoi(argv[1]);
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

  const int size = ni * nj * sizeof(float);

  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);
  temp1 = (float*)malloc(size);
  temp2 = (float*)malloc(size);

  cudaEventCreate(&start_malloc);
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&end_gpu);
  cudaEventCreate(&end_malloc);

  cudaEventRecord(start_malloc, 0);
  float *temp1_dev, *temp2_dev, *temp_tmp_dev;
  cudaMalloc((void **) &temp1_dev, size);
  cudaMalloc((void **) &temp2_dev, size);
  cudaMalloc((void **) &temp_tmp_dev, size);

  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }

  cudaMemcpy(temp1_dev, temp1_ref, size, cudaMemcpyHostToDevice);
  cudaMemcpy(temp2_dev, temp2_ref, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(num_threads);
  dim3 blocksPerGrid(((ni/2)*(nj/2) + threadsPerBlock.x - 1) / threadsPerBlock.x);

  //clock_t start, end;
  //start = clock();
  // Execute the CPU-only reference version
  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }
  //end = clock();
  //printf("CPU-only execution time: %f seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);

  // dim3 threadsPerBlock(num_threads);
  // dim3 blocksPerGrid(((ni/2) * (nj/2) + threadsPerBlock.x - 1) / threadsPerBlock.x);

  cudaEventRecord(start_gpu, 0);
  // Execute the modified version using same data
  for (istep=0; istep < nstep; istep++) {
    step_kernel_mod<<<blocksPerGrid, threadsPerBlock>>>(ni, nj, tfac, temp1_dev, temp2_dev);
    cudaDeviceSynchronize();
    // swap the temperature pointers
    temp_tmp_dev = temp1_dev;
    temp1_dev = temp2_dev;
    temp2_dev = temp_tmp_dev;
  }
  cudaEventRecord(end_gpu, 0);
  cudaEventSynchronize(end_gpu);

  cudaMemcpy(temp1, temp1_dev, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(temp2, temp2_dev, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(end_malloc, 0);
  cudaEventSynchronize(end_malloc);

  cudaEventElapsedTime(&malloc_tot_time, start_malloc, end_malloc);
  cudaEventElapsedTime(&gpu_tot_time, start_gpu, end_gpu);
  printf("GPU execution time: %f seconds\n", gpu_tot_time*1000);
  printf("GPU + Memory allocation execution time: %f seconds\n", malloc_tot_time*1000);

  float maxError = 0;
  // Output should always be stored in the temp1_host and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1_host[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1_host[i]-temp1_ref[i]); }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  cudaFree(temp1);
  cudaFree(temp2);
  cudaFree(temp_tmp);
  free( temp1_ref );
  free( temp2_ref );
  free( temp1 );
  free( temp2 );

  return 0;
}
