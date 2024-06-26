#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

int main(int argc, char* argv[])
{

  // size of input array
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <num_threads>\n", argv[0]);
        return 1;
    }
  int istep;
  int nstep = 200; // number of time steps
int numt = atoi(argv[2]);
 cudaEvent_t start, stop, start1, stop1;     // using cuda events to measure time
  float elapsed_time_ms;       // which is applicable for asynchronous code also

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

cudaEventCreate( &start );   cudaEventCreate( &start1 );  // instrument code to measure start time
  cudaEventCreate( &stop ); cudaEventCreate( &stop1 );
  cudaEventRecord( start1, 0 );

   float *temp1_dev, *temp2_dev, *tmp_tmp_dev;
  cudaMalloc((void **) &temp1_dev, size);
  cudaMalloc((void **) &temp2_dev, size);
  cudaMalloc((void **) &tmp_tmp_dev, size);
  // printf("Malloc done\n");
  

  

  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = (float)rand()/(float)(RAND_MAX/100.0f); //temp1_dev[i] = temp2_dev[i]
  }
// copy data from host to device
  cudaMemcpy(temp1_dev, temp1_ref, size, cudaMemcpyHostToDevice);
  cudaMemcpy(temp2_dev, temp2_ref, size, cudaMemcpyHostToDevice);

//printf("Memcpy done\n");

  dim3 block(numt);
  dim3 grid(((ni/2)*(nj/2)+block.x-1)/block.x);
  // Execute the CPU-only reference version

  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);
  //printf("step_kernel_ref\n");
    // swap the temperature pointers
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }
cudaEventRecord( start, 0 );
  // Execute the modified version using same data
  for (istep=0; istep < nstep; istep++) {
    step_kernel_mod<<<grid,block>>>( ni, nj, tfac, temp1_dev, temp2_dev);
    cudaDeviceSynchronize();
//printf("step_kernel_mod\n");
    // swap the temperature pointers
    tmp_tmp_dev = temp1_dev;
    temp1_dev = temp2_dev;
    temp2_dev= tmp_tmp_dev;
  }
cudaEventRecord( stop, 0 );     // instrument code to measue end time
  cudaEventSynchronize( stop );
cudaMemcpy(temp1, temp1_dev, size, cudaMemcpyDeviceToHost);
cudaMemcpy(temp2, temp2_dev, size, cudaMemcpyDeviceToHost);
//printf("MemcpyDeviceToHost\n");
  cudaEventRecord( stop1, 0 );     // instrument code to measue end time
  cudaEventSynchronize( stop1 );

  cudaEventElapsedTime( &elapsed_time_ms, start, stop );


  printf("Time to calculate results: %f ms.\n", elapsed_time_ms);  // print out execution time
  cudaEventElapsedTime( &elapsed_time_ms, start1, stop1 );
  printf("Time with I/O: %f ms.\n", elapsed_time_ms);



  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);
 // free resources on device
  cudaFree(temp1_dev);
  cudaFree(temp2_dev);
  cudaFree(tmp_tmp_dev);
  free( temp1_ref );
  free( temp2_ref );
  free( temp1 );
  free( temp2 );

  return 0;
}






