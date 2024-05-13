#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

void mandelbrot_ref(int *image)
{
    for (int pos = 0; pos < HEIGHT * WIDTH; pos++)
    {
        image[pos] = 0;

        const int row = pos / WIDTH;
        const int col = pos % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                image[pos] = i;
                break;
            }
        }
    }
}

__global__ void mandelbrot_dev(int *image)
{
    int colIndexWithinTheGrid = blockIdx.x * blockDim.x + threadIdx.x;
	int rowIndexWithinTheGrid = blockIdx.y * blockDim.y + threadIdx.y;

	if (colIndexWithinTheGrid >= WIDTH || rowIndexWithinTheGrid >= HEIGHT)
		return;

	int indexWithinTheGrid = rowIndexWithinTheGrid * WIDTH + colIndexWithinTheGrid;
    if (indexWithinTheGrid < HEIGHT * WIDTH)
    {
        image[indexWithinTheGrid] = 0;

        const int row = indexWithinTheGrid / WIDTH;
        const int col = indexWithinTheGrid % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                image[indexWithinTheGrid] = i;
                break;
            }
        }
    }
}

int main(int argc, char **argv)
{
    // size of input array
    if(argc != 2){
        printf("Usage: %s <N> <num_threads>\n", argv[0]);
        return 1;
    }
    printf("Number of threads = %s\n", argv[1]);
    int num_threads = atoi(argv[1]);
    int *image_ref = new int[HEIGHT * WIDTH];

    cudaEvent_t start_malloc, stop_malloc, start_gpu, stop_gpu;
    float time_tot;

    const int size = HEIGHT * WIDTH * sizeof(int);
    image_ref = (int*)malloc(size);

    cudaEventCreate( &start_malloc );   
    cudaEventCreate( &start_gpu );
    cudaEventCreate( &stop_malloc );  
    cudaEventCreate( &stop_gpu );

    cudaEventRecord( start_malloc, 0 );
    int *image_dev = new int[HEIGHT * WIDTH];
    cudaMalloc((void **) &image_dev, size);
    // copy data from host to device
    cudaMemcpy(image_dev, image_ref, size, cudaMemcpyHostToDevice);
   
    dim3 threadsPerBlock(num_threads, num_threads);
    dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    //const auto start = chrono::steady_clock::now();
    cudaEventRecord( start_gpu, 0 );

    mandelbrot_dev<<<blocksPerGrid,threadsPerBlock>>>(image_dev);
    //cudaDeviceSynchronize();

    cudaEventRecord( stop_gpu, 0 );
    cudaEventSynchronize( stop_gpu );
    cudaMemcpy(image_ref, image_dev, size, cudaMemcpyDeviceToHost);
    cudaEventRecord( stop_malloc, 0 );
    cudaEventSynchronize( stop_malloc );

    cudaEventElapsedTime( &time_tot, start_gpu, stop_gpu );
    printf("Time to calculate results: %f s.\n", time_tot*1000);
    cudaEventElapsedTime( &time_tot, start_malloc, stop_malloc );
    printf("Time with I/O: %f s.\n", time_tot*1000);

    // free resources on device
    cudaFree(image_dev);

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image_ref[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image_ref; // It's here for coding style, but useless
    return 0;
}
