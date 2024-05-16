#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

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

int main(int argc, char **argv)
{    
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank==0){
        printf("Number of nodes: %s\n", world_size);
    }

    int rows_per_node = HEIGHT / world_size;
    int start_row = world_rank * rows_per_node;
    int *sub_image = new int[rows_per_node * WIDTH];
    double start_time = MPI_Wtime();

    //int num_threads = atoi(argv[1]);
    //omp_set_num_threads(num_threads);

    //#pragma omp parallel for shared(sub_image)  
    for (int pos = 0; pos < rows_per_node*WIDTH; pos++)
    {
        int global_pos = (start_row*WIDTH) + pos;
        sub_image[pos] = 0;
        const int row = global_pos / WIDTH;
        const int col = global_pos % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                sub_image[pos] = i;
                break;
            }
        }
    }

    int *image = nullptr;
    if(world_rank==0){
        image = new int[HEIGHT*WIDTH];
    }

    MPI_Gather(sub_image, rows_per_node*WIDTH, MPI_INT, image, rows_per_node*WIDTH, MPI_INT, 0, MPI_COMM_WORLD);

    if(world_rank==0){
        //const auto end = chrono::steady_clock::now();
        double end_time = MPI_Wtime();
        cout << "Time elapsed: "
             << (end_time - start_time)
             << " seconds." << endl;
    
        // Write the result to a file
        ofstream matrix_out;
    
        if (argc < 2)
        {
            cout << "Please specify the output file as a parameter." << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    
        matrix_out.open(argv[1], ios::trunc);
        if (!matrix_out.is_open())
        {
            cout << "Unable to open file." << endl;
            MPI_Abort(MPI_COMM_WORLD, -2);
        }
    
        for (int row = 0; row < HEIGHT; row++)
        {
            for (int col = 0; col < WIDTH; col++)
            {
                matrix_out << image[row * WIDTH + col];
                if (col < WIDTH - 1)
                    matrix_out << ',';
            }
            if (row < HEIGHT - 1)
                matrix_out << endl;
        }
        matrix_out.close();
    
        delete[] image; // It's here for coding style, but useless
    }
    
    delete[] sub_image;
    MPI_Finalize();
    return 0;
}
