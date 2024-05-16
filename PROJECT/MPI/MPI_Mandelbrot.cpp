#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <iomanip>
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
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank==0){
        printf("Number of nodes: %s\n", world_size);
    }

    const int total_pixels = HEIGHT * WIDTH;
    const int pixels_per_node = total_pixels / world_size;
    const int start_idx = world_rank * pixels_per_node;
    const int end_idx = (world_rank + 1) * pixels_per_node;

    int *image;
    int *sub_image = new int[pixels_per_node];

    if(world_rank==0){
        image = new int[total_pixels];
    }

    //const auto start = chrono::steady_clock::now();
    start_time = MPI_Wtime();
    for (int pos = start_idx; pos < end_idx; pos++)
    {
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
                sub_image[pos-start_idx] = i;
                break;
            }
        }
    }

    MPI_Gather(sub_image, pixels_per_node, MPI_INT, image, pixels_per_node, MPI_INT, 0, MPI_COMM_WORLD);

    if(world_size==0){
        //const auto end = chrono::steady_clock::now();
        end_time = MPI_Wtime();
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
