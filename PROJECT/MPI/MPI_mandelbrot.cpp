#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include "stdio.h" // printf
#include "stdlib.h" // malloc and rand for instance. Rand not thread safe!
#include "time.h"   // time(0) to get random seed
#include "math.h"  // sine and cosine
#include "omp.h"   // openmp library like timing
#include "mpi.h"   // openmp library like timing

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 5000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rows_per_process = HEIGHT / world_size;
    int start_row = world_rank * rows_per_process;
    int *image_part = new int[rows_per_process * WIDTH];

    const auto start = chrono::steady_clock::now();
int numt = atoi(argv[1])
omp_set_num_threads(numt);
    
#pragma omp parallel for shared(image_part) ///for the OMP + MPI version
    for (int pos = 0; pos < rows_per_process * WIDTH; pos++) {
        int global_pos = (start_row * WIDTH) + pos; 
        image_part[pos] = 0;

        const int row = global_pos / WIDTH;
        const int col = global_pos % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++) {
            z = pow(z, 2) + c;
            if (abs(z) >= 2) {
                image_part[pos] = i;
                break;
            }
        }
    }

    int *complete_image = nullptr;
    if (world_rank == 0) {
        complete_image = new int[HEIGHT * WIDTH];
    }

    MPI_Gather(image_part, rows_per_process * WIDTH, MPI_INT,
               complete_image, rows_per_process * WIDTH, MPI_INT,
               0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        const auto end = chrono::steady_clock::now();
        cout << "Time elapsed: "
             << chrono::duration_cast<chrono::milliseconds>(end - start).count()
             << " milliseconds." << endl;

        ofstream matrix_out;

        if (argc < 2) {
            cout << "Please specify the output file as a parameter." << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        matrix_out.open(argv[1], ios::trunc);
        if (!matrix_out.is_open()) {
            cout << "Unable to open file." << endl;
            MPI_Abort(MPI_COMM_WORLD, -2);
        }

        for (int row = 0; row < HEIGHT; row++) {
            for (int col = 0; col < WIDTH; col++) {
                matrix_out << complete_image[row * WIDTH + col];
                if (col < WIDTH - 1)
                    matrix_out << ',';
            }
            if (row < HEIGHT - 1)
                matrix_out << endl;
        }
        matrix_out.close();

        delete[] complete_image;
    }

    delete[] image_part;
    MPI_Finalize();
    return 0;
}
