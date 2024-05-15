#include <stdio.h>
#include <time.h>
#include <mpi.h>
               
#define PI25DT 3.141592653589793238462643

#define INTERVALS 10000000000

int main(int argc, char **argv)
{
    long int i, intervals = INTERVALS;
    double x, dx, f, tot_sum = 0.0, pi, node_sum = 0.0;
    double start_time, end_time;
    int size, rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0){
        printf("Number of intervals: %ld\n", intervals);
    }
  
    dx = 1.0 / (double) intervals;

    start_time = MPI_Wtime();

    for (i = rank; i < intervals; i+=size) {
        x = dx * ((double) (i - 0.5));
        f = 4.0 / (1.0 + x*x);
        node_sum = node_sum + f;
    }

    MPI_Reduce(&node_sum, &tot_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank==0){
        pi = dx*tot_sum;
  
        end_time = MPI_Wtime();
    
        printf("Computed PI %.24f\n", pi);
        printf("The true PI %.24f\n\n", PI25DT);
        printf("Elapsed time (s) = %.2lf\n", (end_time-start_time));
    }
    
    MPI_Finalize();
    return 0;
}
