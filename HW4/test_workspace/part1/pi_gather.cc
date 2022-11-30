#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

long long int toss_func(int rank, int size, long long int tosses){
    // task allocation variables
    long long int step = tosses / size;
    long long int my_first_i = step * rank;
    long long int my_last_i = (rank == size - 1) ? tosses : my_first_i + step;
    
    // computation variables
    long long int my_valid = 0;
    double x, y, distance_squared;
    unsigned seed = time(NULL) * rank;

    for(long long int i = my_first_i; i < my_last_i; i++){
        x = 2 * (((double) rand_r(&seed)) / RAND_MAX) - 1;
        y = 2 * (((double) rand_r(&seed)) / RAND_MAX) - 1;

        distance_squared = x * x + y * y;
        if(distance_squared <= 1)
            my_valid++;
    }

    return my_valid;
}

int main(int argc, char **argv){
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);	/* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);	/* get number of processes */

    srand(time(NULL) * world_rank);
    long long int number_in_circle = toss_func(world_rank, world_size, tosses);

    // TODO: use MPI_Gather

    // pitfall: recv_count is the number of data receive from each process,
    //          rather than the summation of data from all processes
    // man page "recvcount":  Number of elements for any single receive (integer, significant only at root).
    long long int tmp_recv[16];
    MPI_Gather(&number_in_circle, 1, MPI_LONG_LONG, tmp_recv, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if(world_rank == 0){
        // TODO: PI result
        number_in_circle = 0;
        for(int i = 0; i < world_size; i++){
            number_in_circle += tmp_recv[i];
        }

        pi_result = 4 * number_in_circle / ((double) tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
