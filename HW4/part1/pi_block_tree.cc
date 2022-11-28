#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>

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

void init_max_depth(int *max_depth, int world_size){
    // pitfall: do not use sizeof(max_depth), which is the size of the pointer (= 8)
    //          instead, we should use: sizeof(int) * world_size
    memset(max_depth, 0, sizeof(int) * world_size);
    
    for(int i = 0; i < (int) log2(world_size); i++){
        for(int j = 0; j < world_size; j += pow(2, i)){
            max_depth[j]++;
        }
    }
}

void binary_tree_reduction(int original_rank, int max_depth[], long long int &number_in_circle){
    long long int tmp_recv;
    int tmp_rank = original_rank;

    for(int cur_depth = 0; cur_depth < max_depth[0]; cur_depth++, tmp_rank /= 2){
        // synchronization, make sure communication occurs in same depth
        MPI_Barrier(MPI_COMM_WORLD);

        // process does not reach its max depth
        if(cur_depth < max_depth[original_rank]){
            // receiver
            if(tmp_rank % 2 == 0){
                int src = pow(2, cur_depth) * (tmp_rank + 1);
                MPI_Recv(&tmp_recv, 1, MPI_LONG_LONG, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                number_in_circle += tmp_recv;
            }
            // sender
            else{
                int dest = pow(2, cur_depth) * (tmp_rank - 1);
                MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, dest, 0, MPI_COMM_WORLD);
            }
        }
    }
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

    // initialize max depth for each process
    int max_depth[16];
    init_max_depth(max_depth, world_size);

    // TODO: binary tree redunction
    binary_tree_reduction(world_rank, max_depth, number_in_circle);

    if(world_rank == 0){
        // TODO: PI result
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