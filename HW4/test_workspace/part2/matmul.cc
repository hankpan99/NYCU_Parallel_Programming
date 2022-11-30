#include <mpi.h>
#include <stdio.h>
#include <string.h>
using namespace std;

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);	/* get current process id */

    if(world_rank == 0){
        // read header
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        int tmp_n = *n_ptr, tmp_m = *m_ptr, tmp_l = *l_ptr;

        // malloc matrix
        *a_mat_ptr = (int*) malloc(sizeof(int) * tmp_n * tmp_m);
        *b_mat_ptr = (int*) malloc(sizeof(int) * tmp_m * tmp_l);

        // read file and store into matrix
        for(int i = 0; i < tmp_n; i++)
            for(int j = 0; j < tmp_m; j++)
                scanf("%d", (*a_mat_ptr + i * tmp_m + j));

        for(int i = 0; i < tmp_m; i++)
            for(int j = 0; j < tmp_l; j++)
                scanf("%d", (*b_mat_ptr + i * tmp_l + j));
    }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    // MPI init
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);	/* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);	/* get number of processes */

    // variables init
    int numworkers = world_size - 1;
    int averow = n / numworkers;
    int extra = n % numworkers;
    int offset = 0, rows = 0;

    if(world_rank == 0){
        // malloc the result matrix 
        int *c_mat = (int*) malloc(sizeof(int) * n * l);

        // send a, b matrix for computation
        for(int dest = 1; dest <= numworkers; dest++){
            // pitfall: need to send matrix size to workers
            MPI_Send(&m, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

            rows = (dest <= extra) ? averow + 1 : averow;

            MPI_Send(&offset, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&a_mat[offset * m], rows * m, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&b_mat[0], m * l, MPI_INT, dest, 0, MPI_COMM_WORLD);
            
            offset += rows;
        }

        // receive the result matrix
        for(int src = 1; src <= numworkers; src++){
            MPI_Recv(&offset, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rows, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&c_mat[offset * l], rows * l, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // output the matrix result
        for(int i = 0; i < n; i++){
            for(int j = 0; j < l; j++){
                printf("%d ", c_mat[i * l + j]);
            }
            printf("\n");
        }

        // free memory
        free(c_mat);
    }
    else{
        // pitfall:
        int M, L;
        MPI_Recv(&M, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&L, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // receive headers
        MPI_Recv(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // malloc matrixes
        int *a = (int*) malloc(sizeof(int) * rows * M);
        int *b = (int*) malloc(sizeof(int) * M * L);
        int *c = (int*) malloc(sizeof(int) * rows * L);
        
        // receive a, b matrix
        MPI_Recv(&a[0], rows * M, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&b[0], M * L, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // matrix multiplication
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < L; j++){
                c[i * L + j] = 0;

                for(int k = 0; k < M; k++){
                    c[i * L + j] += a[i * M + k] * b[k * L + j];
                }
            }
        }

        // send results
        MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&c[0], rows * L, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // free memory
        free(a);
        free(b);
        free(c);
    }
}

void destruct_matrices(int *a_mat, int *b_mat){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);	/* get current process id */

    if(world_rank == 0){
        free(a_mat);
        free(b_mat);
    }
}