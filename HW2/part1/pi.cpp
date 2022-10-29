#include<random>
#include<ctime>
#include<stdlib.h>
#include<string.h>
#include<limits.h>
#include<pthread.h>
#include<sys/time.h>
using namespace std;

// shared variables
long long int number_in_circle, number_of_tosses, step;
long thread_cnt;
pthread_mutex_t mutex;

// thread function
void* thread_sum_func(void* rank){
    // task allocation variables
    long my_rank = (long) rank;
    long long int my_first_i = step * my_rank;
    long long int my_last_i = (my_rank == thread_cnt - 1) ? number_of_tosses : my_first_i + step;
    
    // computation variables
    long long int my_valid = 0;
    double x, y, distance_squared;
    unsigned seed = my_rank;

    for(long long int i = my_first_i; i < my_last_i; i++){
        x = 2 * (((double) rand_r(&seed)) / RAND_MAX) - 1;
        y = 2 * (((double) rand_r(&seed)) / RAND_MAX) - 1;

        distance_squared = x * x + y * y;
        if(distance_squared <= 1)
            my_valid++;
    }

    // cirtical section
    pthread_mutex_lock(&mutex);
    number_in_circle += my_valid;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char *argv[]){
    pthread_t* thread_handles;
    // struct timeval start, end;
    // gettimeofday(&start, 0);

    // initialization
    srand(time(NULL));

    thread_cnt = strtol(argv[1], NULL, 10);
    number_in_circle = 0, number_of_tosses = strtoll(argv[2], NULL, 10);

    step = number_of_tosses / thread_cnt;
    thread_handles = (pthread_t*) malloc (thread_cnt * sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);

    // create threads
    for(long thread = 0; thread < thread_cnt; thread++)
        pthread_create(&thread_handles[thread], NULL, thread_sum_func, (void*) thread);
    
    // wait for threads join
    for(long thread = 0; thread < thread_cnt; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    printf("%f\n", 4 * number_in_circle / ((double) number_of_tosses));

    // gettimeofday(&end, 0);
    // int sec = end.tv_sec - start.tv_sec;
    // int usec = end.tv_usec - start.tv_usec;
    // printf("Elapsed time: %f sec\n", (sec+(usec/1000000.0))); 
    
    pthread_mutex_destroy(&mutex);
    free(thread_handles);
    
    return 0;
}