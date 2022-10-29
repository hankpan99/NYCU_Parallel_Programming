#include<iostream>
#include<ctime>
using namespace std;

int main(){

    long long int number_in_circle = 0, number_of_tosses = 100000000;

    srand(time(NULL));

    for(int toss = 0; toss < number_of_tosses; toss++){
        double x = (double) rand() / RAND_MAX * 2 - 1;
        double y = (double) rand() / RAND_MAX * 2 - 1;
        double distance_squared = x * x + y * y;
        if(distance_squared <= 1)
            number_in_circle++;
    }
    
    double pi_estimate = 4 * number_in_circle / ((double) number_of_tosses);
    cout << pi_estimate << endl;

    return 0;
}