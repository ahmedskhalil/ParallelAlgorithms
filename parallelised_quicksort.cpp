/*
 * -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: 4; -*-
 * Author        : Ahmed Khalil
 * Created       : 12.04.20
 * 
 * 
 */





#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>
#include <sys/time.h>



#define NUM         1000000000

void swap(double lst[], int x, int y);
int partitioner(double lst[], int first, int last);

void helper(double lst[], int first, int last);
void qsort(double lst[], int size);
int isSorted(double lst[], int size);



struct th_data {
    double* lst; int fst; int dlst; int lv;
};

int doublesComparer(const void* x, const void* b);


int main(int argc, char* argv[]){
    struct timeval start, end;
    double dur;

    srand(time(NULL));

    int NUM0 = NUM;
    if (argc == 2){
        NUM0 = atoi(argv[1]);
    } 

    double* dlstonhold = (double*) malloc(NUM0 * sizeof(double));
    double* dlst       = (double*) malloc(NUM0 * sizeof(double));

    // gen random dlst
    for (int i = 0; i < NUM0; i++){
        dlstonhold[i] = 1.0 * rand() / RAND_MAX;
    }
    memcpy(dlst, dlstonhold, NUM0 * sizeof(double));

    // qsort benchmarking
    gettimeofday(&start, NULL);
    qsort(dlst, NUM0);
    gettimeofday(&end, NULL);

    dur = (  (end.tv_sec * 1000000   + end.tv_usec)
           - (start.tv_sec * 1000000 + start.tv_usec)
          )/ 1000000.0;
    printf("Ordinary quicksort processed in: %lf sec.\n", dur);




    return 0;
}



void swap(double lst[], int x, int y){
    double temp = lst[x];
    lst[x] = lst[y];
    lst[y] = temp;
}

int partitioner(double lst[], int first, int last){
    int b = first;
    int r = (int)(first + (last-first) * (1.0 * rand() / RAND_MAX));
    double pivot = lst[r];
    swap(lst,r,last);
    for (int i = first; i<last; i++){
        if(lst[i] < pivot){
            swap(lst, i, b);
            b++;
        }
    }
    swap(lst, last, b);
    return b;
}

void helper(double lst[], int first, int last){
    if (first >= last) return;
    int b = partitioner(lst, first, last);
    helper(lst, first, b-1);
    helper(lst, b+1, last);
}

void qsort(double lst[], int size){
    helper(lst, 0, size-1);
}

int isSorted(double lst[], int size){
    for (int i = 0; i< size; i++){
        if (lst[i] < lst[i-1]){
            printf("isSorted -> check at %d : %e < %e\n", 
                    i, lst[i], lst[i-1]);
                    return 0;
        }
    }
    return 1; // ascendingly sorted
}

//void  pqsort(double lst[], int size, int thlevel){}
//void* phelper(void* threadarg) __attribute__ ((noreturn)){}