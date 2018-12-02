/**
 * parallel_c1 - parallel and sequential comparative implementation of the
 * relaxation technique for solving partial differential equations.
 *
 *
 * The program parallel_c1.c contains the parallel implementation
 * of the above algorithm.
 * It also contains an identical sequential version of the
 * algorithm for correctness testing and scalability reference.
 *
 * For the parallel part parallel_c1.c implements the above algorithm using a
 * SIMD model with a superstep.
 * For each iteration kk all threads compute the avearages of their elements
 * without the need of communication giving good parallisation. This is achieved by
 * using two separate double arrays - one for reading (iteration k-1) and one
 * for writing (result of iteration k). At the end of each iteration a barrier
 * is used to synchronise the threads before the following iteration.
 * This introduces an element of sequentiality to the program, but the nature
 * of the algorithm requires it.
 *
 * For O(1) complexity, just a simple pointer swap is used to transfer the
 * elements of the write array to the read array at the end of each iteration.
 * If there was at least one element that is unsettled in iteration k, all of
 * the elements are recomputed in the next iteration, otherwise the algorithm
 * is considered to be finished.
 *
 * The barrier at the end of each iteration is customly implemented using
 * POSIX semaphores for maximal portability and good speed.
 *
 * At the end of the program the result of the parallel algorithm is compared with
 * the result of the sequential and if there is an different element an
 * error is returned and both values and their positions in the array
 * are printed to stderr.
 *
 *
 * @(#) $Revision$
 * @(#) $Id$
 * @(#) $Source$
 *
 * Copyright (c) 2018 by Konstantin Simeonov.  All Rights Reserved.
 *
 * Permission to use, copy, modify, and distribute this software and
 * its documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright, this permission notice and text
 * this comment, and the disclaimer below appear in all of the following:
 *
 *       supporting documentation
 *       source copies
 *       source works derived from this source
 *       binaries derived from this source or from derived source
 *
 * KONSTANTIN SIMEONOV DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
 * EVENT SHALL KONSTANTIN SIMEONOV BE LIABLE FOR ANY SPECIAL, INDIRECT OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
 * USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <pthread.h>
#include <memory.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/times.h>
#include <semaphore.h>
#include <mpi.h>

/* critical macros */
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
/*
 *                         Ts
 * calculate speedup Sp = ----
 *                         Tp
 */
#define SPEEDUP(seq_time, par_time) ( \
                (par_time == 0.0) ? 0.0 : seq_time / par_time)
/*
 *                           Sp
 * calculate efficiency E = ----
 *                            p
 */
#define EFFICIENCY(seq_time, par_time, threads) ( \
                SPEEDUP(seq_time, par_time) / threads)
/*
 *                                    1/S_p - 1/p
 * calculate the sequential part e = -------------
 *                                      1 - 1/p
 */
#define SEQ_PART(speedup, threads) ( \
               (((1.0 / speedup) - (1.0 / threads)) / \
               (1.0 - 1.0 / threads)))
/*
 * calculate thread overhead T_O = p*Tp - Ts
 */
#define T_O(seq_time, par_time, threads) ( \
               (threads * par_time) - seq_time)
/*
 *                                   1
 * calculate iso-efficiency E = ------------
 *                               1 + T_O/Ts
 */
#define ISO_EFFICIENCY(seq_time, par_time , threads) ( \
               (1.0 / \
               (1.0 + (T_O(seq_time, par_time, threads) / seq_time))))

/* constants */
#define NUM_NEIGHBOURS (4.0)

#define PARENT_PROCESS (0)

#define TRUE (1U)
#define FALSE (0U)

struct thread_data{
    double precision;
    uint64_t dimension;
    uint8_t tid;
    uint8_t num_threads;
};

typedef struct thread_data thread_data;

struct barrier{
    pthread_mutex_t lock;
    pthread_cond_t continue_cond;
    uint8_t finished;
};

typedef struct barrier barrier;

/* shared variables */
double *seq_val;           /* 2D array of sequential version values */
double *test_seq_val;      /* 2D array of test sequential values */
double *par_val;           /* 2D shared array of test values */
double *test_par_val;      /* 2D shared array of actual values */

/* distributed variables */
double *mpi_process_val;   /* the chunk of the array that process operates on */
int *data_spread;          /* the number of elements allocated to each process */
int *displacement;         /* the location onwards the process calculates elements */

barrier thread_barrier;    /* shared thread barrier with the expected functionality */
_Bool done = FALSE;        /* flag indicating completion of the algorithm */
uint8_t change = FALSE;    /* flag another iteration is needed for the algorithm */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

_Bool verbose = FALSE;     /* verbose level */
static const char *usage = "[-v] [-t] [-M] [-f] test_values_file [-d] dimension "
                           "num_threads precision\n"
                           "\n"
                           "\t-v\t\t\tset to verbose mode\n"
                           "\t-t\t\t\ttime the parallel and sequential execution\n"
                           "\t-f test_values_file\tthe location of the file containing the double array\n"
                           "\t-d dimension\t\tthe dimension of the 2D double array to be read\n"
                           "\t-M\t\t\tcompute the parallel part using Open MPI\n"
                           "\tnum_threads\t\tthe number of parallel threads to use (def: " STRINGIZE(DEFAULT_THREAD_COUNT) ")\n"
                           "\tprecision\t\tthe minimal allowed precision (def: " STRINGIZE(DEFAULT_PRECISION) ")\n";

/* function declarations */

/**
 * A function that synchronises a set of threads. (Acts like a barrier)
 * Using two semaphores - one acting as a lock and the other as a
 * continuation condition the function achieves the desired functionality.
 *              (logic and proof of work is motivated from
 *              http://greenteapress.com/semaphores/LittleBookOfSemaphores.pdf [pages 40-45])
 *
 *      input:
 *        thread_barrier - pointer the the barrier structure
 *        thread_count   - number of computing threads
 *      output:
 *         0 - on success
 *         1 - on error
 */
int
sync_threads(barrier *restrict thread_barrier, uint8_t thread_count);

/**
 * A function that performs the relaxation algorithm in a
 * sequential manner.
 *
 *      input:
 *        dimension - the dimension of the 2D array to be read
 *        precision - the precision of the calculation
 *                              (|old_val - new_val| < precision)
 *      output:
 *        0 - on success
 *        1 - on error
 */
int
df_relax_seq(register uint64_t dimension, register double precision);

/**
 * A function that orchestrates the parallel execution of the
 * relaxation algorithm using POSIX pthreads.
 *
 *      input:
 *          dimension    - the dimension of the 2D array to be read
 *          num_threads  - number of computing threads
 *          precision    - the precision of the calculation
 *                              (|old_val - new_val| < precision)
 *      output:
 *          0 - on success
 *          1 - thread creation error
 *          2 - thread joining error
 *          3 - thread canceled error
 *          4 - condition semaphore initialisation error
 *          5 - lock semaphore initialisation error
 */
int
df_relax(register uint64_t dimension, register uint8_t num_threads, register double precision);

/**
 * A function that orchestrates the parallel execution of the
 * relaxation algorithm using Open MPI.
 *
 *      input:
 *          dimension    - the dimension of the 2D array to be read
 *          num_threads  - number of computing threads
 *          precision    - the precision of the calculation
 *                              (|old_val - new_val| < precision)
 *      output:
 *          0 - on success
 *          1 - thread creation error
 *          2 - thread joining error
 *          3 - thread canceled error
 *          4 - condition semaphore initialisation error
 *          5 - lock semaphore initialisation error
 */
int
df_relax_mpi(register uint64_t dimension, register double precision,
        int rank, int num_procs);

/**
 * A function that reads a 2D input double array from a file.
 *
 *      input:
 *        file      - pointer to the 2D array input file
 *        dimension - the dimension of the 2D array to be read
 *      output:
 */
void
read_file(char* file, uint64_t dimension, _Bool use_mpi, int rank);

/**
 * A function that allocates dynamic memory for the data structures
 * of the program.
 *
 *      input:
 *        dimension - the dimension of the 2D array to be read
 *      output:
 */
void
allocate_test_array(uint64_t dimension, _Bool use_mpi, int num_procs, int rank);


/**
 * A pointer to a function that performs the relaxation algorithm
 * in a parallel manner.
 *
 *      input:
 *        t_data - pointer the the thread specific arguments strucutre
 *      output:
 */
void
mpi_neighbor_average(register int64_t grid_size, register int rank,
                     register int num_procs, register uint64_t dimension, register double precision);

/**
 * A pointer to a function that performs the relaxation algorithm
 * in a parallel manner.
 *
 *      input:
 *        t_data - pointer the the thread specific arguments structure
 *      output:
 */
void *
neighbor_average(void *t_data);

int
main(int argc, char *argv[]) {
    char *num_threads_arg;                      /* num_threads as a string */
    char *precision_arg;                        /* precision as a string */
    int c;                                      /* option */
    uint8_t num_threads;                        /* number of pthreads to run on */
    char* test_val_file_name;                   /* path to the file containing the test array*/
    int relax_exit = 0;                         /* exit code of df_relax() */
    double precision;                           /* minimal allowed precision */
    uint64_t dimension;                         /* the dimension of the array */
    _Bool timed = FALSE;                        /* flag specifying if the run is timed */
    struct tms start_time;                      /* start of the operation time information */
    struct tms end_time;                        /* end of the operation time information */
    clock_t start, end;                         /* start and end times */
    double seq_time = 0.0;                      /* time it took the sequential algo to complete */
    double par_time = 0.0;                      /* time it took the parallel algo to complete */
    long clock_ticks = CLOCKS_PER_SEC;          /* CPU clock ticks per second */
    register uint64_t i;                        /* counter in cache */
    _Bool use_mpi = FALSE;                      /* flag specifying if Open MPI is to be used for parallel */
    int ret;                                    /* return value from Open MPI calls */
    int rank = PARENT_PROCESS;                  /* the rank of the process */
    int num_procs = 0;                          /* total number of Open MPI processes */

    /* Collect arguments */
    while ((c = getopt(argc, argv, "vd:f:tM")) != -1) {
        switch (c) {
            case 'v':
                errno = 0;
                verbose = TRUE;
                break;
            case 't':
                errno = 0;
                timed = TRUE;
                break;
            case 'd':
                errno = 0;
                dimension = (uint64_t) strtol(optarg, NULL, 0);
                if (errno != 0 || dimension < 3 || dimension >= (UINT64_MAX >> 32)) {
                    fprintf(stderr, "FATAL: error in parsing -d dimension: %s\n", optarg);
                    exit(2);
                    /* NOT REACHED */
                }
                break;
            case 'f':
                test_val_file_name = optarg;
                break;
            case 'M':
                use_mpi = TRUE;
                break;
            default:
                fprintf(stderr, "usage: %s", usage);
                exit(7);
                /* NOT REACHED */
        }
    }
    argv += (optind - 1);
    argc -= (optind - 1);
    /*
     * if we are using Open MPI we
     * don't need to specify number of threads
     */
    if(use_mpi){
        if (argc != 2) {
            fprintf(stderr, "usage: %s", usage);
            exit(8);
            /* NOT REACHED */
        }
    }else{
        if (argc != 3) {
            fprintf(stderr, "usage: %s", usage);
            exit(8);
            /* NOT REACHED */
        }
        num_threads_arg = argv[1];
        errno = 0;
        num_threads = (uint8_t) strtoul(num_threads_arg, NULL, 0);
        if (errno != 0 || num_threads < 2 || num_threads > dimension * dimension) {
            fprintf(stderr, "FATAL: num_threads must an integer > 2 and < dimension^2,"
                            " not %s\n", num_threads_arg);
            fprintf(stderr, "usage: %s", usage);
            exit(9);
            /* NOT REACHED */
        }
    }
    if(use_mpi)
        /* get the precision argument */
        precision_arg = argv[1];
    else
        /* get the precision argument */
        precision_arg = argv[2];

    errno = 0;
    precision = strtod(precision_arg, NULL);
    if (errno != 0 || precision <= 0.0) {
        fprintf(stderr, "FATAL: precision must be a rational number > 0"
                        ", not %s\n", precision_arg);
        fprintf(stderr, "usage: %s", usage);
        exit(10);
        /* NOT REACHED */
    }
    /*****************************/
    /* END OF ARGUMENT GATHERING */
    /*****************************/
    if(use_mpi){
        ret = MPI_Init(NULL, NULL);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "FATAL: error initialising Open MPI.\n");
            MPI_Abort(MPI_COMM_WORLD, ret);
            exit(207);
        }
        ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "FATAL: error getting the process rank.\n");
            MPI_Abort(MPI_COMM_WORLD, ret);
            exit(208);
        }
        ret = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "FATAL: error getting the total "
                            "number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, ret);
            exit(209);
        }
    }
    /* allocate enough memory to store the values */
    allocate_test_array(dimension, use_mpi, num_procs, rank);
    /* then read the input array file */
    read_file(test_val_file_name, dimension, use_mpi, rank);

    if(use_mpi){
        if(rank == PARENT_PROCESS){
            if(verbose){
                fprintf(stderr, "Notice: computing with relaxation technique array of size:"
                                " %"PRIu64"x%"PRIu64",\n"
                                                   "\twith precision: %lf, on %d processes in parallel.\n"
                        ,dimension, dimension
                        ,precision, num_procs);
            }
        }
    }else{
        if(verbose){
            fprintf(stderr, "Notice: computing with relaxation technique array of size:"
                            " %"PRIu64"x%"PRIu64",\n"
                                               "\twith precision: %lf, on %u threads in parallel.\n"
                    ,dimension, dimension
                    ,precision, num_threads);
        }
    }
    /*****************************/
    /* SEQUENTIAL PART           */
    /*****************************/
    if(use_mpi){
        if(rank == PARENT_PROCESS){
            if(timed){
                /*
                 * initiate the timed relaxation of the array sequentially
                 */
                if(verbose)
                    fprintf(stderr, "Notice: timing df_relax_seq()\n");
                /* get the CPU clock ticks per sec */
                clock_ticks = sysconf(_SC_CLK_TCK);
                /* start timing */
                start = times(&start_time);
                if (start < 0) {
                    fprintf(stderr, "FATAL: start_clock failed");
                    exit(20);
                    /* NOT REACHED */
                }
                /*
                 * initiate the relaxation of the array sequentially
                 */
                if((relax_exit = df_relax_seq(dimension, precision))){
                    fprintf(stderr, "FATAL: df_relax_seq() failed with exit code:"
                                    "%d\n", relax_exit);
                    exit(12);
                    /* NOT REACHED */
                }
                end = times(&end_time);
                if (start < 0) {
                    fprintf(stderr, "FATAL: end_clock failed");
                    exit(20);
                    /* NOT REACHED */
                }
                seq_time = (double) (end - start) / (clock_ticks);
                /* end timing */
                if(verbose){
                    fprintf(stderr, "Notice: df_relax_seq() successfully finished\n");
                }
                fprintf(stderr, "Notice: df_relax_seq() took %lf seconds to execute\n",
                        seq_time);
                fflush(stderr);
            } else{
                /*
                 * initiate the not timed relaxation of the array sequentially
                 */
                if((relax_exit = df_relax_seq(dimension, precision))){
                    fprintf(stderr, "FATAL: df_relax_seq() failed with exit code:"
                                    "%d\n", relax_exit);
                    exit(12);
                    /* NOT REACHED */
                } else{
                    if(verbose){
                        fprintf(stderr, "Notice: df_relax_seq() successfully finished\n");
                    }
                }
            }
        }
    }else{
        if(timed){
            /*
             * initiate the timed relaxation of the array sequentially
             */
            if(verbose)
                fprintf(stderr, "Notice: timing df_relax_seq()\n");
            /* get the CPU clock ticks per sec */
            clock_ticks = sysconf(_SC_CLK_TCK);
            /* start timing */
            start = times(&start_time);
            if (start < 0) {
                fprintf(stderr, "FATAL: start_clock failed");
                exit(20);
                /* NOT REACHED */
            }
            /*
             * initiate the relaxation of the array sequentially
             */
            if((relax_exit = df_relax_seq(dimension, precision))){
                fprintf(stderr, "FATAL: df_relax_seq() failed with exit code:"
                                "%d\n", relax_exit);
                exit(12);
                /* NOT REACHED */
            }
            end = times(&end_time);
            if (start < 0) {
                fprintf(stderr, "FATAL: end_clock failed");
                exit(20);
                /* NOT REACHED */
            }
            seq_time = (double) (end - start) / (clock_ticks);
            /* end timing */
            if(verbose){
                fprintf(stderr, "Notice: df_relax_seq() successfully finished\n");
            }
            fprintf(stderr, "Notice: df_relax_seq() took %lf seconds to execute\n",
                    seq_time);
            fflush(stderr);
        } else{
            /*
             * initiate the not timed relaxation of the array sequentially
             */
            if((relax_exit = df_relax_seq(dimension, precision))){
                fprintf(stderr, "FATAL: df_relax_seq() failed with exit code:"
                                "%d\n", relax_exit);
                exit(12);
                /* NOT REACHED */
            } else{
                if(verbose){
                    fprintf(stderr, "Notice: df_relax_seq() successfully finished\n");
                }
            }
        }
    }

    /*****************************/
    /* PARALLEL PART             */
    /*****************************/
    if(timed){
        /*
         * initiate the timed relaxation of the array in parallel
         */
        if(verbose)
            fprintf(stderr, "Notice: timing df_relax()\n");
        /* start timing */
        start = times(&start_time);
        if (start < 0) {
            fprintf(stderr, "FATAL: start_clock failed");
            exit(20);
            /* NOT REACHED */
        }
        if(use_mpi){
            if((relax_exit = df_relax_mpi(dimension, precision,
                    rank, num_procs))){
                fprintf(stderr, "FATAL: df_relax() failed with exit code: "
                                "%d\n", relax_exit);
                exit(13);
                /* NOT REACHED */
            } else{
                if(verbose){
                    if(rank == PARENT_PROCESS)
                        fprintf(stderr, "Notice: df_relax_mpi() successfully finished\n");
                }
            }
        }else{
            /*
             * initiate the relaxation of the array in parallel
             */
            if((relax_exit = df_relax(dimension, num_threads, precision))){
                fprintf(stderr, "FATAL: df_relax() failed with exit code: "
                                "%d\n", relax_exit);
                exit(13);
                /* NOT REACHED */
            } else{
                if(verbose){
                    fprintf(stderr, "Notice: df_relax() successfully finished\n");
                }
            }
        }
        /* end timing */
        end = times(&end_time);
        if (end < 0) {
            fprintf(stderr, "FATAL: end_clock failed");
            exit(20);
            /* NOT REACHED */
        }
        if(rank == PARENT_PROCESS){
            par_time = (double) (end - start) / (clock_ticks);
            fprintf(stderr, "Notice: %s took %lf seconds to execute\n",
                    ((use_mpi) ? "df_relax_mpi()" : "df_relax()"), par_time);
            fflush(stderr);
        }
    } else{
        if(use_mpi) {
            if((relax_exit = df_relax_mpi(dimension, precision,
                    rank, num_procs))){
                fprintf(stderr, "FATAL: df_relax() failed with exit code: "
                                "%d\n", relax_exit);
                exit(13);
                /* NOT REACHED */
            } else{
                if(verbose){
                    if(rank == PARENT_PROCESS)
                        fprintf(stderr, "Notice: df_relax_mpi() successfully finished\n");
                }
            }
        } else {
            /*
             * initiate the not timed relaxation of the array in parallel
             */
            if((relax_exit = df_relax(dimension, num_threads, precision))){
                fprintf(stderr, "FATAL: df_relax() failed with exit code: "
                                "%d\n", relax_exit);
                exit(13);
                /* NOT REACHED */
            } else{
                if(verbose){
                    fprintf(stderr, "Notice: df_relax() successfully finished\n");
                }
            }
        }

    }
    /*****************************/
    /* END OF COMPUTATION */
    /*****************************/
    if(use_mpi){
        if(rank == PARENT_PROCESS){
            /* Correctness checking */
            if(verbose){
                fprintf(stderr, "Notice: error checking the resulting sequential and parallel"
                                " %"PRIu64"x%"PRIu64" grid.\n", dimension, dimension);
            }
            /* compare the sequential and parallel arrays */
            for (i = 0; i < dimension * dimension; i++) {
                if(fabs(par_val[i] - seq_val[i]) >= precision) {
                    fprintf(stderr, "\nFATAL: the sequential and the parallel version "
                                    "of the code differ at position: %"PRIu64"\n"
                                    "seq_val[%"PRIu64"]=""%lf != par_val[%"PRIu64"]=%lf\a\n",
                            i, i, seq_val[i], i, par_val[i]);
                    exit(14);
                    /* NOT REACHED */
                }
            }
            /*
             * if timed print scalability information
             */
            if(timed){
                fprintf(stderr, "Notice: parallel speedup of: %lf times --- %u processes --- "
                                "%"PRIu64" elements\n",
                        SPEEDUP(seq_time, par_time), num_procs, dimension * dimension);
                fprintf(stderr, "Notice: process efficiency is: %lf\n",
                        EFFICIENCY(seq_time, par_time, num_procs));
                fprintf(stderr, "Notice: The sequential part of the parallel "
                                "implementation is: e = %lf\n",
                        SEQ_PART(SPEEDUP(seq_time, par_time), num_procs));
                fprintf(stderr, "Notice: work efficiency is: T_O = %lf\n",
                        T_O(seq_time, par_time, num_procs));
                fprintf(stderr, "Notice: iso-efficiency is: E = %lf\n",
                        ISO_EFFICIENCY(seq_time, par_time, num_procs));

            }
            if(verbose){
                fprintf(stderr, "Success!: the parallel version matches the sequential\n");
            }
        }
    }else{
        /* Correctness checking */
        if(verbose){
            fprintf(stderr, "Notice: error checking the resulting sequential and parallel"
                            " %"PRIu64"x%"PRIu64" grid\n", dimension, dimension);
        }
        /* compare the sequential and parallel arrays */
        for (i = 0; i < dimension * dimension; i++) {
            if(fabs(par_val[i] - seq_val[i]) >= precision) {
                fprintf(stderr, "\nFATAL: the sequential and the parallel version "
                                "of the code differ at position: %"PRIu64"\n"
                                "seq_val[%"PRIu64"]=""%lf != par_val[%"PRIu64"]=%lf\a\n",
                        i, i, seq_val[i], i, par_val[i]);
                exit(14);
                /* NOT REACHED */
            }
        }
        /*
         * if timed print scalability information
         */
        if(timed){
            fprintf(stderr, "Notice: parallel speedup of: %lf times --- %u threads --- "
                            "%"PRIu64" elements\n",
                    SPEEDUP(seq_time, par_time), num_threads, dimension * dimension);
            fprintf(stderr, "Notice: thread efficiency is: %lf\n",
                    EFFICIENCY(seq_time, par_time, num_threads));
            fprintf(stderr, "Notice: The sequential part of the parallel "
                            "implementation is: e = %lf\n",
                    SEQ_PART(SPEEDUP(seq_time, par_time), num_threads));
            fprintf(stderr, "Notice: work efficiency is: T_O = %lf\n",
                    T_O(seq_time, par_time, num_threads));
            fprintf(stderr, "Notice: iso-efficiency is: E = %lf\n",
                    ISO_EFFICIENCY(seq_time, par_time, num_threads));

        }
        if(verbose){
            fprintf(stderr, "Success!: the parallel version matches the sequential\n");
        }
    }
    /* memory management */
    free(seq_val);
    free(test_seq_val);
    free(par_val);
    free(test_par_val);

    if(use_mpi)
        MPI_Finalize();
    return 0;
    /* PROGRAM END */
}

/**
 * A function that orchestrates the parallel execution of the
 * relaxation algorithm.
 *
 *      input:
 *          dimension    - the dimension of the 2D array to be read
 *          num_threads  - number of computing threads
 *          precision    - the precision of the calculation
 *                              (|old_val - new_val| < precision)
 *      output:
 *          0 - on success
 *          1 - thread creation error
 *          2 - thread joining error
 *          3 - thread canceled error
 *          4 - condition variable initialisation error
 *          5 - mutex initialisation error
 */
int
df_relax(register uint64_t dimension, register uint8_t num_threads, register double precision){
    register uint8_t i = 0;         /* counter */
    char *thread_ret;               /* the return value of thread_join() */
    thread_data t_data[num_threads];/* the local data on every thread */
    pthread_t thread[num_threads];  /* array of threads */

    /* initialise all the data structures needed for thread synchronisation */
    if(pthread_cond_init(&thread_barrier.continue_cond, NULL)){
        fprintf(stderr, "FATAL: error initializing condition variable in barrier.\n");
        return 4;
    }
    if(pthread_mutex_init(&thread_barrier.lock, NULL)){
        fprintf(stderr, "FATAL: error initializing mutex in barrier.\n");
        return 5;
    }
    thread_barrier.finished = 0U;

    /* start all of the threads */
    for (i = 0; i < num_threads; ++i) {
        /* assign thread specific values */
        t_data[i].tid = i;
        t_data[i].dimension = dimension;
        t_data[i].num_threads = num_threads;
        t_data[i].precision = precision;

        if(pthread_create(&thread[i], NULL, neighbor_average, &t_data[i])) {
            fprintf(stderr, "FATAL: error creating thread: %d \n", i);
            return 1;
            /* NOT REACHED */
        }
        if(verbose)
            fprintf(stderr, "Notice: thread: %d successfully started \n", i);
    }
    if(verbose)
        fprintf(stderr, "Notice: All of the threads started successfully\n");

    /************************************************/

    /* collect the result from the finished threads */
    for (i = 0; i < num_threads; ++i) {
        if(pthread_join(thread[i], (void **) &thread_ret)) {
            fprintf(stderr, "FATAL: error joining thread: %d\n", i);
            return 2;
            /* NOT REACHED */
        }
        if(thread_ret == PTHREAD_CANCELED){
            fprintf(stderr, "FATAL: error pthread-%d got cancelled during"
                            "execution\n", i);
            return 3;
            /* NOT REACHED */
        }
        if(verbose)
            fprintf(stderr, "Notice: thread: %d "
                            "successfully finished executing\n", i);
    }
    if(verbose)
        fprintf(stderr, "Notice: All of the threads successfully finished\n");
    /* Free the pthread types */
    pthread_mutex_destroy(&thread_barrier.lock);
    pthread_cond_destroy(&thread_barrier.continue_cond);
    /* All done */
    return 0;
    /* NOT REACHED */
}
/**
 * A function that orchestrates the parallel execution of the
 * relaxation algorithm using Open MPI.
 *
 *      input:
 *          dimension    - the dimension of the 2D array to be read
 *          num_threads  - number of computing threads
 *          precision    - the precision of the calculation
 *                              (|old_val - new_val| < precision)
 *      output:
 *          0 - on success
 *          1 - failed to broadcast the initial array
 *          2 - failed to gather results from some or
 *                          all of the MPI processes
 *          3 - failed to reduce the changes on each process
 */
int
df_relax_mpi(register uint64_t dimension, register double precision,
        int rank, int num_procs){
    int ret;                       /* return value form the Open MPI calls */
    int64_t size =                 /* the size of the grid */
            dimension * dimension;
    /* arrange data to all processes */
    /* initiate the actual computation */
    if (rank == PARENT_PROCESS) {
        if(verbose)
            fprintf(stderr, "Notice: MPI relaxation initialised with %d"
                            " number of processes\n", num_procs);
    }
    /* send the original sets of memory to all child processes */
    ret = MPI_Bcast(par_val /* pointer to the memory */, size /* num of elements */, MPI_DOUBLE,
                    PARENT_PROCESS /* rank of the dest */, MPI_COMM_WORLD /* node space */);
    if(ret != MPI_SUCCESS) {
        fprintf(stderr, "FATAL: error broadcasting the initial"
                        " values in %s.\n", __func__);
        MPI_Abort(MPI_COMM_WORLD, ret);
        return 1;
    }
    /*
     * perform the relaxation algorithm
     * until all values settle
     */
    while(TRUE){
        /* compute the parents part of the calculation */
        // perform the algorithm
        mpi_neighbor_average(size, rank, num_procs, dimension, precision);
        // gather the result back
        ret = MPI_Allgatherv(mpi_process_val, data_spread[rank], MPI_DOUBLE, par_val,
                data_spread, displacement, MPI_DOUBLE, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "FATAL: failed to perform MPI_Allgatherv"
                            " in  %s.\n", __func__);
            MPI_Abort(MPI_COMM_WORLD, ret);
            return 2;
        }
        ret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "FATAL: failed to perform changes reduction"
                            " in  %s.\n", __func__);
            MPI_Abort(MPI_COMM_WORLD, ret);
            return 3;
        }
        /* if no changes were made on any process - finish */
        if(!change)
            break;
    }
    /* All done */
    return 0;
    /* NOT REACHED */
}
/**
 * A function that performs the relaxation algorithm in a
 * sequential manner.
 *
 *      input:
 *        dimension - the dimension of the 2D array to be read
 *        precision - the precision of the calculation
 *                              (|old_val - new_val| < precision)
 *      output:
 *        0 - on success
 *        1 - on error
 */
int
df_relax_seq(register uint64_t dimension, register double precision){
    register uint64_t i = 0ULL;                          /* counter */
    register uint64_t row = 0ULL;                        /* the row coordinate of the current thread iteration */
    register uint64_t col = 0ULL;                        /* the column coordinate of the current thread iteration */
    register double old_val;                             /* the old value stored at the test array location */
    register double new_val;                             /* the old value to be stored at the test array location */
    register uint64_t grid_size = dimension * dimension; /* the size of the grid */
    double *tmp;                                         /* a pointer to hold the address of the par_val array */

    while (!done) {
        for (i = 0ULL; i < grid_size; i++) {
            row = (i) / dimension;
            col = (i) % dimension;

            if(row == 0ULL || col == 0ULL ||
               row == dimension - 1ULL || col == dimension - 1ULL) {
                continue;
                /* NOT REACHED */
            }
            /*
             * get the neighbouring values.
             *
             * visual representation:
             *
             *           ...     [row - 1,col]     ...
             *       [row,col - 1] [row,col] [row,col + 1]
             *           ...     [row + 1,col]     ...
             */
            old_val = seq_val[row * dimension + col];
            new_val = (seq_val[(row * dimension) + col + 1ULL] +
                       seq_val[(row + 1ULL) * dimension + col] +
                       seq_val[(row - 1ULL) * dimension + col] +
                       seq_val[(row * dimension) + col - 1ULL]) / NUM_NEIGHBOURS;
            /*
             * if precision is achieved, than no more changes needed to that value -
             * continue to next iteration
             */
            if(fabs(old_val - new_val) < precision) {
                test_seq_val[(row * dimension) + col] = old_val;
                continue;
                /* NOT REACHED */
            }
            /* if change was made to the array perform an additional iteration */
            if(!change)
                change = TRUE;
            /* update the element with the average of its neighbours */
            test_seq_val[(row * dimension) + col] = new_val;
        }
        /*
         * check if values in all the positions the threads is responsible for
         * have 'settled'
         */
        if(!change)
            done = TRUE;
        else
            change = FALSE;
        /* update the current values in the grid for the next iteration */
        tmp = seq_val;
        seq_val = test_seq_val;
        test_seq_val = tmp;
    }
    /*
     * reset values for the parallel version
     */
    change = FALSE;
    done = FALSE;

    return 0;
    /* NOT REACHED */
}

/**
 * A pointer to a function that performs the relaxation algorithm
 * in a parallel manner.
 *
 *      input:
 *        t_data - pointer the the thread specific arguments strucutre
 *      output:
 */
void
mpi_neighbor_average(register int64_t grid_size, register int rank,
        register int num_procs, register uint64_t dimension, register double precision){

    register int64_t i = 0;                     /* counter */
    register int64_t j = 0;                     /* position in the sub array indicator */
    register uint64_t row = 0ULL;               /* the row coordinate of the current thread iteration */
    register uint64_t col = 0ULL;               /* the column coordinate of the current thread iteration */
    register double old_val;                    /* the old value stored at the test array location */
    register double new_val;                    /* the old value to be stored at the test array location */
    double vals[5];				/* array of the central and 4 neighboring values */
    register int64_t start = displacement[rank];/* starting array location for this process */
    register int64_t ending = displacement[rank]/* ending array location for this porcess */
            + data_spread[rank];

    /* reset the changes indicator to false */
    change = FALSE;

    for (i = start; i < ending; i++, j++) {
        /* get the row and the column of the array */
        row = (i) / dimension;
        col = (i) % dimension;
        /*
         * if we hit an edge value continue to the next iteration and
         * mark the value as needing no change
         */
        if(row == 0ULL || col == 0ULL ||
           row == dimension - 1ULL || col == dimension - 1ULL){
            mpi_process_val[j] = par_val[row * dimension + col];
            continue;
            /* NOT REACHED */
        }
        /*
         * get the neighbouring values from shared memory and store them in
         * in the local memory.
         *
         * visual representation:
         *
         *           ...     [row - 1,col]     ...
         *       [row,col - 1] [row,col] [row,col + 1]
         *           ...     [row + 1,col]     ...
         */
        vals[0] = par_val[row * dimension + col];
        vals[1] = par_val[(row * dimension) + col + 1ULL];
        vals[2] = par_val[(row + 1ULL) * dimension + col];
        vals[3] = par_val[(row - 1ULL) * dimension + col];
        vals[4] = par_val[(row * dimension) + col - 1ULL];
        /* store the old value at the location */
        old_val = vals[0];
        /* compute the new value at that location */
        new_val = (vals[1] +
                   vals[2] +
                   vals[3] +
                   vals[4]) / NUM_NEIGHBOURS;
        /*
         * if precision is achieved, than no more changes needed to that value -
         * continue to next iteration
         */
        if(fabs(old_val - new_val) < precision){
            mpi_process_val[j] = old_val;
            continue;
            /* NOT REACHED */
        }
        if(!change)
            change = TRUE;
        /* update the test array */
        mpi_process_val[j] = new_val;
    }
    return;
    /* NOT REACHED */
}

/**
 * A pointer to a function that performs the relaxation algorithm
 * in a parallel manner.
 *
 *      input:
 *        t_data - pointer the the thread specific arguments strucutre
 *      output:
 */
void *
neighbor_average(void *t_data){
    register uint64_t i = 0ULL;                 /* counter */
    thread_data *data = (thread_data *) t_data; /* get the thread specific data*/
    register uint64_t grid_size =               /* the size of the grid */
        data->dimension * data->dimension;
    register uint64_t row = 0ULL;               /* the row coordinate of the current thread iteration */
    register uint64_t col = 0ULL;               /* the column coordinate of the current thread iteration */
    register double old_val;                    /* the old value stored at the test array location */
    register double new_val;                    /* the old value to be stored at the test array location */
    double vals[5];				/* array of the central and 4 neighboring values */

    while (!done) {
        /* create an array of threads which execute neighbor_average() iterations time */
        for (i = data->tid; i < grid_size; i += data->num_threads) {
            /* get the row and the column of the array */
            row = (i) / data->dimension;
            col = (i) % data->dimension;
            /*
             * if we hit an edge value continue to the next iteration and
             * mark the value as needing no change
             */
            if(row == 0ULL || col == 0ULL ||
                row == data->dimension - 1ULL || col == data->dimension - 1ULL){
                continue;
                /* NOT REACHED */
            }
            /*
             * get the neighbouring values from shared memory and store them in
             * in the local memory.
             *
             * visual representation:
             *
             *           ...     [row - 1,col]     ...
             *       [row,col - 1] [row,col] [row,col + 1]
             *           ...     [row + 1,col]     ...
             */
            vals[0] = par_val[row * data->dimension + col];
            vals[1] = par_val[(row * data->dimension) + col + 1ULL];
            vals[2] = par_val[(row + 1ULL) * data->dimension + col];
            vals[3] = par_val[(row - 1ULL) * data->dimension + col];
            vals[4] = par_val[(row * data->dimension) + col - 1ULL];
            /* store the old value at the location */
            old_val = vals[0];
            /* compute the new value at that location */
            new_val = (vals[1] +
                       vals[2] +
                       vals[3] +
                       vals[4]) / NUM_NEIGHBOURS;
            /*
             * if precision is achieved, than no more changes needed to that value -
             * continue to next iteration
             */
            if(fabs(old_val - new_val) < data->precision){
                test_par_val[(row * data->dimension) + col] = old_val;
                continue;
                /* NOT REACHED */
            }
            /* if change was made to the array perform an additional iteration */
            if(!change){
                /* CRITICAL */
                pthread_mutex_lock(&mutex);
                change = TRUE;
                pthread_mutex_unlock(&mutex);
                /* END CRITICAL */
            }
            /* update the test array */
            test_par_val[(row * data->dimension) + col] = new_val;
        }
        /* BARRIER
         * wait for all threads to finish their part of the work
         * on the current grid
         */
        if(sync_threads(&thread_barrier, data->num_threads)){
            /* on error with the thread sync cancel the thread */
            fprintf(stderr, "FATAL: failed to sync threads in %s\n",
                    __func__);
            pthread_cancel(pthread_self());
            /* NOT REACHED */
        }
        /* CONTINUE */
    }

    return NULL;
    /* NOT REACHED */
}
/**
 * A function that synchronises all threads. (Acts like a barrier)
 * Using a mutex acting as a lock and a condition variable we achieve the
 * desired functionality.
 * This is done due to the low portability of pthread_barrier and its often
 * unspecified behaviour.
 *
 *      input:
 *        thread_barrier - pointer the the barrier structure
 *        thread_count   - number of computing threads
 *      output:
 *         0 - on success
 *         1 - on error
 */
int
sync_threads(barrier *restrict thread_barrier, uint8_t thread_count){
    double *tmp;  /* a pointer to hold the address of the par_val array */

    pthread_mutex_lock(&thread_barrier->lock);
    thread_barrier->finished++;
    if((thread_barrier->finished) == thread_count){
        /* update the current values in the grid for the next iteration */
        tmp = par_val;
        par_val = test_par_val;
        test_par_val = tmp;
        /*
         * check if values in all the positions the threads is responsible for
         * have 'settled'
         */
        if(!change)
            done = TRUE;
        else
            change = FALSE;
        thread_barrier->finished = 0U;
        pthread_cond_broadcast(&thread_barrier->continue_cond);
    }else{
        if(pthread_cond_wait(&thread_barrier->continue_cond, &thread_barrier->lock)) {
            /* on error with the thread sync cancel the thread */
            fprintf(stderr, "FATAL: condition variable failed in %s\n",
                    __func__);
            return 1;
        }
    }
    pthread_mutex_unlock(&thread_barrier->lock);

    return 0;
}

/**
 * A function that allocates dynamic memory for the data structures
 * of the program.
 *
 *      input:
 *        dimension - the dimension of the 2D array to be read
 *      output:
 */
void
allocate_test_array(uint64_t dimension, _Bool use_mpi, int num_procs, int rank){
    register int i;
    /* configure the size of the shared memory object */
    par_val = (double *) malloc(dimension * dimension * sizeof(double));
    if(par_val == NULL){
        fprintf(stderr, "FATAL: Error allocating memory for test_val\n");
        exit(243);
        /* NOT REACHED */
    }
    test_par_val = (double *) malloc(dimension * dimension * sizeof(double));
    if(test_par_val == NULL){
        fprintf(stderr, "FATAL: Error allocating memory for real_val\n");
        exit(243);
        /* NOT REACHED */
    }
    /* allocate the array of sequential values */
    seq_val = (double *) malloc(dimension * dimension * sizeof(double));
    if(seq_val == NULL){
        fprintf(stderr, "FATAL: Error allocating memory for test_val\n");
        exit(243);
        /* NOT REACHED */
    }
    test_seq_val = (double *) malloc(dimension * dimension * sizeof(double));
    if(test_seq_val == NULL) {
        fprintf(stderr, "FATAL: Error allocating memory for test_seq_val\n");
        exit(243);
        /* NOT REACHED */
    }
    if(use_mpi){
        if(rank == PARENT_PROCESS){
            mpi_process_val = (double *) malloc(((dimension * dimension) / num_procs
                    + (dimension * dimension) % num_procs) * sizeof(double));
            if(mpi_process_val == NULL) {
                fprintf(stderr, "FATAL: Error allocating memory for mpi_process_val\n");
                exit(243);
                /* NOT REACHED */
            }
        }else{
            mpi_process_val = (double *) malloc(((dimension * dimension) / num_procs) * sizeof(double));
            if(mpi_process_val == NULL) {
                fprintf(stderr, "FATAL: Error allocating memory for mpi_process_val\n");
                exit(243);
                /* NOT REACHED */
            }
        }
        data_spread = (int *) malloc(num_procs * sizeof(int));
        if(mpi_process_val == NULL) {
            fprintf(stderr, "FATAL: Error allocating memory for data_spread\n");
            exit(243);
            /* NOT REACHED */
        }
        data_spread[0] = (dimension * dimension) / num_procs + (dimension * dimension) % num_procs;
        for (i = 1; i < num_procs; ++i){
            data_spread[i] = dimension * dimension / num_procs;
        }

        displacement = (int *) malloc(num_procs * sizeof(int));
        displacement[0] = 0;
        for (i = 1; i < num_procs; ++i){
            displacement[i] = displacement[i - 1] + data_spread[i - 1];
        }
    }
}
/**
 * A function that reads a 2D input double array from a file.
 *
 *      input:
 *        file      - pointer to the 2D array input file
 *        dimension - the dimension of the 2D array to be read
 *      output:
 */
void
read_file(char* test_val_file_name, uint64_t dimension, _Bool use_mpi, int rank){
    register uint64_t i = 0ULL;         /* counter */
    int size = (int) dimension * 50;	/* the size of a line */
    char line[size];	                /* string to hold each line */
    char* tch;                          /* token pointer */
    register double element;            /* the floating  point value of each element */
    FILE* test_val_file = NULL;         /* test file */

    if(use_mpi){
        if(rank == PARENT_PROCESS){
            test_val_file = fopen(test_val_file_name, "r");
            if(!test_val_file){
                fprintf(stderr, "FATAL: incorrect path to test values file: %s\n", test_val_file_name);
                exit(1);
            }
            if(verbose)
                fprintf(stderr, "Notice: Reading input file\n");
            /* read file line by line */
            while(fgets(line, size, test_val_file) != NULL){
                tch = strtok(line, " ");
                while(tch != NULL){
                    /* Store each double in a token */
                    element = strtod(tch, NULL);
                    par_val[i] = element;
                    test_par_val[i] = element;
                    seq_val[i] = element;
                    test_seq_val[i] = element;
                    //if (errno != 0) {
                    //    fprintf(stderr, "Notice: failed reading double at location: %"PRIu64
                    //            " with value: %lf\n", i, par_val[i]);
                    //    exit(211);
                    //    /* NOT REACHED */
                    //}
                    i++;
                    tch = strtok(NULL, " ");
                }
            }
            fclose(test_val_file);
        }
    }else{
        test_val_file = fopen(test_val_file_name, "r");
        if(!test_val_file){
            fprintf(stderr, "FATAL: incorrect path to test values file: %s\n", test_val_file_name);
            exit(1);
        }
        if(verbose)
            fprintf(stderr, "Notice: Reading input file\n");
        /* read file line by line */
        while(fgets(line, size, test_val_file) != NULL){
            tch = strtok(line, " ");
            while(tch != NULL){
                /* Store each double in a token */
                element = strtod(tch, NULL);
                par_val[i] = element;
                test_par_val[i] = element;
                seq_val[i] = element;
                test_seq_val[i] = element;
                if (errno != 0) {
                    fprintf(stderr, "Notice: failed reading double at location: %"PRIu64
                            " with value: %lf\n", i, par_val[i]);
                    exit(211);
                    /* NOT REACHED */
                }
                i++;
                tch = strtok(NULL, " ");
            }
        }
        fclose(test_val_file);
    }
}
