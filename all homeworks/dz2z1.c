# include "mpi.h"
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>


double cpu_time ( void ) {
  double value;

  //value = ( double ) clock ( ) 
  //     / ( double ) CLOCKS_PER_SEC;
  value = (double) omp_get_wtime();

  return value;
}

int prime_number ( int n ) {
  int i;
  int j;
  int prime;
  int total;

  total = 0;

  for ( i = 2; i <= n; i++ )
  {
    prime = 1;
    for ( j = 2; j < i; j++ )
    {
      if ( ( i % j ) == 0 )
      {
        prime = 0;
        break;
      }
    }
    total = total + prime;
  }
  return total;
}

int prime_number_mpi ( int n ) {
  int i;
  int j;
  int prime;
  int total;
  int numThreads;
  int myId;
  int start;

  total = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numThreads);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    //chunk = (n-1+numThreads-1)/numThreads;
    start = 2+2*myId;
    //end = (start+chunk-1<n?start + chunk-1:n);
    for ( i = start; i <= n; i=(i+2*numThreads-1)*(i%2)+(i+1)*((i+1)%2) )
    {
      prime = 1;
      for ( j = 2; j < i; j++ )
      {
        if ( ( i % j ) == 0 )
        {
          prime = 0;
          break;
        }
      }
      total = total + prime;
    }
  return total;
}


void timestamp ( void ){
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}

int test ( int n_lo, int n_hi, int n_factor );
int test_omp (int n_lo, int n_hi, int n_factor);

int main (int argc, char *argv[] ) {
  int n_factor;
  int n_hi;
  int n_lo;

  //MPI vars
  int rank, size;
  int br_p0;

  double s_time, p_time;
  int br_s, br_p;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  if (rank==0)
  {
    timestamp ( );
    printf ( "\n" );
    printf ( "PRIME TEST\n" );

    if ( argc !=4 ) {
      n_lo = 1;
      n_hi = 131072;
      n_factor = 2;
    } else {
      n_lo = atoi(argv[1]);
      n_hi = atoi(argv[2]);
      n_factor = atoi(argv[3]);
    }  

    //merenje perfomansi sekvencijalnog koda
    s_time = cpu_time();
    br_s = test(n_lo, n_hi, n_factor );
    s_time = cpu_time()-s_time;
    //sekvencijalne performanse izmerene
  }

  //merenje performansi paralelnog koda
  p_time = cpu_time();
 
  MPI_Bcast(&n_lo, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&n_hi, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&n_factor, 1, MPI_INT, 0, MPI_COMM_WORLD); 

  br_p = test_mpi(n_lo, n_hi, n_factor);

  MPI_Reduce(&br_p, &br_p0, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  p_time = cpu_time()-p_time;
  //zavrseno merenje performansi paralelnog koda
  if (rank==0)
  {
    printf ( "\n" );
    printf ( "PRIME_TEST\n" );
    printf ( "  Normal end of execution.\n" );
    printf ("Vreme sekvencijalne implementacije: %f\n", s_time);
    printf ("Vreme paralelne     implementacije: %f\n", p_time);  
  
    printf ("Sekvencijalno izbrojano: %d\n", br_s);
    printf ("Paraleno izbrojano: %d\n ", br_p0);
    if (br_s==br_p0) printf ("Test PASSED");
    else printf ("Test FAILED");
    printf ( "\n" );
    timestamp ( );
  } 
  return 0;
}

int test ( int n_lo, int n_hi, int n_factor ) {
  int i;
  int n;
  int primes;
  double ctime;

  printf ( "\n" );
  printf ( "  Call PRIME_NUMBER to count the primes from 1 to N (sequential).\n" );
  printf ( "\n" );
  printf ( "         N        Pi          Time\n" );
  printf ( "\n" );

  n = n_lo;

  while ( n <= n_hi )
  {
    ctime = cpu_time ( );

    primes = prime_number ( n );

    ctime = cpu_time ( ) - ctime;

    printf ( "  %8d  %8d  %14f\n", n, primes, ctime );
    n = n * n_factor;
  }
 
  return primes;
}

int test_mpi ( int n_lo, int n_hi, int n_factor ) {
  int i;
  int n;
  int primes;
  int ret=0;
  int rank;
  double ctime;

  printf ( "\n" );
  printf ( "  Call PRIME_NUMBER to count the primes from 1 to N (parallel).\n" );
  printf ( "\n" );
  printf ( "         N        Pi          Time\n" );
  printf ( "\n" );

  n = n_lo;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  while ( n <= n_hi )
  {
    ctime = cpu_time ( );

    primes = prime_number_mpi ( n );
    //MPI_Reduce(&primes,&ret, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    ctime = cpu_time ( ) - ctime;

    printf ( "  %8d  %8d  %14f\n", n, primes, ctime );
    n = n * n_factor;
  }
 
  return primes;
}



