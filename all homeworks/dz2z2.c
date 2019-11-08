# include "mpi.h"
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>


double cpu_time ( void ) {
  double value;

  //value = ( double ) clock ( ) 
  //      / ( double ) CLOCKS_PER_SEC;
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

int prime_number_mpi ( int start, int end ) {
  int i;
  int j;
  int prime;
  int total;

  total = 0;

  //chunk = (n-1+numThreads-1)/numThreads;
  //start = 2+2*myId;
  //end = (start+chunk-1<n?start + chunk-1:n);
  for ( i = start; i <= end; i++)
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


  
  if(rank==0)
  {
    timestamp ( );
    printf ( "\n" );
    printf ( "PRIME TEST\n" );
  }

    if ( argc !=4 ) {
      n_lo = 1;
      n_hi = 131072;
      n_factor = 2;
    } else {
      n_lo = atoi(argv[1]);
      n_hi = atoi(argv[2]);
      n_factor = atoi(argv[3]);
      //printf ( "My New Rank: %d, n_lo: %d, n_hi: %d n_factor: %d\n", rank, n_lo, n_hi, n_factor);
    }  

  if (rank==0)
  {
    //merenje perfomansi sekvencijalnog koda
    s_time = cpu_time();
    br_s = test(n_lo, n_hi, n_factor );
    s_time = cpu_time()-s_time;
    //sekvencijalne performanse izmerene
  }

  //merenje performansi paralelnog koda
  p_time = cpu_time();
 
  //MPI_Bcast(&n_lo, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  //MPI_Bcast(&n_hi, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  //MPI_Bcast(&n_factor, 1, MPI_INT, 0, MPI_COMM_WORLD); 

  br_p = test_mpi(n_lo, n_hi, n_factor);

  //MPI_Reduce(&br_p, &br_p0, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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
    printf ("Paraleno izbrojano: %d\n ", br_p);
    if (br_s==br_p) printf ("Test PASSED");
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
  int primes=0;
  int ret=0;
  
  int partial_result = 0;
  double ctime;
  int bejbe_const=100;
  int next[2];
  MPI_Status status;
  int sender;
  int tag;

  int chunks_sent=0;
  int chunks_received=0;

  int size;
  int rank;

  if(rank==0)
  {
    printf ( "\n" );
    printf ( "  Call PRIME_NUMBER to count the primes from 1 to N (parallel).\n" );
    
    printf ( "\n" );
    printf ( "         N        Pi          Time\n" );
    printf ( "\n" );
  }

  n = n_lo;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  //za svako n koje obradjujemo, menadzer treba da izdeli posao na chunk-ove brojeva koji se proveravaju, posalje pocetne chunk-ove i onda se vrti u petlji cekajuci odgovor od bilo kojeg workera. 
  //kad dobije odgovor, salje naredni chunk ako postoji, ako ne postoji, onda treba da vrati rezultat i predje na sledece n.
  
  while (n <= n_hi)
  {
    if (rank == 0)
    {
    
      //menadzer code
      primes = 0;
      ctime = cpu_time();
      next[0] = 2;
      next[1] = 2+bejbe_const;
      if (next[1]>n) next[1]=n;
      while (next[0]<=n)/*ima posla da se podeli*/
      {
        /*cekanje da se neki worker javi da je spreman da mu se da posao*/
        MPI_Recv(&partial_result, 1, MPI_INT,  MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        sender = status.MPI_SOURCE;
        tag = status.MPI_TAG;
        if(tag==0)
        {
          //"I am alive" message

        }
        else
        {
          //received unit of work
          //printf("Primio uow: primes = %d od workera %d, poruka imala tag %d\n",partial_result, sender, tag);
          primes+=partial_result;
          chunks_received++;
        }
        MPI_Send(next, 2, MPI_INT, sender, tag, MPI_COMM_WORLD);
        chunks_sent++;
        next[0] = next[1] + 1;
        next[1] = next[0] + bejbe_const;
        if (next[1]>n) next[1]=n;
      }     
      //primanje zaostalih poslova
      while(chunks_received<chunks_sent)
      {
        MPI_Recv(&partial_result, 1, MPI_INT,  MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        primes+=partial_result;
        chunks_received++;
      }

      
      //MPI_Finalize();

      ctime = cpu_time() - ctime;     
      printf("  %8d  %8d  %14f\n", n, primes, ctime);
      //printf("bejbe: %d\n", bejbe_const);
  
      if (n>=n_hi)
      {
        next[0]=next[1]=-1;
        for(i=1; i<size; i++)
        {
          MPI_Send(next, 2, MPI_INT, i, i, MPI_COMM_WORLD); 
        }   
        return primes;    
      }
      else 
      {
        next[0] = 0;
        next[1] = -1;
        for(i=1; i<size; i++)
        {
          MPI_Send(next, 2, MPI_INT, i, i, MPI_COMM_WORLD); 
        } 
      }
    }
    else
    {
      //worker code
      partial_result = 0;
      tag = rank;
      //send work request        
      MPI_Send(&partial_result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); 
      while(1)
      {
        
        //recv sledeceg zadatka
        MPI_Recv(next, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        //provera da li treba dalje da radi ili ne
        if(next[0]==next[1] && next[1]==-1)
        {
          //MPI_Finalize();
          return 0;
        }
        else if (next[0]==0 && next[1]==-1)
        {
          break;
        }
        //ako ima jos posla, izracunavanje
        //tag = tag + size;
        partial_result = prime_number_mpi(next[0], next[1]);
        MPI_Send(&partial_result, 1, MPI_INT, 0, 1, MPI_COMM_WORLD); 
      }   
    }
  n = n * n_factor;
  }
  return primes;  
}



