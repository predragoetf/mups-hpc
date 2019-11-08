# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <math.h>
# include <omp.h>

#define ACCURACY 0.01
//const double ACCURACY = 0.01;

int main ( int argc, char *argv[] );
void compute ( int np, int nd, double pos[], double vel[], 
  double mass, double f[], double *pot, double *kin );
void compute_parallel ( int np, int nd, double pos[], double vel[], 
  double mass, double f[], double *pot, double *kin );
double cpu_time ( );
double dist ( int nd, double r1[], double r2[], double dr[] );
void initialize ( int np, int nd, double pos[], double vel[], double acc[] );
void r8mat_uniform_ab ( int m, int n, double a, double b, int *seed, double r[] );
void timestamp ( );
void update ( int np, int nd, double pos[], double vel[], double f[], 
  double acc[], double mass, double dt );
void update_parallel ( int np, int nd, double pos[], double vel[], double f[], 
  double acc[], double mass, double dt );

int check_equal_m( double a[], double b[], int np, int nd)
{
  
  int equal = 1;
  int i;
  for (i=0;i<np*nd;i++)
  {
    if ((a[i]-b[i])*(a[i]-b[i])>ACCURACY*ACCURACY) 
    {
      equal = 0;
      break;
    }
  }
  return equal;
}

int check_equal(double a, double b)
{
  return ( (a-b)*(a-b)<ACCURACY*ACCURACY );
}

int main ( int argc, char *argv[] )
{
  double *acc, *acc_par;
  double ctime;
  double dt;
  double e0, e0_par;
  double *force, *force_par;
  int i,j;
  int id;
  double kinetic, kinetic_par;
  double mass = 1.0;
  int nd;
  int np;
  double *pos, *pos_par;
  double potential, potential_par;
  int step;
  int step_num;
  int step_print;
  int step_print_index;
  int step_print_num;
  double *vel, *vel_par;

  timestamp ( );
  printf ( "\n" );
  printf ( "MD\n" );
  printf ( "  C version\n" );
  printf ( "  A molecular dynamics program.\n" );
/*
  Get the spatial dimension.
*/
  if ( 1 < argc )
  {
    nd = atoi ( argv[1] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter ND, the spatial dimension (2 or 3).\n" );
    scanf ( "%d", &nd );
  }
//
//  Get the number of particles.
//
  if ( 2 < argc )
  {
    np = atoi ( argv[2] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter NP, the number of particles (500, for instance).\n" );
    scanf ( "%d", &np );
  }
//
//  Get the number of time steps.
//
  if ( 3 < argc )
  {
    step_num = atoi ( argv[3] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter STEP_NUM, the number of time steps (500 or 1000, for instance).\n" );
    scanf ( "%d", &step_num );
  }
//
//  Get the time steps.
//
  if ( 4 < argc )
  {
    dt = atof ( argv[4] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter DT, the size of the time step (0.1, for instance).\n" );
    scanf ( "%lf", &dt );
  }
/*
  Report.
*/
  printf ( "\n" );
  printf ( "  ND, the spatial dimension, is %d\n", nd );
  printf ( "  NP, the number of particles in the simulation, is %d\n", np );
  printf ( "  STEP_NUM, the number of time steps, is %d\n", step_num );
  printf ( "  DT, the size of each time step, is %f\n", dt );
/*
  Allocate memory.
*/
  acc = ( double * ) malloc ( nd * np * sizeof ( double ) );
  force = ( double * ) malloc ( nd * np * sizeof ( double ) );
  pos = ( double * ) malloc ( nd * np * sizeof ( double ) );
  vel = ( double * ) malloc ( nd * np * sizeof ( double ) );

  acc_par = ( double * ) malloc ( nd * np * sizeof ( double ) );
  force_par = ( double * ) malloc ( nd * np * sizeof ( double ) );
  pos_par = ( double * ) malloc ( nd * np * sizeof ( double ) );
  vel_par = ( double * ) malloc ( nd * np * sizeof ( double ) );

/*
  This is the main time stepping loop:
    Compute forces and energies,
    Update positions, velocities, accelerations.
*/
  printf ( "\n" );
  printf ( "  At each step, we report the potential and kinetic energies.\n" );
  printf ( "  The sum of these energies should be a constant.\n" );
  printf ( "  As an accuracy check, we also print the relative error\n" );
  printf ( "  in the total energy.\n" );
  printf ( "\n" );
  

  step_print = 0;
  step_print_index = 0;
  step_print_num = 10;


  /*Initialize the starting data vectors for both sequential and parallel version*/
  initialize ( np, nd, pos, vel, acc );
  for(j=0;j<np;j++)
    for(i=0;i<nd;i++)
    {
      pos_par[i+j*nd]=pos[i+j*nd];
      vel_par[i+j*nd]=vel[i+j*nd];
      acc_par[i+j*nd]=acc[i+j*nd];
    }
  
  ctime = cpu_time ( );

  for ( step = 0; step <= step_num; step++ )
  {
    if ( step == 0 )
    {
      printf ( "---------------------SEQUENTIAL COMPUTATION---------------------------- \n");
      printf ( "      Step      Potential       Kinetic        (P+K-E0)/E0\n" );
      printf ( "                Energy P        Energy K       Relative Energy Error\n" );
      printf ( "\n" );
      
    }
    else
    {
      update ( np, nd, pos, vel, force, acc, mass, dt );
    }

    compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );

    if ( step == 0 )
    {
      e0 = potential + kinetic;
    }

    if ( step == step_print )
    {
      printf ( "  %8d  %14f  %14f  %14e\n", step, potential, kinetic,
       ( potential + kinetic - e0 ) / e0 );
      step_print_index = step_print_index + 1;
      step_print = ( step_print_index * step_num ) / step_print_num;
    }

  }
/*
  Report sequential timing.
*/
  ctime = cpu_time ( ) - ctime;
  printf ( "\n" );
  printf ( "  Elapsed cpu time (sequential): %f seconds.\n", ctime );
  printf ( "\n" );

/*PARALLEL*/
  step_print = 0;
  step_print_index = 0;
  step_print_num = 10;
  ctime = cpu_time();

  for ( step = 0; step <= step_num; step++ )
  {
    if ( step == 0 )
    {
      printf ( "---------------------PARALLEL COMPUTATION---------------------------- \n");
      printf ( "      Step      Potential       Kinetic        (P+K-E0)/E0\n" );
      printf ( "                Energy P        Energy K       Relative Energy Error\n" );
      printf ( "\n" );
    }
    else
    {
      update_parallel ( np, nd, pos_par, vel_par, force_par, acc_par, mass, dt );
      //update ( np, nd, pos_par, vel_par, force_par, acc_par, mass, dt );
    }

    compute_parallel ( np, nd, pos_par, vel_par, mass, force_par, &potential_par, &kinetic_par );
    //compute ( np, nd, pos_par, vel_par, mass, force_par, &potential, &kinetic );

    if ( step == 0 )
    {
      e0_par = potential_par + kinetic_par;
    }

    if ( step == step_print )
    {
      printf ( "  %8d  %14f  %14f  %14e\n", step, potential_par, kinetic_par,
       ( potential_par + kinetic_par - e0_par ) / e0_par );
      step_print_index = step_print_index + 1;
      step_print = ( step_print_index * step_num ) / step_print_num;
    }

  }

/*
  Report parallel timing.
*/
ctime = cpu_time ( ) - ctime;
printf ( "\n" );
printf ( "  Elapsed cpu time (parallel): %f seconds.\n", ctime );
printf ( "\n" );

/*----------Compare results------------*/
printf("\n");
printf("-----------COMPARE RESULTS-------\n");
printf("\n");
int equal = 1;
/*Compare position matrixes*/
int b = check_equal_m(pos, pos_par, np, nd);
if (b) printf("Pozicione matrice jednake\n");
else 
{
  printf("Pozicione matrice nisu jednake!\n");
  equal=0;
}

/*Compare velocity matrixes*/
b = check_equal_m(vel, vel_par, np, nd);
if (b) printf("Brzinske matrice jednake\n");
else 
{
  printf("Brzinske matrice nisu jednake\n");
  equal=0;
}

/*Compare acceleration matrixes*/
b = check_equal_m(acc, acc_par, np, nd);
if (b) printf("Ubrzanjske matrice jednake\n");
else 
{
  printf("Ubrzanjske matrice nisu jednake\n");
  equal=0;
}

/*Compare total potential energy*/
b = check_equal(potential, potential_par);
if (b) printf("Potencijalne energije jednake\n");
else
{
  printf("Potencijalne energije nisu jednake\n");
  equal=0;
}

/*Compare total kinetic energy*/
b = check_equal(kinetic, kinetic_par);
if (b) printf("Kineticke energije jednake\n");
else
{
  printf("Kineticke energije nisu jednake\n");
  equal=0;
}

/*Compare relative errors*/

/*Print final verdict*/
printf("-------------------------------------\n");
printf("\nFinal verdict:\n");
printf("\n");
if (equal) printf("Test PASSED\n");
else printf("Test FAILED\n");

/*-------------------------------------*/

/*
  Free memory.
*/
  free ( acc );
  free ( force );
  free ( pos );
  free ( vel );

  free ( acc_par );
  free ( force_par );
  free ( pos_par );
  free ( vel_par );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "MD\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}

void compute ( int np, int nd, double pos[], double vel[], double mass, 
  double f[], double *pot, double *kin )
{
  double d;
  double d2;
  int i;
  int j;
  int k;
  double ke;
  double pe;
  double PI2 = 3.141592653589793 / 2.0;
  double rij[3];

  pe = 0.0;
  ke = 0.0;

  for ( k = 0; k < np; k++ )
  {
/*
  Compute the potential energy and forces.
*/
    for ( i = 0; i < nd; i++ )
    {
      f[i+k*nd] = 0.0;
    }

    for ( j = 0; j < np; j++ )
    {
      if ( k != j )
      {
        d = dist ( nd, pos+k*nd, pos+j*nd, rij );
/*
  Attribute half of the potential energy to particle J.
*/
        if ( d < PI2 )
        {
          d2 = d;
        }
        else
        {
          d2 = PI2;
        }

        pe = pe + 0.5 * pow ( sin ( d2 ), 2 );

        for ( i = 0; i < nd; i++ )
        {
          f[i+k*nd] = f[i+k*nd] - rij[i] * sin ( 2.0 * d2 ) / d;
        }
      }
    }/*End of potential energy computation*/
/*
  Compute the kinetic energy.
*/
    for ( i = 0; i < nd; i++ )
    {
      ke = ke + vel[i+k*nd] * vel[i+k*nd];
    }
  }

  ke = ke * 0.5 * mass;
  
  *pot = pe;
  *kin = ke;

  return;
}

void compute_parallel ( int np, int nd, double pos[], double vel[], double mass, 
  double f[], double *pot, double *kin )
{
  double d;
  double d2;
  int i;
  int j;
  int k;
  double ke;
  double pe;
  double PI2 = 3.141592653589793 / 2.0;
  double rij[3];

  pe = 0.0;
  ke = 0.0;

  #pragma omp parallel for reduction(+:pe,ke) shared(pos,vel,f) private(d,d2,i,j,k, rij) firstprivate(PI2,np,nd)
  for ( k = 0; k < np; k++ )
  {
/*
  Compute the potential energy and forces.
*/
    for ( i = 0; i < nd; i++ )
    {
      f[i+k*nd] = 0.0;
    }

    for ( j = 0; j < np; j++ )
    {
      if ( k != j )
      {
        d = dist ( nd, pos+k*nd, pos+j*nd, rij );
/*
  Attribute half of the potential energy to particle J.
*/
        if ( d < PI2 )
        {
          d2 = d;
        }
        else
        {
          d2 = PI2;
        }

        pe = pe + 0.5 * pow ( sin ( d2 ), 2 );

        for ( i = 0; i < nd; i++ )
        {
          if(j==0) 
          {
            /*Reset force vector*/
            //f[i+k*nd]=0;
            /*Compute kinetic energy*/
            //ke = ke + vel[i+k*nd] * vel[i+k*nd];
          }
          f[i+k*nd] = f[i+k*nd] - rij[i] * sin ( 2.0 * d2 ) / d;
        }
      }
    }/*End of potential energy computation*/
/*
  Compute the kinetic energy.
*/
    for ( i = 0; i < nd; i++ )
    {
      ke = ke + vel[i+k*nd] * vel[i+k*nd];
    }
  }/*End of for loop*/

  ke = ke * 0.5 * mass;
  
  *pot = pe;
  *kin = ke;

  return;
}

double cpu_time ( )
{
  double value;

  value = ( double ) omp_get_wtime();

  return value;
}
/******************************************************************************/

double dist ( int nd, double r1[], double r2[], double dr[] )
{
  double d;
  int i;

  d = 0.0;
  for ( i = 0; i < nd; i++ )
  {
    dr[i] = r1[i] - r2[i];
    d = d + dr[i] * dr[i];
  }
  d = sqrt ( d );

  return d;
}
/******************************************************************************/

void initialize ( int np, int nd, double pos[], double vel[], double acc[] )
{
  int i;
  int j;
  int seed;
/*
  Set positions.
*/
  seed = 123456789;
  r8mat_uniform_ab ( nd, np, 0.0, 10.0, &seed, pos );
/*
  Set velocities.
*/
  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      vel[i+j*nd] = 0.0;
    }
  }
/*
  Set accelerations.
*/
  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      acc[i+j*nd] = 0.0;
    }
  }

  return;
}

void r8mat_uniform_ab ( int m, int n, double a, double b, int *seed, double r[] )
{
  int i;
  const int i4_huge = 2147483647;
  int j;
  int k;

  if ( *seed == 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R8MAT_UNIFORM_AB - Fatal error!\n" );
    fprintf ( stderr, "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      k = *seed / 127773;

      *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

      if ( *seed < 0 )
      {
        *seed = *seed + i4_huge;
      }
      r[i+j*m] = a + ( b - a ) * ( double ) ( *seed ) * 4.656612875E-10;
    }
  }

  return;
}

void timestamp ( )
{
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

void update ( int np, int nd, double pos[], double vel[], double f[], 
  double acc[], double mass, double dt )
{
  int i;
  int j;
  double rmass;

  rmass = 1.0 / mass;


  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      pos[i+j*nd] = pos[i+j*nd] + vel[i+j*nd] * dt + 0.5 * acc[i+j*nd] * dt * dt;
      vel[i+j*nd] = vel[i+j*nd] + 0.5 * dt * ( f[i+j*nd] * rmass + acc[i+j*nd] );
      acc[i+j*nd] = f[i+j*nd] * rmass;
    }
  }

  return;
}

void update_parallel ( int np, int nd, double pos[], double vel[], double f[], 
  double acc[], double mass, double dt )
{
  int i;
  int j;
  double rmass;

  rmass = 1.0 / mass;

  #pragma omp parallel for collapse(2) private(i,j) shared(pos, vel, acc, f, rmass, dt)
  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      pos[i+j*nd] = pos[i+j*nd] + vel[i+j*nd] * dt + 0.5 * acc[i+j*nd] * dt * dt;
      vel[i+j*nd] = vel[i+j*nd] + 0.5 * dt * ( f[i+j*nd] * rmass + acc[i+j*nd] );
      acc[i+j*nd] = f[i+j*nd] * rmass;
    }
  }

  return;
}
