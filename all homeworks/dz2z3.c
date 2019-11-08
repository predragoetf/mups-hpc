#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y

double cpu_time ( void ) {
  double value;

  //value = ( double ) clock ( ) 
  //      / ( double ) CLOCKS_PER_SEC;
  value = (double) omp_get_wtime();

  return value;
}

void init(void *u, int w, int h)
{
	int(*univ)[w] = u;
	for_xy
	{
		univ[y][x] = rand() < RAND_MAX / 10 ? 1 : 0;
	}
}

void init2(void *u, int w, int h)
{
	int(*univ)[w] = u;
	for_xy
	{
		univ[y][x] = x;
	}
}


void show(void *u, int w, int h)
{
	int(*univ)[w] = u;
	printf("\033[H");
	for_y
	{
		for_x printf(univ[y][x] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
}

void evolve(void *u, int w, int h)
{
	unsigned(*univ)[w] = u;
	unsigned new[h][w];

	for_y for_x
	{
		int n = 0;
		for (int y1 = y - 1; y1 <= y + 1; y1++)
			for (int x1 = x - 1; x1 <= x + 1; x1++)
				if (univ[(y1 + h) % h][(x1 + w) % w])
					n++;

		if (univ[y][x])
			n--;
		new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
	}
	for_y for_x univ[y][x] = new[y][x];
}

void evolve_parallel(void *u, int w, int h)
{
	unsigned(*univ)[w] = u;
	unsigned new[h][w];
#pragma omp parallel for collapse(2) shared(univ, new, w, h)
	for_y for_x
	{
		int n = 0;
		for (int y1 = y - 1; y1 <= y + 1; y1++)
			for (int x1 = x - 1; x1 <= x + 1; x1++)
				if (univ[(y1 + h) % h][(x1 + w) % w])
					n++;

		if (univ[y][x])
			n--;
		new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
	}

	for_y for_x univ[y][x] = new[y][x];
}

void game(unsigned *u, int w, int h, int iter)
{
	for (int i = 0; i < iter; i++)
	{
#ifdef LIFE_VISUAL
		show(u, w, h);
#endif
		evolve(u, w, h);
#ifdef LIFE_VISUAL
		usleep(200000);
#endif
	}
}

void game_parallel(unsigned *u, int w, int h, int iter, MPI_Datatype kolona)
{
	int rank, size;
	
	int i,j;
	int x, y;
	int rank_left, rank_right;
	MPI_Request request1;
	MPI_Request request2;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Datatype kolona2;
	MPI_Type_vector(h, 1, w, MPI_UNSIGNED, &kolona2);
	MPI_Type_commit(&kolona2);
	
	rank_left = (rank-1)*(rank!=0)+(size-1)*(rank==0);
	rank_right = (rank+1)*(rank!=size-1)+0*(rank==size-1);

	//printf("Ja sam process:%d, line:%d, moji w i h su %d i %d i moji susedi su levo %d i desno %d \n", rank, __LINE__, w, h, rank_left, rank_right); //fflush(stdout);
	unsigned levaivica[h];
	unsigned desnaivica[h];

	unsigned(*univ)[w] = u;
	unsigned new[h][w];
	

	for (int i = 0; i < iter; i++)
	{
		

		//asinhrono slanje leve i desne ivice 
		//MPI_Send(&univ[0][0], 1, kolona2, rank_left, i, MPI_COMM_WORLD); 
		//MPI_Send(&univ[0][w-1], 1, kolona2, rank_right, i, MPI_COMM_WORLD);

		
		MPI_Isend(&univ[0][0], 1, kolona2, rank_left, i, MPI_COMM_WORLD, &request1); 
		MPI_Isend(&univ[0][w-1], 1, kolona2, rank_right, i, MPI_COMM_WORLD, &request2);
		
		/*-----------------------racunanje sredine--------------*/
		for (y = 0; y<h;y++)/*po visini*/
			for (x=1; x<=w-2; x++)/*po sirini*/
			{
				int n = 0;

				for (int y1 = y - 1; y1 <= y + 1; y1++)
					for (int x1 = x - 1; x1 <= x + 1; x1++)
						if (univ[(y1 + h) % h][(x1 + w) % w])
							n++;

				if (univ[y][x])	n--;
				new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
			}

		/*-----------------------sracunata sredina!--------------*/


		/*-----------dobijanje levog i desnog suseda-------------*/
		MPI_Recv(desnaivica, h, MPI_UNSIGNED, rank_right, i, MPI_COMM_WORLD, &status);		
		MPI_Recv(levaivica, h, MPI_UNSIGNED, rank_left, i, MPI_COMM_WORLD, &status);
		/*-----------dobijena leva i desna susedska ivica--------*/

		/*--------------izracunavanje ivica--------------*/

		//racunanje leve ivice	
		for(y=0;y<h;y++)
		{
			int n = 0;

			for (int y1 = y - 1; y1 <= y + 1; y1++)
			{
					for (int x1 = 0; x1 <= 1; x1++)
						if (univ[(y1 + h) % h][(x1 + w) % w])
							n++;
					if(levaivica[(y1 + h) % h]) n++;
			}
			
			if (univ[y][0])
					n--;
			new[y][0] = (n == 3 || (n == 2 && univ[y][0]));

		}

		//racunanje desne ivice
		for(y=0;y<h;y++)
		{
			int n = 0;

			for (int y1 = y - 1; y1 <= y + 1; y1++)
			{
					for (int x1 = w-2; x1 <= w-1; x1++)
						if (univ[(y1 + h) % h][(x1 + w) % w])
							n++;
					if(desnaivica[(y1 + h) % h]) n++;
			}
			
			if (univ[y][w-1])
					n--;
			new[y][w-1] = (n == 3 || (n == 2 && univ[y][w-1]));

		}

		/*----------------kraj racunanja ivica-----------*/
		for_y for_x univ[y][x] = new[y][x];
		
	}
}

int main(int c, char *v[])
{
	int w = 0, h = 0, iter = 0;
	int x,y;
	double s_time, p_time;
	unsigned *u;
	unsigned *u_parallel;
	int dataIn[3];
	int i, j;
	
	int rank, size;
	MPI_Status status;

	int chunksize, pom, osakaceni;

	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank==0)
	{
	if (c > 1)
		w = atoi(v[1]);
	if (c > 2)
		h = atoi(v[2]);
	if (c > 3)
		iter = atoi(v[3]);
	if (w <= 0)
		w = 30;
	if (h <= 0)
		h = 30;
	if (iter <= 0)
		iter = 1000;

	
	
	u = (unsigned *)malloc(w * h * sizeof(unsigned));
	if (!u)
		exit(1);
	u_parallel = (unsigned *)malloc(w * h * sizeof(unsigned));
	if (!u_parallel)
		exit(1);
	//init(u, w, h);
	init(u,w,h);
	

	
	for (y = 0; y<h; y++)
		for (x=0; x<w; x++)
			u_parallel[y*w+x]=u[y*w+x];
	

	s_time = omp_get_wtime();
	game(u, w, h, iter);
	s_time = omp_get_wtime() - s_time;
	

	/*pocetak paralelnog koda*/
	p_time = omp_get_wtime();

	/*spread-ovanje pocetnih podataka*/
	dataIn[0] = w;
	dataIn[1] = h;
	dataIn[2] = iter;
	}
	
	MPI_Bcast(dataIn, 3, MPI_INT, 0, MPI_COMM_WORLD); 
	
	w = dataIn[0];
	h = dataIn[1];
	iter = dataIn[2];
	/*sad svi imaju w,h,iter*/

	MPI_Datatype kolona;
	MPI_Type_vector(h, 1, w, MPI_UNSIGNED, &kolona);
	MPI_Type_commit(&kolona);

	/*--------------raspodela kolona-----------------------*/
	
	if (rank==0)
	{
		chunksize=(w+size-1)/size;
		osakaceni = w-(size-1)*chunksize;
		
		for (i = 1; i<size-1; i++)
		{
			//slanje broja kolona koje su dodeljene procesu sa rankom i
			MPI_Send(&chunksize, 1, MPI_INT, i, i, MPI_COMM_WORLD); 
			
		}		
		MPI_Send(&osakaceni, 1, MPI_INT, size-1, size-1, MPI_COMM_WORLD);
		
	}
	else {
		
		MPI_Recv(&chunksize, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);		
	}
	//svako ima lokalni bafer
	unsigned * private_data = malloc(sizeof(unsigned)*h*chunksize);
	unsigned * linearizovano =  malloc(sizeof(unsigned)*h*chunksize);
	unsigned column_buffer[h];
	
	/*svakom damo njegovih chunksize kolona*/
	if (rank==0)
	{
		for (i = 0; i<size-1; i++)
		{
			for (j = 0; j<chunksize; j++)
			{
				MPI_Send(&u_parallel[i*chunksize+j], 1, kolona, i, i, MPI_COMM_WORLD); 	
			}
			//slanje kolona koje su dodeljene procesu sa rankom i				
		}
		//slanje kolona dodeljenih poslednjem procesu
		for (j = 0; j<osakaceni; j++)
			{
				MPI_Send(&u_parallel[(size-1)*chunksize+j], 1, kolona, size-1, size-1, MPI_COMM_WORLD); 	
			}
	}
	
	/*sad svi primaju kolone, cak i rank0 koji je samom sebi poslao kolone*/
	for (i = 0; i<chunksize; i++)
	{
		MPI_Recv(linearizovano+i*h, h, MPI_UNSIGNED, 0, rank, MPI_COMM_WORLD, &status);
		//MPI_Recv(linearizovano+i*h, 1, kolona, 0, rank, MPI_COMM_WORLD, &status);
	}
	
	/*transponovanje*/
	for(int colcnt = 0; colcnt<chunksize; colcnt++)
	{
		for(int rowcnt = 0; rowcnt<h; rowcnt++)
		{
			private_data[rowcnt*chunksize+colcnt] = linearizovano[colcnt*h+rowcnt];
			//printf("%d, ", private_data[rowcnt*chunksize+colcnt]); fflush(stdout);
		}
		//printf("\n"); fflush(stdout);
	}
	
	/*pokretanje paralelnog izracunavanja*/
	


	game_parallel(private_data, chunksize, h, iter, kolona);

	//sad se u private_data svakog od njih nalaze sracunate njegove kolone!

	/*------------------skupljanje privatnih podataka u matricu u_parallel---------------*/

	

	MPI_Datatype kolona3;
	MPI_Type_vector(h, 1, chunksize, MPI_UNSIGNED, &kolona3);
	MPI_Type_commit(&kolona3);

	//svi salju svoje kolone ranku0
	for (i = 0; i<chunksize; i++)
	{
		MPI_Send(private_data+i, 1, kolona3, 0, i, MPI_COMM_WORLD);
	}


	//rank0 prima kolone
	if (rank==0)
	{
		for (i = 0; i<size-1; i++)
		{
			//primanje kolona koje su dodeljene procesu sa rankom i u linearizovano, pa transponovanje
			for(j=0; j<chunksize; j++)
			{
				//MPI_Recv(column_buffer, 1, kolona, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
				MPI_Recv(column_buffer, h, MPI_UNSIGNED, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
				//int sender = status.MPI_SOURCE;
        		int tag = status.MPI_TAG;
				for (int k = 0; k<h; k++)
				{
					//linearizovano[i*chunksize*h+h*tag+k]=column_buffer[k];
					u_parallel[k*w + i*chunksize+tag] = column_buffer[k];
				}
			}
			/*
			for(int colcnt = 0; colcnt<chunksize; colcnt++)
				for(int rowcnt = 0; rowcnt<h; rowcnt++)
				{
					u_parallel[rowcnt][i*chunksize+colcnt] = linearizovano[colcnt*h+rowcnt];

				}*/
		}
		//primanje kolona dodeljenih poslednjem procesu
		for(j = 0; j<osakaceni; j++)
		{
			MPI_Recv(column_buffer, h, MPI_UNSIGNED, size-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
			//MPI_Recv(column_buffer, 1, kolona, size-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
			//int sender = status.MPI_SOURCE;
        	int tag = status.MPI_TAG;
			for (int k = 0; k<h; k++)
			{
				//linearizovano[i*chunksize*h+h*tag+k]=column_buffer[k];
				u_parallel[k*w + (size-1)*chunksize+tag] = column_buffer[k];
			}
		}	
	}
	/*-----------------skupljeni podaci izracunavanja u u_parallel------------------------*/

	//printf("process:%d, line:%d\n", rank, __LINE__); fflush(stdout);
	MPI_Finalize();
	//printf("process:%d, line:%d\n", rank, __LINE__); fflush(stdout);
	if(rank==0)
	{
		p_time = omp_get_wtime() - p_time;

		/*Poredjenje implementacija*/
		/*Poredjenje krajnjih matrica*/
		int equal = 1;
		for (y = 0;y<w*h;y++)
		{
			if (u[y]!=u_parallel[y]) 
			{
				equal = 0;
				break;
			}
		}
		if (equal)
		{
			printf("Krajnje stanje univerzuma je isto posle izvrsavanja paralelnog i sekvencijalnog koda!\n");
			printf("Test PASSED\n");
		}
		else
		{
			printf("Krajnje stanje univerzuma nije isto posle izvrsavanja paralelnog i sekvencijalnog koda!\n");
			printf("Test FAILED\n");
		}


		free(u);
		free(u_parallel);


		printf("W: %d, H: %d, I: %d\n", w, h, iter);
		printf("Sekvencijlno vreme: %f\n", s_time);
		printf("Paralelno vreme: %f\n", p_time);
		printf("\n");
		fflush(stdout);
	}
}
