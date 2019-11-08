#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y

#define OUTER_WIDTH 16
#define INNER_WIDTH (OUTER_WIDTH - 2)

void init(unsigned *u, int w, int h)
{
	//int(*univ)[w] = u;
	for_xy
	{
		u[y * w + x] = rand() < RAND_MAX / 10 ? 1 : 0;
	}
}

void checkCUDAError(char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void checkCUDAErrorIter(int i)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: Puklo posle iteracije %d: %s.\n", i, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

double cpu_time(void)
{
	double value;

	value = (double)clock() / (double)CLOCKS_PER_SEC;

	return value;
}

void show(unsigned *u, int w, int h)
{
	//int(*univ)[w] = u;
	printf("\033[H");
	for_y
	{
		for_x printf(u[y * w + x] ? "\033[07m  \033[m" : "  ");
		printf("\033[E");
	}
	fflush(stdout);
}

void evolve(unsigned *u, int w, int h)
{
	//unsigned(*univ)[w] = u;
	unsigned neu[h][w];

	for_y for_x
	{
		int n = 0;
		//int pom = 10000000;
		
		for (int y1 = y - 1; y1 <= y + 1; y1++)
			for (int x1 = x - 1; x1 <= x + 1; x1++)
				if (u[((y1 + h) % h) * w + ((x1 + w) % w)])
					n++;

		if (u[y * w + x])
			n--;
		
		/*
		for (int y1 = y - 1; y1 <= y + 1; y1++)
			for (int x1 = x - 1; x1 <= x + 1; x1++)
			{	
				if (x1==x && y1==y) continue;
				if (u[((y1 + h) % h) * w + ((x1 + w) % w)])
				{
					n+=pom;
					
				}
				pom/=10;
			}
			*/

		neu[y][x] = (n == 3 || (n == 2 && u[y * w + x]));
		//neu[y][x] = n;//DEBUG
		

	}
	for_y for_x u[y * w + x] = neu[y][x];
}

__global__ void evolve_parallel(unsigned *in, unsigned *out, int w, int h)
{

	extern __shared__ unsigned univ[];

	//blockIdx.x=2; blockIdx.y=15;
	//threadIdx.x=2; threadIdx.y=2; 
	int globalX = threadIdx.x + blockIdx.x * INNER_WIDTH - 1; //2+2*1-1=3
	int globalY = threadIdx.y + blockIdx.y * INNER_WIDTH - 1; //2+15*1-1=16
	int globalId;
	int x, y, localId;

	int myWidth =  ( gridDim.x-1 == blockIdx.x ) *  ( w  - INNER_WIDTH * (gridDim.x - 1) ) + ( gridDim.x-1 != blockIdx.x ) * INNER_WIDTH;//1
	int myHeight = ( gridDim.y-1 == blockIdx.y ) * ( h - INNER_WIDTH * (gridDim.y - 1) ) + ( gridDim.y-1 != blockIdx.y ) * INNER_WIDTH;//1

	int unutar_matrice = (globalX <= w) && (globalY <= h);//true
	int radni_deo = 0;
	x = threadIdx.x;//2
	y = threadIdx.y;//2

	if (unutar_matrice != 0) //da li je ovo polje mapirano na matricu?
	{
		globalX = (globalX + w) % w;//3
		globalY = (globalY + h) % h;//0

		globalId = w * globalY + globalX; //16*0+3=3
		localId = OUTER_WIDTH * y + x;	//3*2+2=8

		univ[localId] = in[globalId];//shared[8] = u[0][3]
		//DEBUG
		/*if ( (blockIdx.x == 2) && blockIdx.y == 15)
			devDebug[localId] = univ[localId];*/
		
	}
	__syncthreads();
	
	
	unsigned myNewState;

	unsigned n = 0;
	radni_deo = (x >= 1) && (x <= myWidth) && (y >= 1) && (y <= myHeight);

	if ((unutar_matrice != 0) && (radni_deo != 0))
	{

		
		n = univ[x - 1 + (y - 1) * OUTER_WIDTH]
		 + univ[x + (y - 1) * OUTER_WIDTH]
		 + univ[x + 1 + (y - 1) * OUTER_WIDTH]
		 + univ[x - 1 + y * OUTER_WIDTH]
		 + univ[x + 1 + y * OUTER_WIDTH]
		 + univ[x - 1 + (y + 1) * OUTER_WIDTH]
		 + univ[x + (y + 1) * OUTER_WIDTH]
		 + univ[x + 1 + (y + 1) * OUTER_WIDTH];		

		 
		out[globalId] = (n == 3 || (n == 2 && univ[localId]));
		//u[globalId] = ( univ[x - 1 + (y + 1) * OUTER_WIDTH] == 1 ); //DEBUG
		

		//u[globalId] = n;
		//u[globalId] = univ[localId];
		
		
	}
	//__syncthreads(); // TO DO da li nam ovo treba?
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

void game_parallel(unsigned *u, int w, int h, int iter)
{

	//priprema svih mogucih stvari za cuda poziv

	int blockCountWidth = ceil(((double)w) / INNER_WIDTH);
	int blockCountHeight = ceil(((double)h) / INNER_WIDTH);
	cudaEvent_t start, stop;

	cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("Sracunato blockCountWidth=%d blockCountHeight=%d \n", blockCountWidth, blockCountHeight);

	dim3 blockDim(OUTER_WIDTH, OUTER_WIDTH);
	dim3 gridDim(blockCountWidth, blockCountHeight); //proveri kojim redom idu dimenzije, mozda je naopacke!

	unsigned *devU;
	unsigned *devUnew;
	//unsigned *debug = (unsigned *)malloc(sizeof(unsigned) * OUTER_WIDTH * OUTER_WIDTH);

	//unsigned *devDebug;

	//cudaMalloc(&devDebug, OUTER_WIDTH * OUTER_WIDTH * sizeof(unsigned));
	cudaMalloc(&devU, w * h * sizeof(unsigned));
	cudaMalloc(&devUnew, w * h * sizeof(unsigned));

	//checkCUDAError("Ne radi alokacija niza");

	//kopiranje pocetne matrice
	cudaMemcpy(devU, u, w * h * sizeof(unsigned), cudaMemcpyHostToDevice);

	//checkCUDAError("Ne radi kopiranje niza");

	unsigned * bufferInput = devU;
	unsigned * bufferOutput = devUnew;

	cudaEventRecord(start);	
	for (int i = 0; i < iter; i++)
	{
		//cuda poziv jezgra
		
		//evolve_parallel<<<gridDim, blockDim, OUTER_WIDTH * OUTER_WIDTH * sizeof(unsigned)>>>(bufferInput, bufferOutput, w, h, devDebug);
		evolve_parallel<<<gridDim, blockDim, OUTER_WIDTH * OUTER_WIDTH * sizeof(unsigned)>>>(bufferInput, bufferOutput, w, h);
		unsigned * pom = bufferInput;
		bufferInput = bufferOutput;
		bufferOutput = pom;

		//checkCUDAErrorIter(i);
		//cudaDeviceSynchronize();
		//checkCUDAError("Puko na sinhronizaciji");
	}
	cudaEventRecord(stop);
	//checkCUDAError("Ne radi record-ovanje stop-a");
	cudaMemcpy(u, bufferInput, w * h * sizeof(unsigned), cudaMemcpyDeviceToHost);

	//cudaMemcpy(debug, devDebug, OUTER_WIDTH * OUTER_WIDTH * sizeof(unsigned), cudaMemcpyDeviceToHost);

	//checkCUDAError("Ne radi kopiranje niza posle petlje");

	cudaEventSynchronize(stop);
	//checkCUDAError("Ne radi sinhronizovanje stop eventa");
	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	//checkCUDAError("Ne radi merenje vremena");
	printf("Vreme CUDA implementacije mereno CUDA timer-ima: %f (ver 2)\n", ms / 1000);

	cudaFree(devU);
	//checkCUDAError("Ne radi oslobadjanje niza devU");

	/*
	FILE *f4;
	f4 = fopen("matricaDebug.txt", "w");
	int x, y;
	for (y = 0; y < OUTER_WIDTH; y++)
	{
		for (x = 0; x < OUTER_WIDTH; x++)
		{
			fprintf(f4, "[%u]", debug[y * OUTER_WIDTH + x]);
		}
		fprintf(f4, "\n");
	}
	fclose(f4);*/
}

int main(int c, char *v[])
{
	int w = 0, h = 0, iter = 0;
	int x, y;
	double s_time, p_time;
	unsigned *u;
	unsigned *u_parallel;

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

	init(u, w, h);
	for (y = 0; y < h; y++)
		for (x = 0; x < w; x++)
			u_parallel[y * w + x] = u[y * w + x];

	FILE *f1;
	f1 = fopen("matricaPocetno.txt", "w");
	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			if (u[y * w + x] == 1)
				fprintf(f1, "#");
			else
				fprintf(f1, "_");
		}
		fprintf(f1, "\n");
	}
	fclose(f1);

	s_time = cpu_time();
	game(u, w, h, iter);
	s_time = cpu_time() - s_time;

	/*
	FILE *f2;
	f2 = fopen("matricaSekvencijalno.txt", "w");
	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			if (u[y * w + x] == 1)
				fprintf(f2, "#");
			else
				fprintf(f2, "_");
		}
		fprintf(f2, "\n");
	}
	fclose(f2);
	*/

	FILE *f2;
	f2 = fopen("matricaSekvencijalno.txt", "w");
	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			
			fprintf(f2, "[%u]", u[ y*w + x ]);
		}
		fprintf(f2, "\n");
	}
	fclose(f2);


	p_time = cpu_time();
	game_parallel(u_parallel, w, h, iter);
	p_time = cpu_time() - p_time;

	/*
	FILE * f3;
	f3 = fopen("matricaCUDA.txt", "w");
	for (y = 0; y < h; y++)
	{
		for(x = 0; x < w; x++)
		{
			if (u_parallel[y*w+x] == 1) fprintf(f3, "#");
			else fprintf(f3, "_");

		}
		fprintf(f3, "\n");
	}
	fclose(f3);
	*/

	FILE *f3;
	f3 = fopen("matricaCUDA.txt", "w");
	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			/*
			if (u_parallel[y * w + x] != 0)
				fprintf(f3, "[%u]", u_parallel[y * w + x]);
			else
				fprintf(f3, "_");
			*/
			fprintf(f3, "[%u]", u_parallel[y * w + x]);
		}
		fprintf(f3, "\n");
	}
	fclose(f3);

	/*Poredjenje implementacija*/
	/*Poredjenje krajnjih matrica*/
	int equal = 1;
	for (y = 0; y < w * h; y++)
	{
		if (u[y] != u_parallel[y])
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
