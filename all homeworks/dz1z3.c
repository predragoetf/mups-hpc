#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y

void init(void *u, int w, int h)
{
	int(*univ)[w] = u;
	for_xy
	{
		univ[y][x] = rand() < RAND_MAX / 10 ? 1 : 0;
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

void game_parallel(unsigned *u, int w, int h, int iter)
{
	for (int i = 0; i < iter; i++)
	{
#ifdef LIFE_VISUAL
		show(u, w, h);
#endif
		evolve_parallel(u, w, h);
#ifdef LIFE_VISUAL
		usleep(200000);
#endif
	}
}

int main(int c, char *v[])
{
	int w = 0, h = 0, iter = 0;
	int x,y;
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
	for (y = 0; y<h; y++)
		for (x=0; x<w; x++)
			u_parallel[y*w+x]=u[y*w+x];

	s_time = omp_get_wtime();
	game(u, w, h, iter);
	s_time = omp_get_wtime() - s_time;

	p_time = omp_get_wtime();
	game_parallel(u_parallel, w, h, iter);
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
