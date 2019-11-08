#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define BLOCKSIZE 1024

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;
  //value = (double)omp_get_wtime();

  return value;
}

int prime_number(int n)
{
  int i;
  int j;
  int prime;
  int total;

  total = 0;

  for (i = 2; i <= n; i++)
  {
    prime = 1;
    for (j = 2; j < i; j++)
    {
      if ((i % j) == 0)
      {
        prime = 0;
        break;
      }
    }
    total = total + prime;
  }
  return total;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

int test(int n_lo, int n_hi, int n_factor);
int test_cuda(int n_lo, int n_hi, int n_factor);

int main(int argc, char *argv[])
{
  int n_factor;
  int n_hi;
  int n_lo;

  timestamp();
  printf("\n");
  printf("PRIME TEST\n");

  if (argc != 4)
  {
    n_lo = 1;
    n_hi = 131072;
    n_factor = 2;
  }
  else
  {
    n_lo = atoi(argv[1]);
    n_hi = atoi(argv[2]);
    n_factor = atoi(argv[3]);
  }

  double s_time = cpu_time();
  int br_s = test(n_lo, n_hi, n_factor);
  s_time = cpu_time() - s_time;
  double p_time = cpu_time();
  int br_p = test_cuda(n_lo, n_hi, n_factor);
  p_time = cpu_time() - p_time;

  printf("\n");
  printf("PRIME_TEST\n");
  printf("  Normal end of execution.\n");
  printf("Ukupno wall clock vreme sekvencijalne implementacije: %f\n", s_time);
  printf("Ukupno wall clock vreme paralelne     implementacije: %f\n", p_time);

  if (br_s == br_p)
    printf("Test PASSED");
  else
    printf("Test FAILED");
  printf("\n");
  timestamp();

  return 0;
}

int test(int n_lo, int n_hi, int n_factor)
{
  int i;
  int n;
  int primes;
  double ctime;

  printf("\n");
  printf("  Call PRIME_NUMBER to count the primes from 1 to N (sequential).\n");
  printf("\n");
  printf("         N        Pi          Time\n");
  printf("\n");

  n = n_lo;

  while (n <= n_hi)
  {
    ctime = cpu_time();

    primes = prime_number(n);

    ctime = cpu_time() - ctime;

    printf("  %8d  %8d  %14f\n", n, primes, ctime);
    n = n * n_factor;
  }

  return primes;
}

__global__ void prime_number_cuda(int p, int k, int *devResults)
{
  int i;
  int j;
  int prime;
  int globalId;
  int myNumberToCheck;

  extern __shared__ int sharedArray[];

  globalId = blockIdx.x * blockDim.x + threadIdx.x;
  myNumberToCheck = 2 + globalId;

  if (myNumberToCheck <= p)
  {
    prime = 1;
    for (j = 2; j < myNumberToCheck; j++)
    {
      if ((myNumberToCheck % j) == 0)
      {
        prime = 0;
        break;
      }
    }

    sharedArray[threadIdx.x] = prime;
  }
  else
  {
    sharedArray[threadIdx.x] = 0;
  }
  __syncthreads();
  //redukcija
  unsigned int tid = threadIdx.x;
  //unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int s;

  for (s = blockDim.x / 2; s > 32; s >>= 1)
  {
    if (tid < s)
    {
      sharedArray[tid] += sharedArray[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32)
    sharedArray[tid] += sharedArray[tid + 32];
  if (tid < 16)
    sharedArray[tid] += sharedArray[tid + 16];
  if (tid < 8)
    sharedArray[tid] += sharedArray[tid + 8];
  if (tid < 4)
    sharedArray[tid] += sharedArray[tid + 4];
  if (tid < 2)
    sharedArray[tid] += sharedArray[tid + 2];
  if (tid < 1)
  {
    sharedArray[tid] += sharedArray[tid + 1];
    devResults[blockIdx.x] = sharedArray[tid];
  }
}

int test_cuda(int n_lo, int n_hi, int n_factor)
{
  int i;
  int n;
  int primes;
  int sharedMemSize;

  int *devResults;
  int *hostResults;

  cudaEvent_t start, stop;
  cudaDeviceSynchronize(); //neki poziv jezgru cisto radi inicijalizacije

  printf("\n");
  printf("  Call PRIME_NUMBER to count the primes from 1 to N (parallel).\n");
  printf("\n");
  printf("         N        Pi          Time\n");
  printf("\n");

  n = n_lo;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  while (n <= n_hi)
  {

    sharedMemSize = sizeof(int) * BLOCKSIZE;
    int blocksNum = ceil(((double)n) / BLOCKSIZE);
    dim3 blockDim(BLOCKSIZE);
    dim3 gridDim(blocksNum);

    hostResults = (int *)malloc(blocksNum * sizeof(int));
    cudaMalloc(&devResults, blocksNum * sizeof(int));

    cudaEventRecord(start);
    prime_number_cuda<<<gridDim, blockDim, sharedMemSize>>>(n, 0, devResults);
    cudaEventRecord(stop);

    cudaMemcpy(hostResults, devResults, blocksNum * sizeof(int), cudaMemcpyDeviceToHost);

    primes = 0;
    for (i = 0; i < blocksNum; i++)
    {
      primes += hostResults[i];
    }
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    free(hostResults);
    cudaFree(devResults);

    printf("  %8d  %8d  %14f\n", n, primes, milliseconds / 1000);

    n = n * n_factor;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return primes;
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
