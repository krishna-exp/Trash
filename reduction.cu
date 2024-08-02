#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static inline void expAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("assert failed: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define expErrChk(err) { expAssert((err), __FILE__, __LINE__); }

template<size_t blocks, size_t threads>
__global__ void reduce_sum(const int *a, int *o, size_t size) {
  __shared__ int sums[threads];

  sums[threadIdx.x] = 0;

  // using threads as template to perform
  // compile time optimizations
  // assert(threads == blockDim.x);

  for (size_t i = 0; i < size; i += blocks * threads) {
    sums[threadIdx.x] += a[i + threadIdx.x + (blockIdx.x * threads)];
  }

  __syncthreads();

  // all the sums must be reduced to one value now
  for (int t = threads >> 1; t > 0; t = t >> 1) {
    if (threadIdx.x < t) {
      sums[threadIdx.x] = sums[threadIdx.x] + sums[threadIdx.x + t];
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    o[blockIdx.x] = sums[0];
  }
}

int main() {
  size_t size = 4 * 1024 * 1024;

  int *a = (int *) malloc(sizeof(int) * size);

  if (NULL == a) {
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < size; i++) {
    const int upper = 10;
    const int lower = 0;

    a[i] = (rand() % (upper - lower + 1)) + lower;
  }

  const int threads = 1024;
  const int blocks = 512;

  int *d_a = NULL;
  int *d_o = NULL;

  int o[blocks] = { 0 };

  expErrChk(cudaMalloc(&d_a, sizeof(int) * size));
  expErrChk(cudaMalloc(&d_o, sizeof(int) * blocks));
  expErrChk(cudaMemcpy(d_a, a, sizeof(int) * size, cudaMemcpyHostToDevice));

  reduce_sum<blocks, threads><<<blocks, threads>>>(d_a, d_o, size);
  expErrChk(cudaGetLastError());

  expErrChk(cudaMemcpy(o, d_o, sizeof(int) * blocks, cudaMemcpyDeviceToHost));

  int d_sum = 0;

  for (int i = 0; i < blocks; i++) {
    d_sum += o[i];
  }

  int sum = 0;

  for (int i = 0; i < size; i++) {
    sum += a[i];
  }

  printf("sum: %d, d_sum: %d\n\n", sum, d_sum);

  if (sum != d_sum) {
    printf("Error, didn't compute well\n");
  }

  if (a) {
    free(a);
  }

  return 0;
}
