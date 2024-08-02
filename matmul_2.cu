#include <time.h>

#include <mma.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <omp.h>

static inline void expAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("assert failed: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

static inline void expAssert(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("assert failed, cublas error: status(%d) %s %d\n", status, file, line);
    exit(status);
  }
}

#define expErrChk(err) { expAssert((err), __FILE__, __LINE__); }

static float randf() {
    const int upper = 10;
    const int lower = -10;

    return ((rand() % (upper - lower + 1)) + lower) / 10.0;
}

static void randff(float *x, size_t size) {
    for (size_t i = 0; i < size; i++) {
        // printf("Setting %zu\n", i);
        x[i] = randf();
    }
}

static void init_mat(float *mat, int m, int n) {
  randff(mat, m * n);
}

static void copy_mat(half *d, const float *s, size_t size) {
  for (size_t idx = 0; idx < size; idx++) {
    d[idx] = s[idx];
  }
}

__global__ static void mat_mul_d(float *a, float *b, float *c,
                                 size_t m, size_t n, size_t k) {

  size_t idxX = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t idxY = threadIdx.y + (blockIdx.y * blockDim.y);

  if (idxX >= n || idxY >= m) {
    return;
  }

  float sum = 0;

  for (size_t p = 0; p < k; p++) {
    sum += a[idxY * k + p] * b[p * n + idxX];
  }

  c[idxY * n + idxX] += sum;
}

static inline void mat_mul(float *da, float *db, float *dc,
                           size_t m, size_t n, size_t k) {

  int threadsY = 32;
  int threadsX = 32;
  int blocksY = (m + threadsY - 1) / threadsY;
  int blocksX = (n + threadsX - 1) / threadsX;

  dim3 threads(threadsX, threadsY);
  dim3 blocks(blocksX, blocksY);

  mat_mul_d<<<blocks, threads>>>(da, db, dc, m, n, k);
  expErrChk(cudaGetLastError());
}

template<int threads, int scalingFactor>
__global__ static void mat_mul_dt(const half *a, const half *b, half *c,
                                  size_t m, size_t n, size_t k) {

  // each thread can process 16 * 16 elements
  // So, the cache should be able to feed all of the
  // threads simultaneously.
  // Cache, the bigger, the better.

  // ideally scaling factor should be = threads
  // that way each of the thread can work on it's own
  // slice of the matrix
  // But, that would also make the matrices not fit in the
  // shared memory.
  constexpr int tDim = 16;
  constexpr int sDim = tDim * scalingFactor;

  // sDim can be increased in size by using
  // co-operative thread groups?
  // todo: L1 cache is being used extensively
  //  figure if co-operative threads groups can help

  assert(threads % sDim == 0);

  __shared__ half aS[sDim * sDim];
  __shared__ half bS[sDim * sDim];
  __shared__ half cS[sDim * sDim];

  // j is the horizontal dimension
  // x is also the horizontal dimension
  const int jT = threadIdx.x;
  const int jB = blockIdx.x;
  const int iB = blockIdx.y;

  // the max warp index
  const int wMax = threads / warpSize;
  const int jW = jT / warpSize;
  // no wraps in the vertical direction
  const int iW = 0;
  const int wIdx = (iW * wMax) + jW;

  assert((scalingFactor * scalingFactor) % wMax == 0);

  constexpr int oFactor = threads / sDim;

  // load c
  for (int i = 0; i < sDim; i += oFactor) {
    int idx = jT;

    int jIdx = idx % sDim;
    int iIdx = idx / sDim;

    cS[jIdx + ((i + iIdx) * sDim)] = c[jIdx + (jB * sDim) + ((i + iIdx + (iB * sDim)) * n)];
  }

  __syncthreads();

  for (size_t p0 = 0; p0 < k; p0 += sDim) {
    // load a
    for (int o = 0; o < sDim; o += oFactor) {
      // load the aS Array just like cS

      int idx = jT;

      int jIdx = idx % sDim;
      int iIdx = idx / sDim;

      aS[jIdx + ((o + iIdx) * sDim)] = a[jIdx + p0 + ((o + iIdx + (iB * sDim)) * k)];
    }

    // load b
    for (int o = 0; o < sDim; o += oFactor) {
      int idx = jT;

      int jIdx = idx % sDim;
      int iIdx = idx / sDim;

      bS[jIdx + ((o + iIdx) * sDim)] = b[jIdx + (jB * sDim) + (p0 + o + iIdx) * n];
    }

    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tDim, tDim, tDim, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tDim, tDim, tDim, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tDim, tDim, tDim, half> c_frag;

    // this is compute in the horizontal direction
    // and, this will cause memory bank conflicts
    // If instead compute in the diagonal direction
    // It will not have any memory bank conflicts
    //
    // Compute order of the cS matrix
    // if threads == 256
    // Current compute order:
    // |-------------|
    // | w0 w1 w2 w3 |
    // | w4 w5 w6 w7 |
    // | w0 w1 w2 w3 |
    // | w4 w5 w6 w7 |
    // |-------------|
    //
    // Ideal compute order (probably):
    // Diagonal compte for each warp
    // |-------------|
    // | w0 w4 w0 w4 |
    // | w5 w1 w5 w1 |
    // | w2 w6 w2 w6 |
    // | w7 w3 w7 w3 |
    // |-------------|
    //
    // if threads == 128
    // Current compute order:
    // |-------------|
    // | w0 w1 w2 w3 |
    // | w0 w1 w2 w3 |
    // | w0 w1 w2 w3 |
    // | w0 w1 w2 w3 |
    // |-------------|
    //
    // Ideal compute order (probably):
    // Diagonal compte for each warp
    // |-------------|
    // | w0 w0 w0 w0 |
    // | w1 w1 w1 w1 |
    // | w2 w2 w2 w2 |
    // | w3 w3 w3 w3 |
    // |-------------|
    //
    // NOTE:
    // Time Step 1: w0, w1, w2, w3 compute idx 0, idx 5, idx 10, idx 15 respectively
    // Time Step 2: w0, w1, w2, w3 compute idx 1, idx 6, idx 11, idx 12 respectively
    // Time Step 3: w0, w1, w2, w3 compute idx 2, idx 7,  idx 8, idx 13 respectively
    // Time Step 4: w0, w1, w2, w3 compute idx 3, idx 4,  idx 9, idx 14 respectively

    const int cMax = (scalingFactor * scalingFactor) / wMax;

    #pragma unroll
    for (int cIdx = 0; cIdx < cMax; cIdx++) {
      // j -> horizontal direction
      // i -> vertical direction

      // slow order with a lot of bank conflicts
      // const int j = (wIdx % scalingFactor);
      // const int i = (cIdx * scalingFactor / cMax) + (wIdx / scalingFactor);

      // better order
      const int j = (wIdx + cIdx) % scalingFactor;
      const int i = wIdx % scalingFactor;

      nvcuda::wmma::load_matrix_sync(c_frag, &cS[(i * tDim * sDim) + (j * tDim)], sDim, nvcuda::wmma::mem_row_major);

      #pragma unroll
      for (int p = 0; p < sDim; p += tDim) {
        // Load the inputs
        nvcuda::wmma::load_matrix_sync(a_frag, &aS[p + (i * tDim * sDim)], sDim);
        nvcuda::wmma::load_matrix_sync(b_frag, &bS[(j * tDim) + (p * sDim)], sDim);

        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      nvcuda::wmma::store_matrix_sync(&cS[(i * tDim * sDim) + (j * tDim)], c_frag, sDim, nvcuda::wmma::mem_row_major);
    }

    __syncthreads();
  }

  __syncthreads();

  // store c
  for (int i = 0; i < sDim; i += oFactor) {
    int idx = jT;

    int jIdx = idx % sDim;
    int iIdx = idx / sDim;

    c[jIdx + (jB * sDim) + ((i + iIdx + (iB * sDim)) * n)] = cS[jIdx + ((i + iIdx) * sDim)];
  }
}

static inline void mat_mulT(const half* da, const half* db, half* dc,
                            size_t m, size_t n, size_t k) {

  constexpr int threads = 128;
  constexpr int scalingFactor = 4;

  // the kernel parameter
  constexpr int tDim = 16;

  // this logic is also present in the mat_mul_dt
  constexpr int sDim = tDim * scalingFactor;

  int blocksY = (m + sDim - 1) / sDim;
  int blocksX = (n + sDim - 1) / sDim;

  // all threads will be in the X dimension
  // warp level scheduling is performed internal to kernel
  dim3 threadsDim(threads);
  dim3 blocksDim(blocksX, blocksY);

  // printf("blocksX: %d, blocksY: %d\n", blocksX, blocksY);

  mat_mul_dt<threads, scalingFactor><<<blocksDim, threadsDim>>>(da, db, dc, m, n, k);
  expErrChk(cudaGetLastError());
}

template<typename T>
static void print_mat(T *a, size_t m, size_t n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      printf("%.2f, ", (float) a[j + i * n]);
    }
    printf("\n");
  }
}

static float time_micro() {
  // DO NOT use chrono, it's just not precise enough

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

  float time = (ts.tv_sec * 1000000.0f) + (ts.tv_nsec / 1000.0f);

  return time;
}

int main() {
  constexpr int reps = 10;

  srand((unsigned int) time_micro());

  printf("rand: %f\n", randf());
  printf("time: %f\n", time_micro());

  size_t size = 8192;
  size_t m = size;
  size_t n = size;
  size_t k = size;

  // a -> m * k
  // b -> k * n
  // c -> m * n
  float *a = (float *) malloc(sizeof(float) * m * k);
  float *b = (float *) malloc(sizeof(float) * k * n);
  float *cS = (float *) malloc(sizeof(float) * m * n);
  float *c_ref = (float *) malloc(sizeof(float) * m * n);
  half *aH = (half *) malloc(sizeof(half) * m * k);
  half *bH = (half *) malloc(sizeof(half) * k * n);
  half *cH = (half *) malloc(sizeof(half) * m * n);
  half *cHcuBlas = (half *) malloc(sizeof(half) * m * n);

  if (NULL == a || NULL == b || NULL == cS) {
    printf("Error allocating\n");
    goto end;
  }

  if (NULL == aH || NULL == bH || NULL == cH || NULL == cHcuBlas) {
    printf("Error allocating\n");
    goto end;
  }

  init_mat(a, m, k);
  copy_mat(aH, a, m * k);
  init_mat(b, k, n);
  copy_mat(bH, b, k * n);

  float *da;
  float *db;
  float *dc;

  printf("CUDA incepted\n");

  expErrChk(cudaMalloc((void **) &da, sizeof(float) * m * k));
  expErrChk(cudaMalloc((void **) &db, sizeof(float) * k * n));
  expErrChk(cudaMalloc((void **) &dc, sizeof(float) * m * n));

  /*
  {
    // this takes long enough
    // skipping with one rep
    const int reps = 1;

    printf("CUDA starting\n");

    expErrChk(cudaMemcpy(da, a, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    expErrChk(cudaMemcpy(db, b, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    expErrChk(cudaMemset(dc, 0, sizeof(float) * m * n));

    const float begin = time_micro();

    for (int rep = 0; rep < reps; rep++) {
      mat_mul(da, db, dc, m, n, k);
    }

    const float end = time_micro();

    expErrChk(cudaMemcpy(cS, dc, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    printf("CUDA complete, end: %f, begin: %f, time/rep = %f\n", end, begin, ((end - begin) / reps));
  }
  */

  expErrChk(cudaFree(dc));
  expErrChk(cudaFree(db));
  expErrChk(cudaFree(da));

  printf("Tensor CUDA incepted\n");

  half *daH;
  half *dbH;
  half *dcH;

  expErrChk(cudaMalloc((void **) &daH, sizeof(half) * m * k));
  expErrChk(cudaMalloc((void **) &dbH, sizeof(half) * k * n));
  expErrChk(cudaMalloc((void **) &dcH, sizeof(half) * m * n));

  expErrChk(cudaMemcpy(daH, aH, sizeof(half) * m * k, cudaMemcpyHostToDevice));
  expErrChk(cudaMemcpy(dbH, bH, sizeof(half) * k * n, cudaMemcpyHostToDevice));

  {
    printf("Tensor CUDA Starting\n");
    expErrChk(cudaMemset(dcH, 0, sizeof(half) * m * n));

    const float begin = time_micro();

    for (int rep = 0; rep < reps; rep++) {
      printf("rep: %d\n", rep);
      mat_mulT(daH, dbH, dcH, m, n, k);
      expErrChk(cudaDeviceSynchronize());
    }

    const float end = time_micro();

    expErrChk(cudaMemcpy(cH, dcH, sizeof(half) * m * n, cudaMemcpyDeviceToHost));

    printf("Tensor CUDA complete, end: %f, begin: %f, time/rep = %f\n", end, begin, ((end - begin) / reps));
  }

  expErrChk(cudaFree(dcH));

  printf("cuBLAS incepted\n");

  half* dcHcuBlas;
  expErrChk(cudaMalloc((void **) &dcHcuBlas, sizeof(half) * m * n));

  {
    cublasHandle_t handle;
    expErrChk(cublasCreate(&handle));
    expErrChk(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const half alpha(1.0f);
    const half beta(1.0f);

    printf("cuBLAS Starting\n");
    expErrChk(cudaMemset(dcHcuBlas, 0, sizeof(half) * m * n));

    const float begin = time_micro();

    for (int rep = 0; rep < reps; rep++) {
      printf("rep: %d\n", rep);
      expErrChk(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, daH, k, dbH, n, &beta, dcHcuBlas, n));
      expErrChk(cudaDeviceSynchronize());
    }

    const float end = time_micro();

    expErrChk(cudaMemcpy(cHcuBlas, dcHcuBlas, sizeof(half) * m * n, cudaMemcpyDeviceToHost));
    expErrChk(cublasDestroy(handle));

    printf("cuBLAS complete, end: %f, begin: %f, time/rep = %f\n", end, begin, ((end - begin) / reps));
  }

  expErrChk(cudaFree(dcHcuBlas));
  expErrChk(cudaFree(dbH));
  expErrChk(cudaFree(daH));

  /*
  printf("cS:\n");
  print_mat(cS, m, n);
  printf("-------------------------------------------------\n");
  printf("cH:\n");
  print_mat(cH, m, n);
  printf("-------------------------------------------------\n");

  assert(reps == 1);

  const size_t m_block = 32;
  const size_t n_block = 32;
  const size_t k_block = 32;

  #pragma omp parallel for
  for (size_t io = 0; io < m; io += m_block) {
    for (size_t jo = 0; jo < n; jo += n_block) {
      for (size_t po = 0; po < k; po += k_block) {

        for (size_t ii = 0; ii < m_block; ii++) {
          for (size_t ji = 0; ji < n_block; ji++) {
            size_t i = io + ii;
            size_t j = jo + ji;

            float sum = c_ref[i * n + j];

            for (size_t pi = 0; pi < k_block; pi++) {
              size_t p = po + pi;

              sum += a[i * k + p] * b[p * k + j];
            }

            c_ref[i * n + j] = sum;
          }
        }
      }
    }
  }

  printf("Comparing CUDA\n");
  for (size_t i = 0; i < m * n; i++) {
    float c = cS[i];
    float sum = abs(c + c_ref[i]);
    float diff = abs(c - c_ref[i]);

    sum = (sum == 0) ? 1 : sum;

    if (diff > 1e-3 && (diff / sum) > 1e-3) {
      printf("Error at idx: %zu, result: %f, expected: %f\n", i, c, c_ref[i]);
      break;
    }
  }
  */

  printf("Comparing Tensors CUDA\n");
  // half has really low precision
  // any reps more than 1 might throw off the
  // comparators
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      float c = cH[i * n + j];
      float ac = cHcuBlas[j * m + i];

      float sum = abs(c + ac);
      float diff = abs(c - ac);

      sum = (sum == 0) ? 1 : sum;

      if (diff > 1e-3 && (diff / sum) > 1e-2) {
        printf("Error at idx: %zu, %zu, result: %f, expected: %f\n", i, j, c, ac);
        goto end;
      }
    }
  }

  printf("Completed\n");

  end:
  if (cHcuBlas) {
    free(cHcuBlas);
  }

  if (cH) {
    free(cH);
  }

  if (bH) {
    free(bH);
  }

  if (aH) {
    free(aH);
  }

  if (cS) {
    free(cS);
  }

  if (b) {
    free(b);
  }

  if (a) {
    free(a);
  }

  return 0;
}
