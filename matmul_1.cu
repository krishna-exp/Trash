#include <mma.h>

#include <stdio.h>
#include <omp.h>

typedef float data_t;

static inline void expAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("assert failed: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define expErrChk(err) { expAssert((err), __FILE__, __LINE__); }

static void init_mat(data_t *mat, int m, int n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      mat[i * n + j] = (i + 1) * (j + 1);
    }
  }
}

__global__ static void mat_mul_d(data_t *a, data_t *b, data_t *c,
                                 size_t m, size_t n, size_t k) {

  size_t idxX = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t idxY = threadIdx.y + (blockIdx.y * blockDim.y);

  if (idxX >= n || idxY >= m) {
    return;
  }

  data_t sum = 0;

  for (size_t p = 0; p < k; p++) {
    sum += a[idxY * k + p] * b[p * n + idxX];
  }

  c[idxY * n + idxX] = sum;
}

template<int threads>
__global__ static void mat_mul_dt(half *a, half *b, half *c,
                                  size_t m, size_t n, size_t k) {

  // each thread can process 16 * 16 elements
  // So, the cache should be able to feed all of the
  // threads simultaneously.
  // Cache, the bigger, the better.
  constexpr int sDim = threads * threads;
  constexpr int tDim = 16;

  assert(threads % tDim == 0);

  __shared__ half aS[sDim * sDim];
  __shared__ half bS[sDim * sDim];
  __shared__ half cS[sDim * sDim];

  int iT = threadIdx.x;
  int iIdx = blockIdx.x;
  int jT = threadIdx.y;
  int jIdx = blockIdx.y;

  for (int i = 0; i < sDim; i++) {
    cS[iT + (jT * threads) + (i * sDim)] = c[(p0 + iT + (jT * threads)) + (i * k) + (iIdx * sDim * k)];
  }

  for (size_t p0 = 0; p0 < k; p0 += sDim) {

    for (int i = 0; i < sDim; i++) {
      // loading one row of aS using all the threads
      aS[iT + (jT * threads) + (i * sDim)] = a[(p0 + iT + (jT * threads)) + (i * k) + (iIdx * sDim * k)];
    }

    for (int p1 = 0; p1 < sDim; p1++) {
      bS[iT + (jT * threads) + (p1 * sDim)] = b[(iT + (jT * threads) + (jIdx * sDim)) + (p1 * k) + (p0 * k)];
    }

    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tDim, tDim, tDim, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tDim, tDim, tDim, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tDim, tDim, tDim, half, nvcuda::wmma::row_major> c_frag;

    for (int i = 0; i < sDim; i += tDim * threads) {
      for (int j = 0; j < sDim; j += tDim * threads) {
        nvcuda::wmma::load_matrix_sync(c_frag, &cS[((iT * tDim + i) * sDim) + (jT * tDim) + j], sDim);

        for (int p = 0; p < sDim; p += tDim) {
          // Load the inputs
          nvcuda::wmma::load_matrix_sync(a_frag, &aS[p + ((iT * tDim + i) * sDim)], sDim);
          nvcuda::wmma::load_matrix_sync(b_frag, &bS[(jT * tDim + j) + (p * sDim)], sDim);

          // Perform the matrix multiplication
          nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        nvcuda::wmma::store_matrix_sync(&cS[((iT * tDim + i) * sDim) + (jT * tDim) + j], c_frag, sDim, nvcuda::wmma::mem_row_major);
      }
    }
  }

  // just syncing for fun
  // I don't think this would be necessary
  __syncthreads();

  for (int i = 0; i < sDim; i++) {
    c[(p0 + iT + (jT * threads)) + (i * k) + (iIdx * sDim * k)] = cS[iT + (jT * threads) + (i * sDim)];
  }
}

static inline void mat_mul(data_t *da, data_t *db, data_t *dc,
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

static inline void mat_mul(half* da, half* db, half* dc,
                           size_t m, size_t n, size_t k) {

  constexpr int threads = 32;

  int blocksY = (m + threads - 1) / threads;
}

int main() {
  const size_t m_block = 32;
  const size_t n_block = 32;
  const size_t k_block = 32;

  size_t size = 4096;
  size_t m = size;
  size_t n = size;
  size_t k = size;

  // a -> m * k
  // b -> k * n
  // c -> m * n
  data_t *a = (data_t *) malloc(sizeof(data_t) * m * k);
  data_t *b = (data_t *) malloc(sizeof(data_t) * k * n);
  data_t *c = (data_t *) malloc(sizeof(data_t) * m * n);
  data_t *c_ref = (data_t *) malloc(sizeof(data_t) * m * n);

  if (NULL == a || NULL == b || NULL == c) {
    printf("Error allocating\n");
    goto end;
  }

  init_mat(a, m, k);
  init_mat(b, k, n);

  data_t *da;
  data_t *db;
  data_t *dc;

  printf("CUDA incepted\n");

  expErrChk(cudaMalloc((void **) &da, sizeof(data_t) * m * k));
  expErrChk(cudaMalloc((void **) &db, sizeof(data_t) * k * n));
  expErrChk(cudaMalloc((void **) &dc, sizeof(data_t) * m * n));

  expErrChk(cudaMemcpy(da, a, sizeof(data_t) * m * k, cudaMemcpyHostToDevice));
  expErrChk(cudaMemcpy(db, b, sizeof(data_t) * k * n, cudaMemcpyHostToDevice));

  printf("CUDA starting\n");

  mat_mul(da, db, dc, m, n, k);

  expErrChk(cudaMemcpy(c, dc, sizeof(data_t) * m * n, cudaMemcpyDeviceToHost));

  printf("CUDA complete\n");

  #pragma omp parallel for
  for (size_t io = 0; io < m; io += m_block) {
    for (size_t jo = 0; jo < n; jo += n_block) {
      for (size_t po = 0; po < k; po += k_block) {

        for (size_t ii = 0; ii < m_block; ii++) {
          for (size_t ji = 0; ji < n_block; ji++) {
            size_t i = io + ii;
            size_t j = jo + ji;

            data_t sum = c_ref[i * n + j];

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

  for (size_t i = 0; i < m * n; i++) {
    data_t sum = c[i] + c_ref[i];
    data_t diff = c[i] - c_ref[i];

    sum = (sum == 0) ? 1 : sum;
    diff = (diff < 0) ? -diff : diff;
    sum = (sum < 0) ? -sum : sum;

    if ((diff / sum) > 1e-3) {
      printf("Error at idx: %zu, result: %f, expected: %f\n", i, c[i], c_ref[i]);
      break;
    }
  }

  printf("Completed\n");

  expErrChk(cudaFree(dc));
  expErrChk(cudaFree(db));
  expErrChk(cudaFree(da));

  end:
  if (c) {
    free(c);
  }

  if (b) {
    free(b);
  }

  if (a) {
    free(a);
  }

  return 0;
}
