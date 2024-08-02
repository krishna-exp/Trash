#include <assert.h>
#include <stdio.h>

static inline void expAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("assert failed: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define expErrChk(err) { expAssert((err), __FILE__, __LINE__); }


template<size_t threads>
__global__ static void transpose(float *odata, const float *idata, size_t w, size_t h) {
    // ref: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

    __shared__ float tile[threads][threads + 1];

    int x = blockIdx.x * threads + threadIdx.x;
    int y = blockIdx.y * threads + threadIdx.y;

    if (x < w && y < h)
        tile[threadIdx.y][threadIdx.x] = idata[(y * w) + x];

    __syncthreads();

    x = blockIdx.y * threads + threadIdx.x;
    y = blockIdx.x * threads + threadIdx.y;

    if (x < h && y < w)
        odata[(y * h) + x] = tile[threadIdx.x][threadIdx.y];
}

static void transpose(float *odata, const float *idata, size_t w, size_t h) {
    const int threads = 32;
    const int threadsX = threads;
    const int threadsY = threads;
    const int blocksX = (w + threadsX - 1) / threadsX;
    const int blocksY = (w + threadsY - 1) / threadsY;

    dim3 threadsD(threadsX, threadsY);
    dim3 blocksD(blocksX, blocksY);

    transpose<threads><<<blocksD, threadsD>>>(odata, idata, w, h);
}

int main() {
    size_t m = 128 * 63;
    size_t n = 127 * 128;

    float *in = (float *) malloc(sizeof(float) * m * n);
    float *out = (float *) malloc(sizeof(float) * m * n);

    float *d_in;
    float *d_out;

    expErrChk(cudaMalloc(&d_in, sizeof(float) * m * n));
    expErrChk(cudaMalloc(&d_out, sizeof(float) * m * n));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            const int upper = 10;
            const int lower = 0;

            in[i * n + j] = (rand() % (upper - lower + 1)) + lower;
        }
    }

    expErrChk(cudaMemcpy(d_in, in, sizeof(float) * m * n, cudaMemcpyHostToDevice));

    transpose(d_out, d_in, n, m);
    expErrChk(cudaGetLastError());

    expErrChk(cudaMemcpy(out, d_out, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    /*
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f, ", in[i * n + j]);
        }
        printf("\n");
    }

    printf("------------------------------\n");

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            printf("%f, ", out[j * m + i]);
        }
        printf("\n");
    }

    printf("------------------------------\n");
    */

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (in[i * n + j] != out[j * m + i]) {
                printf("Error found at idx: %d, %d\n", i, j);
                goto exit;
            }
        }
    }
    printf("Completed\n");

    exit:
    expErrChk(cudaFree(d_in));
    expErrChk(cudaFree(d_out));

    free(out);
    free(in);

    return 0;
}
