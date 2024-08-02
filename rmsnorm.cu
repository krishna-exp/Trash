#include <assert.h>
#include <stdio.h>

static inline void expAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("assert failed: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define expErrChk(err) { expAssert((err), __FILE__, __LINE__); }

template<size_t blocks, size_t threads>
__global__ static void rmsnorm_ss(float *o, const float *x, size_t size) {
    // o needs to contain atleast "blocks" number of elements

    // ref: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    __shared__ int ss[threads];

    ss[threadIdx.x] = 0;

    size_t idx = threadIdx.x + (blockIdx.x * threads);

    // using threads as template to perform
    // compile time optimizations
    // assert(threads == blockDim.x);

    for (size_t i = idx; i < size; i += blocks * threads) {
        float v = x[i];
        float sq = v * v;
        ss[threadIdx.x] += sq;
    }

    __syncthreads();

    // all the sums must be reduced to one value now
    for (int t = threads >> 1; t > 0; t = t >> 1) {
        if (threadIdx.x < t) {
            ss[threadIdx.x] = ss[threadIdx.x] + ss[threadIdx.x + t];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        o[blockIdx.x] = ss[0];
    }
}

template<size_t blocks, size_t threads>
__global__ static void rmsnorm_sc(float *o, const float *x, const float *w, float ss, size_t size) {
    for (int i = 0; i < size; i += blocks * threads) {
        size_t idx = i + threadIdx.x + (blockIdx.x * threads);
        o[idx] = w[idx] * x[idx] * ss;
    }
}

void rmsnorm_cu(float* o, const float* x, const float* w, int size) {
    const int blocks = 128;
    const int threads = 1024;

    // becuase we are using o
    // as a scratch buffer
    assert(size >= blocks);

    float ss_scratch[blocks] = { 0 };

    // create the ss scratch buffer

    rmsnorm_ss<blocks, threads><<<blocks, threads>>>(o, x, size);
    expErrChk(cudaGetLastError());

    expErrChk(cudaMemcpy(ss_scratch, o, sizeof(float) * min(blocks, size), cudaMemcpyDeviceToHost));

    // calculate sum of squares
    float ss = 0.0f;

    for (int j = 0; j < blocks; j++) {
        ss += ss_scratch[j];
    }

    ss /= size;
    // to avoid divide by 0
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    printf("CUDA ss: %f\n", ss);

    rmsnorm_sc<blocks, threads><<<blocks, threads>>>(o, x, w, ss, size);
    expErrChk(cudaGetLastError());
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += (x[j] * x[j]);
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    printf("CPU ss: %f\n", ss);

    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

bool areClose(float a, float b) {
    float diff = abs(a - b);
    float sum = abs(a + b) + 1e-5;
    return (diff / sum) < 0.001;
}

int main() {
    const int size = 768 * 1024;

    float *x = (float *) malloc(sizeof(float) * size);
    float *w = (float *) malloc(sizeof(float) * size);
    float *o = (float *) malloc(sizeof(float) * size);
    float *o_ref = (float *) malloc(sizeof(float) * size);

    for (int i = 0; i < size; i++) {
        const int upper = 10;
        const int lower = 0;

        x[i] = (rand() % (upper - lower + 1)) + lower;
        w[i] = (rand() % (upper - lower + 1)) + lower;
    }

    printf("Starting\n");

    rmsnorm(o_ref, x, w, size);

    printf("CUDA Starting\n");

    float *d_x;
    float *d_w;
    float *d_o;

    expErrChk(cudaMalloc(&d_x, sizeof(float) * size));
    expErrChk(cudaMalloc(&d_w, sizeof(float) * size));
    expErrChk(cudaMalloc(&d_o, sizeof(float) * size));

    expErrChk(cudaMemcpy(d_x, x, sizeof(float) * size, cudaMemcpyHostToDevice));
    expErrChk(cudaMemcpy(d_w, w, sizeof(float) * size, cudaMemcpyHostToDevice));

    printf("CUDA Compute starting\n");

    rmsnorm_cu(d_o, d_x, d_w, size);

    expErrChk(cudaMemcpy(o, d_o, sizeof(float) * size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++) {
        if (!areClose(o[i], o_ref[i])) {
            printf("Error at idx: %d, expected: %f, actual: %f\n", i, o_ref[i], o[i]);
            break;
        }
    }

    printf("Completed\n");

    expErrChk(cudaFree(d_w));
    expErrChk(cudaFree(d_o));
    expErrChk(cudaFree(d_x));

    free(o_ref);
    free(o);
    free(w);
    free(x);
}
