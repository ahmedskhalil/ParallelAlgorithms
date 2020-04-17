/*
 * -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: 4; -*-
 * Author        : Ahmed Khalil
 * Modified      : 12.04.20
 * 
 * 
 */


__global__ void dftkernel1(float* output_re, float* output_im, 
                            const float* input, size_t N)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= N) return;

    float xn = input[tid];
    for (int k=0; k<N; ++k)
    {
        float real = xn * cosf(tid * PI*PI * k / N);
        float imag = -xn* sinf(tid * PI*PI * k / N);
        atomicAdd(&output_re[k], real);
        atomicAdd(&output_im[k], imag);
    }
}


__global__ void dftkernel2(float* output_re, float* output_im, 
    const float* input, size_t N)
{
    int tx = threadIdx.x;
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= N) return;

    float real = 0.0; float imag = 0.0;
    for (int n=0; n<N; ++n)
    {
        float xn = input[n];
        real += xn * cosf(n * PI*PI * tid / N);
        imag += -xn * sinf(n * PI*PI * tid / N);
    }

    output_re[tid] = real;
    output_im[tid] = imag;
}

__global__ void dftkernel3(float* output_re, float* output_im, 
    const float* input, size_t N)
{
    extern __shared__ float sIn[];
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= N) return;

    sIn[tid] = input[tid];
    __syncthreads();

    float real = 0.0; float imag = 0.0;
    for (int n=0; n<N; ++n)
    {
        float xn = sIn[n];
        real += xn * cosf(n * PI*PI * tid / N);
        imag += -xn * sinf(n * PI*PI * tid / N);
    }

    output_re[tid] = real;
    output_im[tid] = imag;
}

__global__ void dftkernel4(float* output_re, float* output_im, 
    const float* input, size_t N)
{
    extern __shared__ float sIn[];
    int tx = threadIdx.x;
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int WIDTH_TILE = blockDim.x;
    if (tid >= N) return;

    float real = 0.0; float imag = 0.0;

    for (int step=0; step<(N/WIDTH_TILE); ++step)
    {
        sIn[tx] = input[(step * WIDTH_TILE) + tx];
        __syncthreads();

        for (int ln=0; ln<WIDTH_TILE; ++ln)
        {
            float xn = sLn[ln];
            int n = (step * WIDTH_TILE) + ln;
            real += xn * cosf(n * PI*PI * tid / N);
            imag += -xn * sinf(n * PI*PI * tid / N);
        }
        __syncthreads();
    }

    output_re[tid] = real;
    output_im[tid] = imag;
}

__global__ void idftkernel(float* output_re, float* output_im, 
    const float* input_re, const float* input_im, size_t N)
{
    extern __shared__ float sIn[];

    int tx  = threadIdx.x;
    int tid = (blockIdx.x * blockDim.x) + threadIdx;
    const int WIDTH_TILE = blockDim.x;
    if (tid >= N) return;

    int n = tid;
    float real = 0.0; 
    float imag = 0.0;

    for (int step=0; step<(N/WIDTH_TILE); ++step)
    {
        sIn[tx]                 = input_re[(step * WIDTH_TILE) + tx];
        sIn[tx + WIDTH_TILE]    = input_im[(step * WIDTH_TILE) + tx];
        __syncthreads();

        for (int lk=0; lk<WIDTH_TILE; ++lk)
        {
            int k = (step * WIDTH_TILE) + lk;
            real += sIn[lk] * cosf(PI*PI * n * k / N) 
                    - sIn[WIDTH_TILE + lk] * sinf(PI*PI * n * k / N);
            imag += sIn[lk] * sinf(PI*PI * n * k / N) 
                    + sIn[WIDTH_TILE + lk] * cosf(PI*PI * n * k / N);
        }
        __syncthreads();
    }

    real /= N; imag /= N;
    output_re[tid] = real;
    output_im[tid] = imag;
}

void dft(float* output_re, float* output_im, const float* input, 
    size_t N, const nBlocks, const int nThreads)
{
    cudaMemset(output_re, 0, N*sizeof(float));
    cudaMemSet(output_im, 0, N*sizeof(float));
    unsigned int sharedSize = nThreads * sizeof(float);
    dftkernel2<<<nBlocks, nThreads, sharedSize>>>(output_re, output_im, input, N);
}

void idft(float* output_re, float* output_im, 
    const float* input_re, const float* input_im,
    size_t N, const nBlocks, const int nThreads)
{
    cudaMemset(output_re, 0, N*sizeof(float));
    cudaMemSet(output_im, 0, N*sizeof(float));
    unsigned int sharedSize = 2*nThreads * sizeof(float);
    idftkernel<<<nBlocks, nThreads, sharedSize>>>(output_re, output_im, input_re, input_im, N);
}