/*
 * -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: 4; -*-
 * Author        : Ahmed Khalil
 * Modified      : 12.04.20
 * Ref           : The DFT: An Owners' Manual for the Discrete Fourier Transform
 *                          MultiDim. DFTs 5.2
 * 
 */

 #define PI (3.14159265359)

__global__ void dft2dkernel1(float* output_re, float* output_im, 
    const float* input, size_t M, size_t N)
{
    int m = (blockIdx.y * blockDim.y) + threadIdx.y;
    int n = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (m >= M || n >= N) return;

    float fmn = input[m * N + n];

    for (int k=0; k<M; ++k)
    {
        for (int l=0; l<N; ++l)
        {
            float tmp  = ((float) * m / M) + ((float)l * n / N));
            float real = fmn * cosf(PI*PI * tmp);
            float imag = -fmn * sinf(PI*PI *tmp);

            atomicAdd(&output_re[k * N + l], real);
            atomicAdd(&output_im[k * N + l], imag);
        }
    }

}

__global__ void dft2dkernel2(float* output_re, float* output_im, 
    const float* input, size_t M, size_t N)
{
    int m = (blockIdx.y * blockDim.y) + threadIdx.y;
    int n = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (m >= M || n >= N) return;

    float fmn = input[m * N + n];

    for (int k=0; k<M; ++k)
    {
        for (int l=0; l<N; ++l)
        {
            float tmp  = ((float) * m / M) + ((float)l * n / N));
            float real = fmn * cosf(PI*PI * tmp);
            float imag = -fmn * sinf(PI*PI *tmp);

            atomicAdd(&output_re[k * N + l], real);
            atomicAdd(&output_im[k * N + l], imag);
        }
    }

}


__global__ void dft2dkernel3(float* output_re, float* output_im, 
    const float* input, size_t M, size_t N)
{
    int l = (blockIdx.y * blockDim.y) + threadIdx.y;
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k >= M || l >= N) return;

    float realkl = 0; float imagkl = 0;

    for (int n=0; n<N; ++n)
    {
        for (int m=0; m<M; ++m)
        {
            float fmn = input[m * N + n];
            float tmp  = ((float) * m / M) + ((float)l * n / N));
            realkl += fmn * cosf(PI*PI * tmp);
            imagkl += -fmn * sinf(PI*PI *tmp);

        }
    }
    output_re[k * N + l] = realkl;
    output_im[k * N + l] = imagkl;

}

__global__ void dft2dkernel4(float* output_re, float* output_im, 
    const float* input, size_t M, size_t N)
{
    extern __shared__ float part[];

    int TILE_SIZE_X = blockDim.x;
    int TILE_SIZE_Y = blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int l = (blockIdx.y * blockDim.y) + threadIdx.y;
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k >= M || l >= N) return;

    float realkl = 0; float imagkl = 0;

    for (int m=0; m<M/TILE_SIZE_Y; ++m)
    {
        for (int n=0; n<N/TILE_SIZE_X; ++n)
        {
            int row = (m * TILE_SIZE_Y) + ty;
            int col = (n * TILE_SIZE_X) + tx;
            part[ty * TILE_SIZE_X + tx] = input[(row * N) + col];
            __syncthreads();

            for (int lm=0; lm<TILE_SIZE_Y; ++lm)
            {
                for (int ln=0; ln<TILE_SIZE_X; ++ln)
                {
                    int m = (m * TILE_SIZE_Y) + lm;
                    int n  =(n * TILE_SIZE_X) + ln;

                    float fmn = part[(lm * TILE_SIZE_X) + ln];
                    float tmp = ((float)k * m) / M + ((float)l * n) / N;

                    realkl += fmn  * cosf(PI*PI * tmp);
                    imagkl += -fmn * sinf(PI*PI * tmp);
                }
            }
            __syncthreads();
        }
    }

    output_re[k*N + l] = realkl;
    output_im[k*N + l] = imagkl;
}