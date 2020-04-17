/*
 * -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: 4; -*-
 * Author        : Ahmed Khalil
 * Modified      : 12.04.20
 * 
 * 
 */


__global__ void dftkernel(float* output_re, float* output_im, 
                            const float* input, size_t N)
{
    int tid = (blockIdx.x * blockIdx.x) + threadIdx.x;
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


