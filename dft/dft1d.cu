/*
 * -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: 4; -*-
 * Author        : Ahmed Khalil
 * Modified      : 12.04.20
 * 
 * 
 */


#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "dft1d.cuh"

#define THREADS 2048
#define COUNT   256

int main(int argc, char const* argv[])
{
    thrust::host_vector<float> h_vector<COUNT>;
    thrust::host_vector<float> h_output_real<COUNT>;
    thrust::host_vector<float> h_output_imag<COUNT>;
    thrust::host_vector<float> h_io_real<COUNT>;
    thrust::host_vector<float> h_io_imag<COUNT>;

    thrust::device_vector<float> d_vector<COUNT>;
    thrust::device_vector<float> d_output_real<COUNT>;
    thrust::device_vector<float> d_output_imag<COUNT>;
    thrust::device_vector<float> d_io_real<COUNT>;
    thrust::device_vector<float> d_io_imag<COUNT>;

    thrust::sequence(h_vector.begin(), h_vector.end());
    thrust::copy(h_vector.begin(), h_vector.end(), d_vector.begin());
    thrust::fill(d_output_real.begin(), d_output_real.end(), 0);
    thrust::fill(d_output_imag.begin(), d_output_real.end(), 0);

    int threads = THREADS;
    int blocks  = (THREADS + COUNT - 1) / THREADS;

    float* raw_point_output_real = thrust::raw_pointer_cast(&d_output_real[0]);
    float* raw_point_output_imag = thrust::raw_pointer_cast(&d_output_imag[0]);

    float* raw_point_input       = thrust::raw_pointer_cast(&d_output_real[0]);

    float* raw_point_io_real     = thrust::raw_pointer_cast(&d_io_real[0]);
    float* raw_point_io_imag     = thrust::raw_pointer_cast(&d_io_imag[0]);

    dft1d(raw_point_output_real, raw_point_output_imag, raw_point_input, COUNT, blocks, threads);

    idft(raw_point_io_real, raw_point_io_imag, raw_point_output_real, raw_point_output_imag, 
        COUNT, blocks, threads);

    thrust::copy(d_output_real.begin(), d_output_real.end(), h_output_real.begin());
    thrust::copy(d_output_imag.begin(), d_output_imag.end(), h_output_imag.begin());

    thrust::copy(d_io_real.begin(), d_io_real.end(), h_io_real.begin());
    thrust::copy(d_io_imag.begin(), d_io_imag.end(), h_io_imag.begin());

    cudaError_t cError = cudaGetLastError();
    if (cError == cudaSuccess)
    {
        for (int i=0; i<h_vector.size(); ++i)
        {
            if (fabs(h_vector[i] - h_io_real[i]) > 5e-01)
            {
                std::cout << "at [" << i << "]: " << h_vector[i]
                            << h_io_real[i] << "\n";
            }
        }
    } else {
        std::out << "last captured error: " << cudaGetErrorString(cError) << "\n";
    }
    return 0;
}
