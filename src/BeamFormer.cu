/**
 *
 *
 */

/**
 * Beamforming:
 *
 *                     Samples:                               Coefficients:
 *                     Channels                               Antennas
 *           S1,1 ......................S1,24     C1,1 ......................C1,24
 *           S2,1 ......................S2,24     C2,1 ......................C2,24
 * Antennas  ...............................   *  ................................    Channels
 *           ................................     ................................
 *           S24,1 ....................S24,24     C24,1 ....................C24,24
 *
 *                                    Beamforming:
 *             B1 =  S1,1*C1,1 + S2,1*C1,2+........... + S24,1*C1,24
 * Channels    .................................................
 *             .................................................
 *             B24 = S1,24*C24,1 + S2,24*C24,2 + ................ + S24,24*C24,24
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
/**
 * CUDA Kernel Device code
 *
 * Compute Beamforming of N_CHANS over N_ANTS
 *
 */
typedef struct complex
{
    float real;
    float imag;
} complex;

__device__ complex multiply(complex x, complex y){
	complex z;
    z.real = x.real * y.real - x.real * y.real;
    z.imag = x.imag * y.imag + x.imag * y.imag;
    return z;
}

#define N_ANTS 24
#define N_CHANS 24
// Kernel definition
__global__ void BeamFormer(complex *A, complex *B, complex *C, int num_chans, int num_ants)
{
    int i = threadIdx.x;
    if (i>N_CHANS) return;
    //int j = threadIdx.y;
    complex sum;
    sum.imag = sum.real = 0;
    for (int k = 0; k < N_ANTS; ++k)
    {
    	complex mul = multiply(A[N_CHANS*(i) + k], B[(i) + N_ANTS*k]);
    	sum.real += mul.real;
    	sum.imag += mul.imag;
    }
    C[i] = sum;
    printf("thread ID.x : %d C[%d] = %f + %fi\n ", threadIdx.x,i,C[i].real, C[i].imag);
    //C[threadIdx.x] = B[threadIdx.x];
    //__syncthreads();
}

/**
 * Host main routine
 */

void constantInit(complex *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i].real = (float) 1.00;
        data[i].imag = (float) 1.00;
        printf ("data [i] = %f -- i = %d\n", data[i].real,i);
    }

    //data[0] = 5.0;
}

int
main(void)
{
	    int devID = 0;



	    cudaError_t error;
	    cudaDeviceProp deviceProp;
	    error = cudaGetDevice(&devID);

	    if (error != cudaSuccess)
	    {
	        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    }

	    error = cudaGetDeviceProperties(&deviceProp, devID);

	    if (deviceProp.computeMode == cudaComputeModeProhibited)
	    {
	        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
	        exit(EXIT_SUCCESS);
	    }

	    if (error != cudaSuccess)
	    {
	        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    }
	    else
	    {
	        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	    }

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    cudaDeviceReset();
    StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);
        //sdkStartTimer(&timer);
    dim3 dimsA(N_ANTS, N_CHANS, 1);
    dim3 dimsB(N_CHANS, N_ANTS, 1);
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(complex) * size_A;
    complex *h_A = (complex *)malloc(mem_size_A);

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(complex) * size_B;
    complex *h_B = (complex *)malloc(mem_size_B);
    printf ("%d -- %d -- %d -- %d\n",dimsA.x, dimsA.y, dimsB.x, dimsB.y );

    // Initialize host memory
        const float valB = 1.0f;
        constantInit(h_A, size_A, 1.0f);
        constantInit(h_B, size_B, valB);

        // Allocate device memory
        complex *d_A, *d_B, *d_C;
        sdkStartTimer(&timer);
        // Allocate host matrix C
        dim3 dimsC(dimsB.x, dimsA.y, 1);
        unsigned long mem_size_C = N_CHANS  * sizeof(complex);
        complex *h_C = (complex *) malloc(mem_size_C);
        printf("size C: %lld\n",mem_size_C);
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }



    // Allocate the device input vector A
    err = cudaMalloc((void **)&d_A, mem_size_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    err = cudaMalloc((void **)&d_B, mem_size_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    err = cudaMalloc((void **)&d_C, mem_size_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*// Launch the BeamFormer CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    BeamFormer<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);*/
    // Kernel invocation
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    dim3 threadsPerBlock(2,2);
    dim3 numBlocks(N_ANTS / threadsPerBlock.x, N_CHANS / threadsPerBlock.y);
    printf ("Num Blocks X : %d :: Num Blocks Y : %d\n",numBlocks.x,numBlocks.y);
    BeamFormer<<<1,N_CHANS>>>(d_A, d_B, d_C,N_CHANS,N_ANTS);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    // Verify that the result vector is correct
        for (int i = 0; i < N_CHANS; ++i)
        {
        	printf ("Chan [i] = %f + %fi -- index = %d\n",h_C[i].real, h_C[i].imag,i);
        }


    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    cudaDeviceReset();


    printf("Done\n");
    return 0;
}

