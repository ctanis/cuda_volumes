// -*- C++ -*-

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_ERR_CHECK(x) \
    { cudaError_t err = x; if (err != cudaSuccess) {                \
            fprintf(stderr, "Error \"%s\" at %s:%d \n",	    \
                    cudaGetErrorString(err),                        \
                    __FILE__, __LINE__); exit(-1);                  \
        }}

double copyit(double* vertices, int num_vertices)
{

    size_t available, total;
    CUDA_ERR_CHECK(cudaMemGetInfo(&available, &total));

    printf("Available: %luMB\nTotal: %luMB\n", available>>20, total>>20);

    double* cuda_vertices;

    double *cpyVerts = (double *)malloc(num_vertices*3*sizeof(double));

   CUDA_ERR_CHECK(cudaMalloc(&cuda_vertices, num_vertices*3*sizeof(double)));

    // copy A,B to GPU
    CUDA_ERR_CHECK(cudaMemcpy(cuda_vertices, vertices, num_vertices*3*sizeof(double), cudaMemcpyHostToDevice));

    CUDA_ERR_CHECK(cudaMemcpy(cpyVerts, cuda_vertices, num_vertices*3*sizeof(double), cudaMemcpyDeviceToHost));
    
    double *v, *vo = &cpyVerts[0];	
    double ans = 0;    
    for(v = vo; v<vo+num_vertices*3; v+=3)
    {
	printf("%f\n",*(v+1));
	ans += *(v+1);
    }
    return ans;
}
