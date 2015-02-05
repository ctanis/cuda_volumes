#include <stdio.h>
#include <cuda_runtime.h>


#define CUDA_ERR_CHECK(x) \
    { cudaError_t err = x; if (err != cudaSuccess) {                \
            fprintf(stderr, "Error \"%s\" at %s:%d \n",             \
                    cudaGetErrorString(err),                        \
                    __FILE__, __LINE__); exit(-1);                  \
        }}

__global__ void tet_volume(double* vertices, int num_vertices,
                           int* tets, int num_tets, double* ans)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_tets)
    {
        // calculate tet volume

        ans[i] =  tets[i*4]+tets[i*4+1]+tets[i*4+2]+tets[i*4+3];
    }


}



double calculate_volumes(double* vertices, int num_vertices,
                         int* tets, int num_tets)
{
    double* sizes = (double*)malloc(num_tets * sizeof(double));

    int threadsPerBlock = 1024;
    int blocksPerGrid =(num_tets + threadsPerBlock - 1) / threadsPerBlock;

    double* cuda_vertices;
    double* cuda_tets;
    double* cuda_sizes;
    

    CUDA_ERR_CHECK(cudaMalloc(&cuda_vertices, num_vertices*3*sizeof(double)));
    CUDA_ERR_CHECK(cudaMalloc(&cuda_tets, num_tets*4*sizeof(int)));
    CUDA_ERR_CHECK(cudaMalloc(&cuda_sizes, num_tets*sizeof(double)));
    
    // copy A,B to GPU
    CUDA_ERR_CHECK(cudaMemcpy(cuda_vertices, vertices, num_vertices*3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(cuda_tets, tets, num_tets*4*sizeof(int), cudaMemcpyHostToDevice));

    tet_volume<<<blocksPerGrid, threadsPerBlock>>>(vertices, num_vertices, tets, num_tets, sizes);
    CUDA_ERR_CHECK(cudaGetLastError());

    CUDA_ERR_CHECK(cudaMemcpy(sizes, cuda_sizes, num_tets*sizeof(double), cudaMemcpyDeviceToHost));
    
    
    double ans = 0;
    // printf("num tets: %d\n", num_tets);
    for (int i=0; i<num_tets; i++)
    {
        ans += sizes[i];
    }
    

    return ans;
}

