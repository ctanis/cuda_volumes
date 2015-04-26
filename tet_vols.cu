// -*-C++-*-

//#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_ERR_CHECK(x) \
    { cudaError_t err = x; if (err != cudaSuccess) {                \
            fprintf(stderr, "Error \"%s\" at %s:%d \n",	    \
                    cudaGetErrorString(err),                        \
                    __FILE__, __LINE__); exit(-1);                  \
        }}

__device__
double* sub(double* a, double* b, double* z)
{
    int i; for(i=0;i<3;i++)
	   {
	       z[i] = a[i] - b[i];
	   }
    return z;
}


__device__
double* cross(double* a, double* b, double* z)
{
    z[0] = a[1]*b[2] - a[2]*b[1];
    z[1] = a[2]*b[0] - a[0]*b[2];
    z[2] = a[0]*b[1] - a[1]*b[0];
    return z;
}

__device__
double dot(double* a, double* b)
{
    double ans = 0;
    int i; for(i=0;i<3;i++)
	   {
	       ans += a[i]*b[i];
	   }
    return ans;
}

const int threadsPerBlock = 256;//1024;

__global__
void tet_volume(double* vertices, int num_vertices,
		int* tets, int num_tets, double* ans)
{
  __shared__ double cache[threadsPerBlock];

    int ii = threadIdx.x;
    int i = blockDim.x * blockIdx.x + ii;

    if (i >= num_tets) 
      {
	cache[ii] = 0;
      }
    else{
    int j = i*4;

    double* a = &vertices[3*tets[j++]];
    double* b = &vertices[3*tets[j++]];
    double* c = &vertices[3*tets[j++]];
    double* d = &vertices[3*tets[j]];

    double buff[3*3]; //extra memory for calculation
    double* bPtr = &buff[0];

    double* x = cross(sub(b,d,bPtr+6), sub(c,d,bPtr+3), bPtr); //product of cross, stored @ bPtr

    cache[ii] = std::abs(dot(x, sub(a,d,bPtr+3))/6); //reuse of calculation buffer @ bPtr+3
    }

    int add, div;
    add = div = 1;

    __syncthreads();

    while ((div<<=1) <= threadsPerBlock)
      {
	if (!(ii%div))
	  {
	    cache[ii] += cache[ii+add];
	  }
	add<<=1;
	__syncthreads();
      }

    if (! ii)
      ans[blockIdx.x] = cache[0];
}

double calculate_volumes(double* vertices, int num_vertices,
                         int* tets, int num_tets)
{
    int blocksPerGrid =(num_tets + threadsPerBlock - 1) / threadsPerBlock;
    size_t available, total;
    CUDA_ERR_CHECK(cudaMemGetInfo(&available, &total));

    printf("Available: %luMB\nTotal: %luMB\n", available>>20, total>>20);

    double* sizes = (double*)malloc(blocksPerGrid* sizeof(double));

    printf("Blocks per grid: %d\tThreads per block: %d\n",blocksPerGrid,threadsPerBlock);

    double* cuda_vertices;
    int* cuda_tets;
    double* cuda_sizes;

   CUDA_ERR_CHECK(cudaMalloc(&cuda_vertices, num_vertices*3*sizeof(double)));
   CUDA_ERR_CHECK(cudaMalloc(&cuda_tets, num_tets*4*sizeof(int)));
   CUDA_ERR_CHECK(cudaMalloc(&cuda_sizes, blocksPerGrid*sizeof(double)));

    // copy A,B to GPU
    CUDA_ERR_CHECK(cudaMemcpy(cuda_vertices, vertices, num_vertices*3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(cuda_tets, tets, num_tets*4*sizeof(int), cudaMemcpyHostToDevice));

    tet_volume<<<blocksPerGrid, threadsPerBlock>>>(cuda_vertices, num_vertices, cuda_tets, num_tets, cuda_sizes);
    
//    CUDA_ERR_CHECK(cudaGetLastError());
//Previous line gives me this error: 'Error "invalid device function " at tet_vols.cu:53'

    CUDA_ERR_CHECK(cudaMemcpy(sizes, cuda_sizes, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost));
    
    
    double ans = 0;
    for (int i=0; i<blocksPerGrid; i++)
    {
        ans += sizes[i];
	//	printf("%f %d\n", sizes[i],i);
    }
    return ans;
}

