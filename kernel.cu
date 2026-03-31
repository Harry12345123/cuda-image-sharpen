#include <cuda_runtime.h>
__device__ int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__global__ void sharpen_kernel(const unsigned char *in, unsigned char *out, int w, int h, int c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const int kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
    for (int ch = 0; ch < c; ++ch)
    {
        int sum = 0;
        for (int ky = -1; ky <= 1; ++ky)
        {
            for (int kx = -1; kx <= 1; ++kx)
            {
                int nx = clamp_int(x + kx, 0, w - 1);
                int ny = clamp_int(y + ky, 0, h - 1);
                sum += kernel[ky + 1][kx + 1] * in[(ny * w + nx) * c + ch];
            }
        }
        out[(y * w + x) * c + ch] = static_cast<unsigned char>(clamp_int(sum, 0, 255));
    }
}
void sharpen_cuda(const unsigned char *in, unsigned char *out, int width, int height, int channels)
{
    size_t bytes = width * height * channels;
    unsigned char *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    sharpen_kernel<<<grid, block>>>(d_in, d_out, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}