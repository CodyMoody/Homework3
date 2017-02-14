#include "kernel.h"
#include <stdio.h>
#define TX 32
#define TY 32
#define LEN 5.f
#define STEPNUMBER 2

// scale coordinates onto [-LEN, LEN]
__device__
float scale(int i, int w) { return 2*LEN*(((1.f*i)/w) - 0.5f); }

// Newton's Method for Finding Roots
__device__
float fx(float x) {
  return x*x*x - x;
}

__device__
float fxprime(float x) {
  return 3*x*x - 1;
}

__device__
float2 newton(float x, float y, int stepNumber) {
  float dx = 0.f;
  for (float step = 0; step < stepNumber; step += 1){
    dx = fxprime(x);
    x += -(y/dx);
    y = fx(x);
  }
  return make_float2(x,y);
}

__device__
unsigned char clip(float x){ return x > 255 ? 255 : (x < 0 ? 0 : x); }

// kernel function to compute decay and shading
__global__
void stabImageKernel(uchar4 *d_out, int w, int h, float p, int s) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  if ((c >= w) || (r >= h)) return; // Check if within image bounds
  const int i = c + r*w; // 1D indexing
  const float x0 = scale(c, w);
  const float y0 = scale(r, h);
  const float dist_0 = sqrt(x0*x0);
  const float2 pos = newton(x0, y0, STEPNUMBER);
  const float dist_f = sqrt(pos.x*pos.x);
  // assign colors based on distance from origin
  const float dist_r = dist_f/dist_0;
  d_out[i].x = clip(dist_r*255); // red ~ growth
  d_out[i].y = ((c == w/2) || (r == h/2)) ? 255 : 0; // axes
  d_out[i].z = clip((1/dist_r)*255);  // blue - 1/growth
  d_out[i].w = 255;

}

void kernelLauncher(uchar4 *d_out, int w, int h, float p, int s) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  stabImageKernel<<<gridSize, blockSize>>>(d_out, w, h, p, s);
}
