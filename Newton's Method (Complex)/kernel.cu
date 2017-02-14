#include "kernel.h"
#include <stdio.h>
#define TX 32
#define TY 32
#define LEN 5.f
#define STEPNUMBER 3

// scale coordinates onto [-LEN, LEN]
__device__
float scale(int i, int w) { return 2*LEN*(((1.f*i)/w) - 0.5f); }

// Inputs z = x + iy and outputs zk = xk + iyk after k iterations of Newton's Solver
__device__
float2 newtonComplex(float x, float y, int stepNumber, int *iterations, int *rootnum) {
  float denom = 0.f;
  for (int step = 0; step < stepNumber; step += 1){
    // Thanks Wolfram Alpha for splitting the complex parts...
    denom = 36*x*x*y*y + (3*x*x-3*y*y)*(3*x*x-3*y*y);
    x = -(3*y*y + 3*x*x*x*x*x + 9*x*x*x*y*y - 3*x*x)/denom + x;
    y = -(3*y*y*y*y*y + 3*x*x*x*x*y + 6*x*y)/denom + y;
    if (sqrt((x + 0.5)*(x + 0.5) + (y + 0.866)*(y + 0.866)) < 0.1) {
      *iterations = step;
      *rootnum = 1;
      return make_float2(x,y);
    }
    else if (sqrt((x + 0.5)*(x + 0.5) + (y - 0.866)*(y - 0.866)) < 0.1) {
      *iterations = step;
      *rootnum = 2;
      return make_float2(x,y);
    }
    else if (sqrt((x - 1.0)*(x - 1.0)) < 0.1) {
      *iterations = step;
      *rootnum = 3;
      return make_float2(x,y);
    }
  }
  return make_float2(x,y);;
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
  int numOfIterations = 0;
  int rootnum = 0;
  const float2 pos = newtonComplex(x0, y0, STEPNUMBER, &numOfIterations, &rootnum);

  // assign colors based on distance from roots and number of iterations
  if (rootnum == 1) {
  d_out[i].x = 255; // Red
  d_out[i].y = 0;
  d_out[i].z = 0;
  d_out[i].w = clip((numOfIterations/5)*255);
  }
  else if (rootnum == 2) {
  d_out[i].x = 0;
  d_out[i].y = 255; // Green
  d_out[i].z = 0;
  d_out[i].w = clip((numOfIterations/5)*255);
  }
  else if (rootnum == 3) {
  d_out[i].x = 0;
  d_out[i].y = 0;
  d_out[i].z = 255; // Blue
  d_out[i].w = clip((numOfIterations/5)*255);
  }
  else {
  d_out[i].x = ((c == w/2) || (r == h/2)) ? 255 : 0; // axes
  d_out[i].y = 145;
  d_out[i].z = 255;
  d_out[i].w = 255;
  }

}
void kernelLauncher(uchar4 *d_out, int w, int h, float p, int s) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  stabImageKernel<<<gridSize, blockSize>>>(d_out, w, h, p, s);
}
