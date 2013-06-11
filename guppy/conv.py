#!/usr/bin/env python

import pycuda.autoinit

import glob
from PIL import Image

from mako import template
from pycuda import curandom, compiler, gpuarray
import numpy as N
import pylab


NUM_IMAGES = 16
NUM_FILTERS = 96
FILTER_SIZE = 5

naive_mod = compiler.SourceModule('''
__global__ void convolve(float* img_out, float* images, float *filters,
                         int x, int y, int num_filters, int filter_size) {
  // Each thread operates on one filter
  int filter_idx = threadIdx.y * blockDim.x + threadIdx.x;
  float* filter = filters + filter_idx * (filter_size * filter_size);

  int img_idx = blockIdx.x;
  float* img = images + (img_idx * x * y);
  float* out = img_out + (img_idx * x * y);

  for (int i = 0; i < x - filter_size; ++i) {
    for (int j = 0; j < y - filter_size; ++j) {
      for (int ii = 0; ii < filter_size; ++ii) {
        for (int jj = 0; jj < filter_size; ++jj) {
          out[i * y + j] += img[(i + ii) * y + (j + jj)] * filter[ii * filter_size + jj];
        }
      }
    }
  }
}
''')
naive_convolve = naive_mod.get_function('convolve')

shared_tmpl = template.Template('''
static const int num_filters = ${NUM_FILTERS};
static const int num_images = ${NUM_IMAGES};
static const int filter_size = ${FILTER_SIZE};

__global__ void convolve(float* img_out, float* images, float *gfilters,
                         int x, int y, int num_filters_ignored, int filter_size_ignored) {
  // Each thread operates on one filter
  __shared__ float filters[num_filters][filter_size][filter_size];

  // Cooperate to load filters
  float* f = (float*)filters;
  const int count = num_filters * filter_size * filter_size;
  const int stride = blockDim.x * blockDim.y;
  const int idx = threadIdx.y * blockDim.x + threadIdx.x;
  for (int pos = 0; pos + idx < count; pos += stride) {
    f[pos + idx] = gfilters[pos + idx];
  }
  __syncthreads();
 
  const int pixels = num_images * x * y;
  for (int pos = 0; pos + idx < pixels; pos += stride) {
    img_out[pos + idx] = 0;
  }
  __syncthreads();

  const int filter_idx = threadIdx.y * blockDim.x + threadIdx.x;
  float* filter = (float*)filters[filter_idx];

  const int img_idx = blockIdx.x;
  int offset = img_idx * x * y;
  float* img = images + offset;
  float* out = img_out + offset;

  img_out[0] = 10;
  img_out[3] = 4;
  out[2] = 3;
  for (int i = 0; (i + filter_size) < x; ++i) {
    for (int j = 0; (j + filter_size) < y; ++j) {
      for (int ii = 0; ii < filter_size; ++ii) {
        for (int jj = 0; jj < filter_size; ++jj) {
          atomicAdd(&out[j * x + i], img[(j + jj) * x + (i + ii)] * filter[ii * filter_size + jj]);
        }
      }
    }
  }
}
''').render(**locals())

shared_mod = compiler.SourceModule(shared_tmpl)
shared_convolve = shared_mod.get_function('convolve')

src = N.ndarray((NUM_IMAGES, 256, 256), N.float32)
img_list = glob.glob('/tmp/images/*.JPEG')
for i in range(NUM_IMAGES):
  img = Image.open(img_list[i])
  img = img.convert('L')
  src[i] = img

src = gpuarray.to_gpu(src)
tgt = gpuarray.empty_like(src)

filt = curandom.rand((NUM_FILTERS, FILTER_SIZE, FILTER_SIZE), dtype=N.float32) / \
       N.prod((NUM_FILTERS, FILTER_SIZE, FILTER_SIZE))
       
       

# naive_convolve(tgt, src, filt,
#               N.int32(256), N.int32(256), N.int32(NUM_FILTERS), N.int32(FILTER_SIZE),
#               block=(24, 4, 1),
#               grid=(NUM_IMAGES, 1, 1))

shared_convolve(tgt, src, filt,
    N.int32(256), N.int32(256), N.int32(NUM_FILTERS), N.int32(FILTER_SIZE),
    block=(24, 4, 1),
    grid=(NUM_IMAGES, 1, 1))

print tgt.shape, N.prod(tgt.shape), gpuarray.sum(tgt) / N.prod(tgt.shape)
tgt = tgt.get()
src = src.get()
filt = filt.get()
print tgt.dtype
print tgt[0,:10,:10]

fig, (ax1, ax2) = pylab.subplots(1, 2)
ax1.imshow(src[0].astype(N.uint8), cmap=pylab.gray())
ax2.imshow((256.0 * tgt[0] / tgt[0].max()).astype(N.uint8), cmap=pylab.gray())
pylab.show()
