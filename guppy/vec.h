#ifndef GUPPY_VEC_H
#define GUPPY_VEC_H

#include <cuda.h>
#include <cuda_runtime.h>


struct Vec {
  int _n;
  int _nbytes;
  float* _host_data;
  float* _gpu_data;
  bool _host_dirty;
  bool _gpu_dirty;

  void init(int n) {
    _n = n;
    _nbytes = sizeof(float) * n;
    cudaMallocHost(&_host_data, this->_nbytes);
    cudaMalloc(&_gpu_data, this->_nbytes);
    _host_dirty = false;
    _gpu_dirty = true;
  }

  Vec(int n) {
    this->init(n);
  }

  Vec(int n, float fill_value) {
    this->init(n);
    for (int i = 0; i < n; ++i) {
      _host_data[i] = fill_value;
    }
  }

  float* get_gpu_data() {
    if (_gpu_dirty) {
      this->copy_to_gpu();
    }
    _host_dirty = true;
    _gpu_dirty = false;
    return _gpu_data;
  }

  float* get_host_data() {
    if (_host_dirty) {
      this->copy_to_host();
    }
    _gpu_dirty = true;
    _host_dirty = false;
    return _host_data;
  }

  void copy_to_host() {
    cudaMemcpy(this->_host_data, this->_gpu_data, this->_nbytes, cudaMemcpyDeviceToHost);
  }

  void copy_to_gpu() {
    cudaMemcpy(this->_gpu_data, this->_host_data, this->_nbytes, cudaMemcpyHostToDevice);
  }

  ~Vec() {
    cudaFree(_gpu_data);
    cudaFreeHost(_host_data);
  }

};

#endif
