/*

g++ -O3 -lOpenCL -lm -o memsum memsum.cpp

*/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <malloc.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include <CL/cl.h>

using namespace std;

unsigned char *readFile (const char *filename, size_t *read_bytes)
{
  ifstream file;
  file.open(filename, ios::binary|ios::in|ios::ate);
  size_t sz = file.tellg();
  char *data = (char*)memalign(16, sz+1);
  data[sz] = 0;
  file.seekg(0, ios::beg);
  file.read(data, sz);
  file.close();
  *read_bytes = sz;
  return (unsigned char*)data;
}

void writeFile(const char *filename, char* buf, size_t len)
{
  ofstream file;
  file.open(filename, ios::binary|ios::out|ios::trunc);
  file.write(buf, len);
  file.close();
}

void save_program_binary(cl_program program, const char *filename)
{
  size_t binsize;
  clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(&binsize), (void*)&binsize, NULL);
  const unsigned char *bin = (unsigned char*)malloc(binsize);
  const unsigned char **bins = &bin;
  clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(bins), bins, NULL);
  writeFile(filename, (char*)bin, binsize);
}

#define KERNEL "memsum"
void memsum_openCL( float *dst, float *src, int buf_size, bool useCPU )
{
  cl_platform_id platform;
  clGetPlatformIDs( 1, &platform, NULL );

  cl_device_id device;
  clGetDeviceIDs( platform, useCPU ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, 1, &device, NULL );

  cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, NULL );
  cl_command_queue queue = clCreateCommandQueue( context, device, 0, NULL );

  const char *program_binary_filename = useCPU ? KERNEL ".bc.cpu" : KERNEL ".bc";
  const char *program_source_filename = KERNEL ".cl";

  cl_program program;
  struct stat st;
  bool prebuiltProgram = stat(program_binary_filename, &st) == 0;
  if (prebuiltProgram) {
    size_t programBinaryLength;
    const unsigned char* programBinary = readFile(program_binary_filename, &programBinaryLength);
    program = clCreateProgramWithBinary( context, 1, &device, &programBinaryLength, &programBinary, NULL, NULL );
    clBuildProgram( program, 1, &device, NULL, NULL, NULL );
    free((void*)programBinary);
  } else {
    size_t len;
    const char* programSource = (char*)readFile(program_source_filename, &len);
    program = clCreateProgramWithSource( context, 1, &programSource, NULL, NULL );
    clBuildProgram( program, 1, &device, NULL, NULL, NULL );
    save_program_binary(program, program_binary_filename);
    free((void*)programSource);
  }

  cl_kernel kernel = clCreateKernel( program, KERNEL, NULL );

  cl_mem src_buf = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, buf_size*sizeof(cl_float4), (void*)src, NULL );
  cl_mem dst_buf = clCreateBuffer( context, CL_MEM_WRITE_ONLY, buf_size*sizeof(cl_float4), NULL, NULL );

  clSetKernelArg(kernel, 0, sizeof(dst_buf), (void*) &dst_buf);
  clSetKernelArg(kernel, 1, sizeof(src_buf), (void*) &src_buf);
  clSetKernelArg(kernel, 2, sizeof(cl_int), (void*) &buf_size);

  size_t global_work_size = buf_size;
  clEnqueueNDRangeKernel( queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
  clFinish(queue);

  clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, buf_size*sizeof(cl_float4), (void*)dst, NULL, NULL, NULL);

  clReleaseCommandQueue( queue );
  clReleaseMemObject( src_buf );
  clReleaseMemObject( dst_buf );
  clReleaseKernel( kernel );
  clReleaseProgram( program );
  clReleaseContext( context );
}

double dtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (((double)t.tv_usec) / 1000000.0);
}

float* makeBuf(int sz)
{
  float *img = (float*)memalign(16, sz*sizeof(cl_float4));
  for (int i=0; i<sz*4; i++)
    img[i] = 0.00001f;
  return img;
}


int main () {
  double t0, t1;
  int max_sz = 262144; // 4 MiB (4*float4)

  float *src = makeBuf(max_sz);
  float *dst = makeBuf(max_sz);
  float *dst2 = makeBuf(max_sz);

  // compile OpenCL kernels
  memsum_openCL(dst, src, 10, false);
  memsum_openCL(dst, src, 10, true);

  printf("in_sz\tcl_img\tcl_cpu\n");

  printf("%d", max_sz);

  t0 = dtime();
  memsum_openCL(dst, src, max_sz, false);
  t1 = dtime();
  printf("\t%.4f", (t1-t0));

  t0 = dtime();
  memsum_openCL(dst2, src, max_sz, true);
  t1 = dtime();
  printf("\t%.4f", (t1-t0));

  printf("\n");

  for (int i=0; i<max_sz*4; i++) {
    if (dst[i] != dst2[i] ) {
      fprintf(stderr, "%d: discrepancy: gpu %f cpu %f\n", i, dst[i], dst2[i]);
      break;
    }
  }

  return 0;
}
