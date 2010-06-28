/*

g++ -O3 -msse3 -mfpmath=sse -fopenmp -lOpenCL -lm -o corr corr.cpp

*/
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <malloc.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include <CL/cl.h>

using namespace std;

struct vec4
{
  __m128 xmm;

  vec4 (__m128 v) : xmm (v) {}

  vec4 (float v) { xmm = _mm_set1_ps(v); }

  vec4 (float x, float y, float z, float w)
  { xmm = _mm_set_ps(w,z,y,x); }

  vec4 (const float *v) { xmm = _mm_load_ps(v); }

  vec4 shuffle (int a, int b, int c, int d) const
  { return vec4(_mm_shuffle_ps(xmm, xmm, _MM_SHUFFLE(d,c,b,a))); }

  vec4 shuffle (const vec4 &v, int a, int b, int c, int d) const
  { return vec4(_mm_shuffle_ps(xmm, v.xmm, _MM_SHUFFLE(d,c,b,a))); }

  vec4 hadd (const vec4 &v) const
  { return vec4(_mm_hadd_ps(xmm, v.xmm)); }

  vec4 hsub (const vec4 &v) const
  { return vec4(_mm_hsub_ps(xmm, v.xmm)); }

  float sum () const
  {
    float c;
    vec4 s = hadd(*this).hadd(*this);
    _mm_store_ss(&c, s.xmm);
    return c;
  }

  float dot (const vec4 &v) const
  { return (*this * v).sum(); }

  vec4 operator* (const vec4 &v) const
  { return vec4(_mm_mul_ps(xmm, v.xmm)); }

  vec4 operator+ (const vec4 &v) const
  { return vec4(_mm_add_ps(xmm, v.xmm)); }

  vec4 operator- (const vec4 &v) const
  { return vec4(_mm_sub_ps(xmm, v.xmm)); }

  vec4 operator/ (const vec4 &v) const
  { return vec4(_mm_div_ps(xmm, v.xmm)); }

  void operator*= (const vec4 &v)
  { xmm = _mm_mul_ps(xmm, v.xmm); }

  void operator+= (const vec4 &v)
  { xmm = _mm_add_ps(xmm, v.xmm); }

  void operator-= (const vec4 &v)
  { xmm = _mm_sub_ps(xmm, v.xmm); }

  void operator/= (const vec4 &v)
  { xmm = _mm_div_ps(xmm, v.xmm); }

  void operator>> (float *v) const
  { _mm_store_ps(v, xmm); }

};


void correlate_scalar
(
 float *correlation, int corr_size,
 const float *base, const float *mask,
 int sample_size)
{
  #pragma omp parallel for
  for (int offset_y=0; offset_y < corr_size; offset_y++) {
    for (int offset_x=0; offset_x < corr_size; offset_x++) {
      int correlation_index = offset_y*corr_size + offset_x;
      float sum = 0.0f;
      for (int rows=0; rows < sample_size-offset_y; rows++) {
        for (int columns=0; columns < sample_size-offset_x; columns++) {
          int mi = 4*((offset_y + rows) * sample_size + columns);
          int bi = mi + 4*offset_x;
          sum +=
            base[bi] * mask[mi] +
            base[bi+1] * mask[mi+1] +
            base[bi+2] * mask[mi+2] +
            base[bi+3] * mask[mi+3];
          ;
        }
      }
      correlation[correlation_index] = sum;
    }
  }
}


void correlate
(
 float *correlation, int corr_size,
 const float *basef, const float *maskf,
 int sample_size)
{
  const vec4* base = (vec4*) basef;
  const vec4* mask = (vec4*) maskf;
  #pragma omp parallel for
  for (int offset_y=0; offset_y < corr_size; offset_y++) {
    for (int offset_x=0; offset_x < corr_size; offset_x++) {
      vec4 sum = vec4(0.0);
      for (int rows=0; rows < sample_size-offset_y; rows++) {
        int mask_index = (offset_y + rows) * sample_size;
        int base_index = mask_index + offset_x;
        for (int columns=0; columns < sample_size-offset_x; columns++) {
          sum += base[base_index+columns] * mask[mask_index+columns];
        }
      }
      correlation[offset_y*corr_size + offset_x] = sum.sum();
    }
  }
}


void correlate_optimized
(
 float *correlation, int corr_size,
 const float *basef, const float *maskf,
 int sample_size)
{
  const vec4* base = (vec4*) basef;
  const vec4* mask = (vec4*) maskf;
  vec4* tmp = (vec4*) memalign(16, corr_size*corr_size*sizeof(vec4));
  #pragma omp parallel for
  for (int offset_y=0; offset_y < corr_size; offset_y++) {
    for (int rows=0; rows < sample_size-offset_y; rows++) {
      int mask_index = (offset_y + rows) * sample_size;
      for (int offset_x=0; offset_x < corr_size; offset_x+=16) {
        int corr_idx = offset_y*corr_size + offset_x;
        int base_index = mask_index + offset_x;
        int lidx = corr_size-offset_x > 16 ? 16 : corr_size-offset_x;
        for (int columns=0; columns < sample_size-offset_x; columns++) {
          for (int idx=0; idx<lidx; idx++) {
            tmp[corr_idx+idx] += base[base_index+columns+idx] * mask[mask_index+columns+idx];
          }
        }
      }
    }
  }
  for (int i=0; i<corr_size*corr_size; i++) {
    correlation[i] = tmp[i].sum();
  }
  free(tmp);
}


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



const unsigned char *programBinary = NULL;
size_t programBinaryLength = 0;
const char *programSource = NULL;

void save_program_binary(cl_program program, const char *filename)
{
  size_t binsize;
  clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(&binsize), (void*)&binsize, NULL);
  const unsigned char *bin = (unsigned char*)malloc(binsize);
  const unsigned char **bins = &bin;
  clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(bins), bins, NULL);
  ofstream binfile;
  binfile.open(filename, ios::binary|ios::out|ios::trunc);
  binfile.write((char*)bin, binsize);
  binfile.close();
}

void correlate_openCL
(
 float *correlation, int corr_size,
 const float *base, const float *mask,
 int sample_size)
{
  cl_platform_id platform;
  clGetPlatformIDs( 1, &platform, NULL );

  cl_device_id device;
  clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

  cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, NULL );
  cl_command_queue queue = clCreateCommandQueue( context, device, 0, NULL );

  const char *program_binary_filename = "correlate.bc";
  const char *program_source_filename = "correlate.cl";

  cl_program program;
  struct stat st;
  bool prebuiltProgram = stat(program_binary_filename, &st) == 0;
  if (prebuiltProgram) {
    if (programBinary == NULL) programBinary = readFile(program_binary_filename, &programBinaryLength);
    program = clCreateProgramWithBinary( context, 1, &device, &programBinaryLength, &programBinary, NULL, NULL );
    clBuildProgram( program, 1, &device, NULL, NULL, NULL );
  } else {
    size_t len;
    if (programSource == NULL) programSource = (char*)readFile(program_source_filename, &len);
    program = clCreateProgramWithSource( context, 1, &programSource, NULL, NULL );
    clBuildProgram( program, 1, &device, NULL, NULL, NULL );
    save_program_binary(program, program_binary_filename);
  }

  cl_kernel kernel = clCreateKernel( program, "correlate", NULL );

  cl_mem base_buf = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sample_size*sample_size*4*sizeof(cl_float), (void*)base, NULL );
  cl_mem mask_buf = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sample_size*sample_size*4*sizeof(cl_float), (void*)mask, NULL );
  cl_mem corr_buf = clCreateBuffer( context, CL_MEM_WRITE_ONLY, corr_size*corr_size*sizeof(cl_float), NULL, NULL );

  clSetKernelArg(kernel, 0, sizeof(corr_buf), (void*) &corr_buf);
  clSetKernelArg(kernel, 1, sizeof(cl_int), (void*) &corr_size);
  clSetKernelArg(kernel, 2, sizeof(base_buf), (void*) &base_buf);
  clSetKernelArg(kernel, 3, sizeof(mask_buf), (void*) &mask_buf);
  clSetKernelArg(kernel, 4, sizeof(cl_int), (void*) &sample_size);

  size_t global_work_size = corr_size*corr_size;
  clEnqueueNDRangeKernel( queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
  clFinish(queue);

  clEnqueueReadBuffer(queue, corr_buf, CL_TRUE, 0, corr_size*corr_size*sizeof(cl_float), (void*)correlation, NULL, NULL, NULL);

  clReleaseCommandQueue( queue );
  clReleaseMemObject( base_buf );
  clReleaseMemObject( mask_buf );
  clReleaseMemObject( corr_buf );
  clReleaseKernel( kernel );
  clReleaseProgram( program );
  clReleaseContext( context );
}

double dtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (((double)t.tv_usec) / 1000000.0);
}

float* makeImage(int ssz, bool initialize)
{
  float *img = (float*)memalign(16, ssz*ssz*4*sizeof(float));
  if (initialize)
    for (int i=0; i<ssz*ssz*4; i++)
      img[i] = 0.0f;
  return img;
}


int main () {
  double t0, t1;
  int csz = 256, ssz = 512;

  size_t ilen, mlen;
  const float *base = (const float*)readFile("image", &ilen);
  const float *mask = (const float*)readFile("mask", &mlen);
  fprintf(stderr, "Loaded a %d byte image and %d byte mask\n", ilen, mlen);

  float *corr = (float*)memalign(16, csz*csz*sizeof(float));
  float *corr2 = (float*)memalign(16, csz*csz*sizeof(float));
  float *corr3 = (float*)memalign(16, csz*csz*sizeof(float));

  fprintf(stderr, "Achieved bandwidth in gigabytes per second\n");
  printf("in_sz\tout_sz\tbw_used\tcl\tsse_opt\tsse\n");

  for (int isz=262144; isz<=262144; isz+=20000) {
    int sz = sqrt(isz);
    double gb = 1e-9 * (2*sz*0.75*sz*0.75*4*4 * sz*0.5 * sz*0.5 + sz*0.5*sz*0.5);
    printf("%d\t%d\t%.2f", 2*(sz*sz)*16, (sz/2)*(sz/2)*4, gb);

    t0 = dtime();
    correlate_openCL(corr3, sz/2, base, mask, sz);
    t1 = dtime();
    printf("\t%.4f", gb/(t1-t0));

    t0 = dtime();
    correlate_optimized(corr2, sz/2, base, mask, sz);
    t1 = dtime();
    printf("\t%.4f", gb/(t1-t0));

    t0 = dtime();
    correlate(corr, sz/2, base, mask, sz);
    t1 = dtime();
    printf("\t%.4f", gb/(t1-t0));

    printf("\n");

    for (int i=0; i<(sz/2)*(sz/2); i++) {
      // less than one-thousandth error
      if (
        fabs(corr[i]-corr2[i]) > fabs(corr[i]*0.001) ||
        fabs(corr3[i]-corr2[i]) > fabs(corr2[i]*0.001)
      ) {
        fprintf(stderr, "%d: discrepancy sse %f sse_opt %f gpu %f\n", i, corr[i], corr2[i], corr3[i]);
        break;
      }
    }

  }

  return 0;
}
