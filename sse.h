/*
g++ -msse3 -mfpmath=sse

SSE3 FP vector math structs for C++ with operator overloading.

Uses SSE intrinsics, YMMV.
Includes float4, double2 and double4.

The API uses big-endian element order.
Which might bite you when interacting with other SSE code.
But it makes float4 look like a float[4] in memory, so interacting with
arrays of floats should be easier.

To demonstrate:
  float4 v(1,2,3,4); // (float*)&v == [1,2,3,4]


Remember to align your vector arrays on 16 bytes with e.g.
  float4 *ptr;
  ptr = (float4*)memalign(16, sz*sizeof(float4));
  or
  posix_memalign(&ptr, 16, sz*sizeof(float4));


Have a simple 4x4 matrix multiplication: dst = a X b, ~44 cycles with -O3 -funroll-loops

void mmul4x4 (const float *a, const float *b, float *dst)
{
  for (int i=0; i<16; i+=4) {
    float4 row = float4(a) * float4(b[i]);
    for (int j=1; j<4; j++)
      row += float4(a+j*4) * float4(b[i+j]);
    *(float4*)(&dst[i]) = row;
  }
}

*/

#include <mmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>


struct float4
{
  __m128 xmm;

  float4 () { xmm = _mm_set1_ps(0); }

  float4 (__m128 v) : xmm (v) {}

  float4 (const float4 &v) { xmm = v.xmm; }

  float4 (float v) { xmm = _mm_set1_ps(v); }

  float4 (float x, float y, float z, float w)
  { xmm = _mm_set_ps(w,z,y,x); }

  float4 (const float *v) { xmm = _mm_load_ps(v); }

  float4 shuffle (int a, int b, int c, int d) const
  { return float4(_mm_shuffle_ps(xmm, xmm, _MM_SHUFFLE(d,c,b,a))); }

  float4 shuffle (const float4 &v, int a, int b, int c, int d) const
  { return float4(_mm_shuffle_ps(xmm, v.xmm, _MM_SHUFFLE(d,c,b,a))); }

  float4 hadd (const float4 &v) const
  { return float4(_mm_hadd_ps(xmm, v.xmm)); }

  float4 hsub (const float4 &v) const
  { return float4(_mm_hsub_ps(xmm, v.xmm)); }

  float sum () const
  {
    float c;
    float4 s = hadd(*this).hadd(*this);
    _mm_store_ss(&c, s.xmm);
    return c;
  }

  float dot (const float4 &v) const
  { return (*this * v).sum(); }

  float4 recip () const
  { return float4(_mm_rcp_ps(xmm)); }

  float4 sqrt () const
  { return float4(_mm_sqrt_ps(xmm)); }

  float4 rsqrt () const
  { return float4(_mm_rsqrt_ps(xmm)); }

  float4 min (const float4 &v) const
  { return float4(_mm_min_ps(xmm, v.xmm)); }

  float4 max (const float4 &v) const
  { return float4(_mm_max_ps(xmm, v.xmm)); }

  float4 andnot (const float4 &v) const
  { return float4(_mm_andnot_ps(xmm, v.xmm)); }

  float4 operator& (const float4 &v) const
  { return float4(_mm_and_ps(xmm, v.xmm)); }

  float4 operator| (const float4 &v) const
  { return float4(_mm_or_ps(xmm, v.xmm)); }

  float4 operator^ (const float4 &v) const
  { return float4(_mm_xor_ps(xmm, v.xmm)); }

  float4 operator* (const float4 &v) const
  { return float4(_mm_mul_ps(xmm, v.xmm)); }

  float4 operator+ (const float4 &v) const
  { return float4(_mm_add_ps(xmm, v.xmm)); }

  float4 operator- (const float4 &v) const
  { return float4(_mm_sub_ps(xmm, v.xmm)); }

  float4 operator/ (const float4 &v) const
  { return float4(_mm_div_ps(xmm, v.xmm)); }

  void operator*= (const float4 &v)
  { xmm = _mm_mul_ps(xmm, v.xmm); }

  void operator+= (const float4 &v)
  { xmm = _mm_add_ps(xmm, v.xmm); }

  void operator-= (const float4 &v)
  { xmm = _mm_sub_ps(xmm, v.xmm); }

  void operator/= (const float4 &v)
  { xmm = _mm_div_ps(xmm, v.xmm); }

  float x () const { return ((float*)&xmm)[0]; }
  float y () const { return ((float*)&xmm)[1]; }
  float z () const { return ((float*)&xmm)[2]; }
  float w () const { return ((float*)&xmm)[3]; }

  float4 xyzw () const { return shuffle(0,1,2,3); }
  float4 xywz () const { return shuffle(0,1,3,2); }
  float4 xzyw () const { return shuffle(0,2,1,3); }
  float4 xwyz () const { return shuffle(0,3,1,2); }
  float4 xzwy () const { return shuffle(0,2,3,1); }
  float4 xwzy () const { return shuffle(0,3,2,1); }

  float4 yxzw () const { return shuffle(1,0,2,3); }
  float4 yxwz () const { return shuffle(1,0,3,2); }
  float4 yzxw () const { return shuffle(1,2,0,3); }
  float4 ywxz () const { return shuffle(1,3,0,2); }
  float4 yzwx () const { return shuffle(1,2,3,0); }
  float4 ywzx () const { return shuffle(1,3,2,0); }

  float4 zxyw () const { return shuffle(2,0,1,3); }
  float4 zxwy () const { return shuffle(2,0,3,1); }
  float4 zyxw () const { return shuffle(2,1,0,3); }
  float4 zwxy () const { return shuffle(2,3,0,1); }
  float4 zywx () const { return shuffle(2,1,3,0); }
  float4 zwyx () const { return shuffle(2,3,1,0); }

  float4 wxyz () const { return shuffle(3,0,1,2); }
  float4 wxzy () const { return shuffle(3,0,2,1); }
  float4 wyxz () const { return shuffle(3,1,0,2); }
  float4 wzxy () const { return shuffle(3,2,0,1); }
  float4 wyzx () const { return shuffle(3,1,2,0); }
  float4 wzyx () const { return shuffle(3,2,1,0); }

} __attribute__ ((aligned (16)));


struct double2
{
  __m128d xmm;

  double2 () { xmm = _mm_set1_pd(0); }

  double2 (__m128d v) : xmm (v) {}

  double2 (const double2 &v) { xmm = v.xmm; }

  double2 (double v) { xmm = _mm_set1_pd(v); }

  double2 (double x, double y)
  { xmm = _mm_set_pd(y,x); }

  double2 (const double *v) { xmm = _mm_load_pd(v); }

  double2 shuffle (int a, int b) const
  { return double2(_mm_shuffle_pd(xmm, xmm, _MM_SHUFFLE2(b,a))); }

  double2 shuffle (const double2 &v, int a, int b) const
  { return double2(_mm_shuffle_pd(xmm, v.xmm, _MM_SHUFFLE2(b,a))); }

  double2 hadd (const double2 &v) const
  { return double2(_mm_hadd_pd(xmm, v.xmm)); }

  double2 hsub (const double2 &v) const
  { return double2(_mm_hsub_pd(xmm, v.xmm)); }

  double sum () const
  {
    double c;
    double2 s = hadd(*this);
    _mm_store_sd(&c, s.xmm);
    return c;
  }

  double dot (const double2 &v) const
  { return (*this * v).sum(); }

  double2 sqrt () const
  { return double2(_mm_sqrt_pd(xmm)); }

  double2 min (const double2 &v) const
  { return double2(_mm_min_pd(xmm, v.xmm)); }

  double2 max (const double2 &v) const
  { return double2(_mm_max_pd(xmm, v.xmm)); }

  double2 andnot (const double2 &v) const
  { return double2(_mm_andnot_pd(xmm, v.xmm)); }

  double2 operator& (const double2 &v) const
  { return double2(_mm_and_pd(xmm, v.xmm)); }

  double2 operator| (const double2 &v) const
  { return double2(_mm_or_pd(xmm, v.xmm)); }

  double2 operator^ (const double2 &v) const
  { return double2(_mm_xor_pd(xmm, v.xmm)); }

  double2 operator* (const double2 &v) const
  { return double2(_mm_mul_pd(xmm, v.xmm)); }

  double2 operator+ (const double2 &v) const
  { return double2(_mm_add_pd(xmm, v.xmm)); }

  double2 operator- (const double2 &v) const
  { return double2(_mm_sub_pd(xmm, v.xmm)); }

  double2 operator/ (const double2 &v) const
  { return double2(_mm_div_pd(xmm, v.xmm)); }

  void operator*= (const double2 &v)
  { xmm = _mm_mul_pd(xmm, v.xmm); }

  void operator+= (const double2 &v)
  { xmm = _mm_add_pd(xmm, v.xmm); }

  void operator-= (const double2 &v)
  { xmm = _mm_sub_pd(xmm, v.xmm); }

  void operator/= (const double2 &v)
  { xmm = _mm_div_pd(xmm, v.xmm); }

} __attribute__ ((aligned (16)));


struct double4
{
  __m128d xmm0;
  __m128d xmm1;

  double4 () { xmm0 = _mm_set1_pd(0); xmm1 = _mm_set1_pd(0); }

  double4 (__m128d v, __m128d u) : xmm0 (v), xmm1 (u) {}

  double4 (const double4 &v) { xmm0 = v.xmm0; xmm1 = v.xmm1; }

  double4 (double v) { xmm0 = _mm_set1_pd(v); xmm1 = _mm_set1_pd(v); }

  double4 (double x, double y, double z, double w)
  { xmm0 = _mm_set_pd(y,x); xmm1 = _mm_set_pd(w,z); }

  double4 (const double *v) { xmm0 = _mm_load_pd(v); xmm1 = _mm_load_pd(v+2); }

  double sum () const
  {
    return double2(xmm0).sum() + double2(xmm1).sum();
  }

  double dot (const double4 &v) const
  { return (*this * v).sum(); }

  double4 sqrt () const
  { return double4(_mm_sqrt_pd(xmm0), _mm_sqrt_pd(xmm1)); }

  double4 min (const double4 &v) const
  { return double4(_mm_min_pd(xmm0, v.xmm0), _mm_min_pd(xmm1, v.xmm1)); }

  double4 max (const double4 &v) const
  { return double4(_mm_max_pd(xmm0, v.xmm0), _mm_max_pd(xmm1, v.xmm1)); }

  double4 andnot (const double4 &v) const
  { return double4(_mm_andnot_pd(xmm0, v.xmm0), _mm_andnot_pd(xmm1, v.xmm1)); }

  double4 operator& (const double4 &v) const
  { return double4(_mm_and_pd(xmm0, v.xmm0), _mm_and_pd(xmm1, v.xmm1)); }

  double4 operator| (const double4 &v) const
  { return double4(_mm_or_pd(xmm0, v.xmm0), _mm_or_pd(xmm1, v.xmm1)); }

  double4 operator^ (const double4 &v) const
  { return double4(_mm_xor_pd(xmm0, v.xmm0), _mm_xor_pd(xmm1, v.xmm1)); }

  double4 operator* (const double4 &v) const
  { return double4(_mm_mul_pd(xmm0, v.xmm0), _mm_mul_pd(xmm1, v.xmm1)); }

  double4 operator+ (const double4 &v) const
  { return double4(_mm_add_pd(xmm0, v.xmm0), _mm_add_pd(xmm1, v.xmm1)); }

  double4 operator- (const double4 &v) const
  { return double4(_mm_sub_pd(xmm0, v.xmm0), _mm_sub_pd(xmm1, v.xmm1)); }

  double4 operator/ (const double4 &v) const
  { return double4(_mm_div_pd(xmm0, v.xmm0), _mm_div_pd(xmm1, v.xmm1)); }

  void operator*= (const double4 &v)
  { xmm0 = _mm_mul_pd(xmm0, v.xmm0); xmm1 = _mm_mul_pd(xmm1, v.xmm1); }

  void operator+= (const double4 &v)
  { xmm0 = _mm_add_pd(xmm0, v.xmm0); xmm1 = _mm_add_pd(xmm1, v.xmm1); }

  void operator-= (const double4 &v)
  { xmm0 = _mm_sub_pd(xmm0, v.xmm0); xmm1 = _mm_sub_pd(xmm1, v.xmm1); }

  void operator/= (const double4 &v)
  { xmm0 = _mm_div_pd(xmm0, v.xmm0); xmm1 = _mm_div_pd(xmm1, v.xmm1); }

  double4 shuffle (int a, int b, int c, int d) const
  {
    double *p = ((double*)&xmm0);
    return double4(p[a], p[b], p[c], p[d]);
  }

  double x () const { return ((double*)&xmm0)[0]; }
  double y () const { return ((double*)&xmm0)[1]; }
  double z () const { return ((double*)&xmm1)[0]; }
  double w () const { return ((double*)&xmm1)[1]; }

  double4 xyzw () const { return shuffle(0,1,2,3); }
  double4 xywz () const { return shuffle(0,1,3,2); }
  double4 xzyw () const { return shuffle(0,2,1,3); }
  double4 xwyz () const { return shuffle(0,3,1,2); }
  double4 xzwy () const { return shuffle(0,2,3,1); }
  double4 xwzy () const { return shuffle(0,3,2,1); }

  double4 yxzw () const { return shuffle(1,0,2,3); }
  double4 yxwz () const { return shuffle(1,0,3,2); }
  double4 yzxw () const { return shuffle(1,2,0,3); }
  double4 ywxz () const { return shuffle(1,3,0,2); }
  double4 yzwx () const { return shuffle(1,2,3,0); }
  double4 ywzx () const { return shuffle(1,3,2,0); }

  double4 zxyw () const { return shuffle(2,0,1,3); }
  double4 zxwy () const { return shuffle(2,0,3,1); }
  double4 zyxw () const { return shuffle(2,1,0,3); }
  double4 zwxy () const { return shuffle(2,3,0,1); }
  double4 zywx () const { return shuffle(2,1,3,0); }
  double4 zwyx () const { return shuffle(2,3,1,0); }

  double4 wxyz () const { return shuffle(3,0,1,2); }
  double4 wxzy () const { return shuffle(3,0,2,1); }
  double4 wyxz () const { return shuffle(3,1,0,2); }
  double4 wzxy () const { return shuffle(3,2,0,1); }
  double4 wyzx () const { return shuffle(3,1,2,0); }
  double4 wzyx () const { return shuffle(3,2,1,0); }

} __attribute__ ((aligned (16)));

