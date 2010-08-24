/*
g++ -msse3 -mfpmath=sse

LICENSE: MIT license
2010 (c) Ilmari Heikkinen <ilmari.heikkinen@gmail.com>

SSE3 FP vector math structs for C++ with operator overloading.

Uses SSE intrinsics, YMMV.
Includes float4, double2 and double4.

The API uses big-endian element order.
Which might bite you when interacting with other SSE code.
But it makes float4 look like a float[4] in memory, so interacting with
arrays of floats should be easier.

To demonstrate:
  float4 v(1,2,3,4); // (float*)&v == {1,2,3,4}


Remember to align your vector arrays on 16 bytes with e.g.
  float4 *ptr;
  ptr = (float4*)memalign(16, sz*sizeof(float4));
  or
  posix_memalign(&ptr, 16, sz*sizeof(float4));


float4 API:
  constructors:
    float4() = (0,0,0,0)
    float4(__m128)   = __m128
    float4(float4 v) = v
    float4(float f)  = (f,f,f,f)
    float4(x,y,z,w)  = (x,y,z,w)
    float4(float *v) = (v[0], v[1], v[2], v[3])

  methods:
    v.shuffle(0,1,2,3) = (v.x, v.y, v.z, v.w)
    v.shuffle(float4 u,0,1,2,3) = (v.x, v.y, u.z, u.w)
    v.hadd(float4 u) = (v.x+v.y, v.z+v.w, u.x+u.y, u.z+u.w)
    v.hsub(float4 u) = (v.x-v.y, v.z-v.w, u.x-u.y, u.z-u.w)
    v.sum() = v.x + v.y + v.z + v.w
    v.dot(float4 u) = (v * u).sum()
    v.recip() = (1/v.x, 1/v.y, 1/v.z, 1/v.w)
    v.sqrt()  = (sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w))
    v.rsqrt() = v.sqrt().recip() using the _mm_rsqrt_ps op
    v & u = zipWith & v u = (v.x&u.x, v.y&u.y, v.z&u.z, v.w&u.w)
    v | u = zipWith | v u
    v ^ u = zipWith ^ v u
    v + u = zipWith + v u
    v - u = zipWith - v u
    v * u = zipWith * v u
    v / u = zipWith / v u
    v += u
    v -= u
    v *= u
    v /= u

  swizzles:
    v.xyzw() = v
    v.wzyx() = (v.w, v.z, v.y, v.x)
    etc.

double2 and double4 have the same methods as float4, with double2 having
API using two elements where float4 uses four (e.g. double2 swizzles are
v.xy() and v.yx() and such).

The double2 and double4 recip() and rsqrt() -methods are implemented in
software (1.0 / this, sqrt().recip()), unlike float4 which uses the SSE
_mm_rcp_ps and _mm_rsqrt_ps -ops.

The double4 swizzles and shuffles are implemented by casting
the struct to a double pointer and doing indexed reads,
so they're pretty slow.

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

  // hooray for generated code

  float4 xxxx () const { return shuffle(0,0,0,0); }
  float4 xxxy () const { return shuffle(0,0,0,1); }
  float4 xxxz () const { return shuffle(0,0,0,2); }
  float4 xxxw () const { return shuffle(0,0,0,3); }
  float4 xxyx () const { return shuffle(0,0,1,0); }
  float4 xxyy () const { return shuffle(0,0,1,1); }
  float4 xxyz () const { return shuffle(0,0,1,2); }
  float4 xxyw () const { return shuffle(0,0,1,3); }
  float4 xxzx () const { return shuffle(0,0,2,0); }
  float4 xxzy () const { return shuffle(0,0,2,1); }
  float4 xxzz () const { return shuffle(0,0,2,2); }
  float4 xxzw () const { return shuffle(0,0,2,3); }
  float4 xxwx () const { return shuffle(0,0,3,0); }
  float4 xxwy () const { return shuffle(0,0,3,1); }
  float4 xxwz () const { return shuffle(0,0,3,2); }
  float4 xxww () const { return shuffle(0,0,3,3); }
  float4 xyxx () const { return shuffle(0,1,0,0); }
  float4 xyxy () const { return shuffle(0,1,0,1); }
  float4 xyxz () const { return shuffle(0,1,0,2); }
  float4 xyxw () const { return shuffle(0,1,0,3); }
  float4 xyyx () const { return shuffle(0,1,1,0); }
  float4 xyyy () const { return shuffle(0,1,1,1); }
  float4 xyyz () const { return shuffle(0,1,1,2); }
  float4 xyyw () const { return shuffle(0,1,1,3); }
  float4 xyzx () const { return shuffle(0,1,2,0); }
  float4 xyzy () const { return shuffle(0,1,2,1); }
  float4 xyzz () const { return shuffle(0,1,2,2); }
  float4 xyzw () const { return shuffle(0,1,2,3); }
  float4 xywx () const { return shuffle(0,1,3,0); }
  float4 xywy () const { return shuffle(0,1,3,1); }
  float4 xywz () const { return shuffle(0,1,3,2); }
  float4 xyww () const { return shuffle(0,1,3,3); }
  float4 xzxx () const { return shuffle(0,2,0,0); }
  float4 xzxy () const { return shuffle(0,2,0,1); }
  float4 xzxz () const { return shuffle(0,2,0,2); }
  float4 xzxw () const { return shuffle(0,2,0,3); }
  float4 xzyx () const { return shuffle(0,2,1,0); }
  float4 xzyy () const { return shuffle(0,2,1,1); }
  float4 xzyz () const { return shuffle(0,2,1,2); }
  float4 xzyw () const { return shuffle(0,2,1,3); }
  float4 xzzx () const { return shuffle(0,2,2,0); }
  float4 xzzy () const { return shuffle(0,2,2,1); }
  float4 xzzz () const { return shuffle(0,2,2,2); }
  float4 xzzw () const { return shuffle(0,2,2,3); }
  float4 xzwx () const { return shuffle(0,2,3,0); }
  float4 xzwy () const { return shuffle(0,2,3,1); }
  float4 xzwz () const { return shuffle(0,2,3,2); }
  float4 xzww () const { return shuffle(0,2,3,3); }
  float4 xwxx () const { return shuffle(0,3,0,0); }
  float4 xwxy () const { return shuffle(0,3,0,1); }
  float4 xwxz () const { return shuffle(0,3,0,2); }
  float4 xwxw () const { return shuffle(0,3,0,3); }
  float4 xwyx () const { return shuffle(0,3,1,0); }
  float4 xwyy () const { return shuffle(0,3,1,1); }
  float4 xwyz () const { return shuffle(0,3,1,2); }
  float4 xwyw () const { return shuffle(0,3,1,3); }
  float4 xwzx () const { return shuffle(0,3,2,0); }
  float4 xwzy () const { return shuffle(0,3,2,1); }
  float4 xwzz () const { return shuffle(0,3,2,2); }
  float4 xwzw () const { return shuffle(0,3,2,3); }
  float4 xwwx () const { return shuffle(0,3,3,0); }
  float4 xwwy () const { return shuffle(0,3,3,1); }
  float4 xwwz () const { return shuffle(0,3,3,2); }
  float4 xwww () const { return shuffle(0,3,3,3); }
  float4 yxxx () const { return shuffle(1,0,0,0); }
  float4 yxxy () const { return shuffle(1,0,0,1); }
  float4 yxxz () const { return shuffle(1,0,0,2); }
  float4 yxxw () const { return shuffle(1,0,0,3); }
  float4 yxyx () const { return shuffle(1,0,1,0); }
  float4 yxyy () const { return shuffle(1,0,1,1); }
  float4 yxyz () const { return shuffle(1,0,1,2); }
  float4 yxyw () const { return shuffle(1,0,1,3); }
  float4 yxzx () const { return shuffle(1,0,2,0); }
  float4 yxzy () const { return shuffle(1,0,2,1); }
  float4 yxzz () const { return shuffle(1,0,2,2); }
  float4 yxzw () const { return shuffle(1,0,2,3); }
  float4 yxwx () const { return shuffle(1,0,3,0); }
  float4 yxwy () const { return shuffle(1,0,3,1); }
  float4 yxwz () const { return shuffle(1,0,3,2); }
  float4 yxww () const { return shuffle(1,0,3,3); }
  float4 yyxx () const { return shuffle(1,1,0,0); }
  float4 yyxy () const { return shuffle(1,1,0,1); }
  float4 yyxz () const { return shuffle(1,1,0,2); }
  float4 yyxw () const { return shuffle(1,1,0,3); }
  float4 yyyx () const { return shuffle(1,1,1,0); }
  float4 yyyy () const { return shuffle(1,1,1,1); }
  float4 yyyz () const { return shuffle(1,1,1,2); }
  float4 yyyw () const { return shuffle(1,1,1,3); }
  float4 yyzx () const { return shuffle(1,1,2,0); }
  float4 yyzy () const { return shuffle(1,1,2,1); }
  float4 yyzz () const { return shuffle(1,1,2,2); }
  float4 yyzw () const { return shuffle(1,1,2,3); }
  float4 yywx () const { return shuffle(1,1,3,0); }
  float4 yywy () const { return shuffle(1,1,3,1); }
  float4 yywz () const { return shuffle(1,1,3,2); }
  float4 yyww () const { return shuffle(1,1,3,3); }
  float4 yzxx () const { return shuffle(1,2,0,0); }
  float4 yzxy () const { return shuffle(1,2,0,1); }
  float4 yzxz () const { return shuffle(1,2,0,2); }
  float4 yzxw () const { return shuffle(1,2,0,3); }
  float4 yzyx () const { return shuffle(1,2,1,0); }
  float4 yzyy () const { return shuffle(1,2,1,1); }
  float4 yzyz () const { return shuffle(1,2,1,2); }
  float4 yzyw () const { return shuffle(1,2,1,3); }
  float4 yzzx () const { return shuffle(1,2,2,0); }
  float4 yzzy () const { return shuffle(1,2,2,1); }
  float4 yzzz () const { return shuffle(1,2,2,2); }
  float4 yzzw () const { return shuffle(1,2,2,3); }
  float4 yzwx () const { return shuffle(1,2,3,0); }
  float4 yzwy () const { return shuffle(1,2,3,1); }
  float4 yzwz () const { return shuffle(1,2,3,2); }
  float4 yzww () const { return shuffle(1,2,3,3); }
  float4 ywxx () const { return shuffle(1,3,0,0); }
  float4 ywxy () const { return shuffle(1,3,0,1); }
  float4 ywxz () const { return shuffle(1,3,0,2); }
  float4 ywxw () const { return shuffle(1,3,0,3); }
  float4 ywyx () const { return shuffle(1,3,1,0); }
  float4 ywyy () const { return shuffle(1,3,1,1); }
  float4 ywyz () const { return shuffle(1,3,1,2); }
  float4 ywyw () const { return shuffle(1,3,1,3); }
  float4 ywzx () const { return shuffle(1,3,2,0); }
  float4 ywzy () const { return shuffle(1,3,2,1); }
  float4 ywzz () const { return shuffle(1,3,2,2); }
  float4 ywzw () const { return shuffle(1,3,2,3); }
  float4 ywwx () const { return shuffle(1,3,3,0); }
  float4 ywwy () const { return shuffle(1,3,3,1); }
  float4 ywwz () const { return shuffle(1,3,3,2); }
  float4 ywww () const { return shuffle(1,3,3,3); }
  float4 zxxx () const { return shuffle(2,0,0,0); }
  float4 zxxy () const { return shuffle(2,0,0,1); }
  float4 zxxz () const { return shuffle(2,0,0,2); }
  float4 zxxw () const { return shuffle(2,0,0,3); }
  float4 zxyx () const { return shuffle(2,0,1,0); }
  float4 zxyy () const { return shuffle(2,0,1,1); }
  float4 zxyz () const { return shuffle(2,0,1,2); }
  float4 zxyw () const { return shuffle(2,0,1,3); }
  float4 zxzx () const { return shuffle(2,0,2,0); }
  float4 zxzy () const { return shuffle(2,0,2,1); }
  float4 zxzz () const { return shuffle(2,0,2,2); }
  float4 zxzw () const { return shuffle(2,0,2,3); }
  float4 zxwx () const { return shuffle(2,0,3,0); }
  float4 zxwy () const { return shuffle(2,0,3,1); }
  float4 zxwz () const { return shuffle(2,0,3,2); }
  float4 zxww () const { return shuffle(2,0,3,3); }
  float4 zyxx () const { return shuffle(2,1,0,0); }
  float4 zyxy () const { return shuffle(2,1,0,1); }
  float4 zyxz () const { return shuffle(2,1,0,2); }
  float4 zyxw () const { return shuffle(2,1,0,3); }
  float4 zyyx () const { return shuffle(2,1,1,0); }
  float4 zyyy () const { return shuffle(2,1,1,1); }
  float4 zyyz () const { return shuffle(2,1,1,2); }
  float4 zyyw () const { return shuffle(2,1,1,3); }
  float4 zyzx () const { return shuffle(2,1,2,0); }
  float4 zyzy () const { return shuffle(2,1,2,1); }
  float4 zyzz () const { return shuffle(2,1,2,2); }
  float4 zyzw () const { return shuffle(2,1,2,3); }
  float4 zywx () const { return shuffle(2,1,3,0); }
  float4 zywy () const { return shuffle(2,1,3,1); }
  float4 zywz () const { return shuffle(2,1,3,2); }
  float4 zyww () const { return shuffle(2,1,3,3); }
  float4 zzxx () const { return shuffle(2,2,0,0); }
  float4 zzxy () const { return shuffle(2,2,0,1); }
  float4 zzxz () const { return shuffle(2,2,0,2); }
  float4 zzxw () const { return shuffle(2,2,0,3); }
  float4 zzyx () const { return shuffle(2,2,1,0); }
  float4 zzyy () const { return shuffle(2,2,1,1); }
  float4 zzyz () const { return shuffle(2,2,1,2); }
  float4 zzyw () const { return shuffle(2,2,1,3); }
  float4 zzzx () const { return shuffle(2,2,2,0); }
  float4 zzzy () const { return shuffle(2,2,2,1); }
  float4 zzzz () const { return shuffle(2,2,2,2); }
  float4 zzzw () const { return shuffle(2,2,2,3); }
  float4 zzwx () const { return shuffle(2,2,3,0); }
  float4 zzwy () const { return shuffle(2,2,3,1); }
  float4 zzwz () const { return shuffle(2,2,3,2); }
  float4 zzww () const { return shuffle(2,2,3,3); }
  float4 zwxx () const { return shuffle(2,3,0,0); }
  float4 zwxy () const { return shuffle(2,3,0,1); }
  float4 zwxz () const { return shuffle(2,3,0,2); }
  float4 zwxw () const { return shuffle(2,3,0,3); }
  float4 zwyx () const { return shuffle(2,3,1,0); }
  float4 zwyy () const { return shuffle(2,3,1,1); }
  float4 zwyz () const { return shuffle(2,3,1,2); }
  float4 zwyw () const { return shuffle(2,3,1,3); }
  float4 zwzx () const { return shuffle(2,3,2,0); }
  float4 zwzy () const { return shuffle(2,3,2,1); }
  float4 zwzz () const { return shuffle(2,3,2,2); }
  float4 zwzw () const { return shuffle(2,3,2,3); }
  float4 zwwx () const { return shuffle(2,3,3,0); }
  float4 zwwy () const { return shuffle(2,3,3,1); }
  float4 zwwz () const { return shuffle(2,3,3,2); }
  float4 zwww () const { return shuffle(2,3,3,3); }
  float4 wxxx () const { return shuffle(3,0,0,0); }
  float4 wxxy () const { return shuffle(3,0,0,1); }
  float4 wxxz () const { return shuffle(3,0,0,2); }
  float4 wxxw () const { return shuffle(3,0,0,3); }
  float4 wxyx () const { return shuffle(3,0,1,0); }
  float4 wxyy () const { return shuffle(3,0,1,1); }
  float4 wxyz () const { return shuffle(3,0,1,2); }
  float4 wxyw () const { return shuffle(3,0,1,3); }
  float4 wxzx () const { return shuffle(3,0,2,0); }
  float4 wxzy () const { return shuffle(3,0,2,1); }
  float4 wxzz () const { return shuffle(3,0,2,2); }
  float4 wxzw () const { return shuffle(3,0,2,3); }
  float4 wxwx () const { return shuffle(3,0,3,0); }
  float4 wxwy () const { return shuffle(3,0,3,1); }
  float4 wxwz () const { return shuffle(3,0,3,2); }
  float4 wxww () const { return shuffle(3,0,3,3); }
  float4 wyxx () const { return shuffle(3,1,0,0); }
  float4 wyxy () const { return shuffle(3,1,0,1); }
  float4 wyxz () const { return shuffle(3,1,0,2); }
  float4 wyxw () const { return shuffle(3,1,0,3); }
  float4 wyyx () const { return shuffle(3,1,1,0); }
  float4 wyyy () const { return shuffle(3,1,1,1); }
  float4 wyyz () const { return shuffle(3,1,1,2); }
  float4 wyyw () const { return shuffle(3,1,1,3); }
  float4 wyzx () const { return shuffle(3,1,2,0); }
  float4 wyzy () const { return shuffle(3,1,2,1); }
  float4 wyzz () const { return shuffle(3,1,2,2); }
  float4 wyzw () const { return shuffle(3,1,2,3); }
  float4 wywx () const { return shuffle(3,1,3,0); }
  float4 wywy () const { return shuffle(3,1,3,1); }
  float4 wywz () const { return shuffle(3,1,3,2); }
  float4 wyww () const { return shuffle(3,1,3,3); }
  float4 wzxx () const { return shuffle(3,2,0,0); }
  float4 wzxy () const { return shuffle(3,2,0,1); }
  float4 wzxz () const { return shuffle(3,2,0,2); }
  float4 wzxw () const { return shuffle(3,2,0,3); }
  float4 wzyx () const { return shuffle(3,2,1,0); }
  float4 wzyy () const { return shuffle(3,2,1,1); }
  float4 wzyz () const { return shuffle(3,2,1,2); }
  float4 wzyw () const { return shuffle(3,2,1,3); }
  float4 wzzx () const { return shuffle(3,2,2,0); }
  float4 wzzy () const { return shuffle(3,2,2,1); }
  float4 wzzz () const { return shuffle(3,2,2,2); }
  float4 wzzw () const { return shuffle(3,2,2,3); }
  float4 wzwx () const { return shuffle(3,2,3,0); }
  float4 wzwy () const { return shuffle(3,2,3,1); }
  float4 wzwz () const { return shuffle(3,2,3,2); }
  float4 wzww () const { return shuffle(3,2,3,3); }
  float4 wwxx () const { return shuffle(3,3,0,0); }
  float4 wwxy () const { return shuffle(3,3,0,1); }
  float4 wwxz () const { return shuffle(3,3,0,2); }
  float4 wwxw () const { return shuffle(3,3,0,3); }
  float4 wwyx () const { return shuffle(3,3,1,0); }
  float4 wwyy () const { return shuffle(3,3,1,1); }
  float4 wwyz () const { return shuffle(3,3,1,2); }
  float4 wwyw () const { return shuffle(3,3,1,3); }
  float4 wwzx () const { return shuffle(3,3,2,0); }
  float4 wwzy () const { return shuffle(3,3,2,1); }
  float4 wwzz () const { return shuffle(3,3,2,2); }
  float4 wwzw () const { return shuffle(3,3,2,3); }
  float4 wwwx () const { return shuffle(3,3,3,0); }
  float4 wwwy () const { return shuffle(3,3,3,1); }
  float4 wwwz () const { return shuffle(3,3,3,2); }
  float4 wwww () const { return shuffle(3,3,3,3); }
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

  double2 recip () const
  { return double2(1.0) / *this; }

  double2 sqrt () const
  { return double2(_mm_sqrt_pd(xmm)); }

  double2 rsqrt () const
  { return sqrt().recip(); }

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

  double x () const { return ((double*)&xmm)[0]; }
  double y () const { return ((double*)&xmm)[1]; }

  double2 xx () const { return shuffle(0,0); }
  double2 xy () const { return shuffle(0,1); }
  double2 yx () const { return shuffle(1,0); }
  double2 yy () const { return shuffle(1,1); }

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

  double4 recip () const
  { return double4(1.0) / *this; }

  double4 sqrt () const
  { return double4(_mm_sqrt_pd(xmm0), _mm_sqrt_pd(xmm1)); }

  double4 rsqrt () const
  { return sqrt().recip(); }

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

  double4 shuffle (const double4 &v, int a, int b, int c, int d) const
  {
    double *p = ((double*)&xmm0);
    double *q = ((double*)&(v.xmm0));
    return double4(p[a], p[b], q[c], q[d]);
  }

  double x () const { return ((double*)&xmm0)[0]; }
  double y () const { return ((double*)&xmm0)[1]; }
  double z () const { return ((double*)&xmm1)[0]; }
  double w () const { return ((double*)&xmm1)[1]; }

  double4 xxxx () const { return shuffle(0,0,0,0); }
  double4 xxxy () const { return shuffle(0,0,0,1); }
  double4 xxxz () const { return shuffle(0,0,0,2); }
  double4 xxxw () const { return shuffle(0,0,0,3); }
  double4 xxyx () const { return shuffle(0,0,1,0); }
  double4 xxyy () const { return shuffle(0,0,1,1); }
  double4 xxyz () const { return shuffle(0,0,1,2); }
  double4 xxyw () const { return shuffle(0,0,1,3); }
  double4 xxzx () const { return shuffle(0,0,2,0); }
  double4 xxzy () const { return shuffle(0,0,2,1); }
  double4 xxzz () const { return shuffle(0,0,2,2); }
  double4 xxzw () const { return shuffle(0,0,2,3); }
  double4 xxwx () const { return shuffle(0,0,3,0); }
  double4 xxwy () const { return shuffle(0,0,3,1); }
  double4 xxwz () const { return shuffle(0,0,3,2); }
  double4 xxww () const { return shuffle(0,0,3,3); }
  double4 xyxx () const { return shuffle(0,1,0,0); }
  double4 xyxy () const { return shuffle(0,1,0,1); }
  double4 xyxz () const { return shuffle(0,1,0,2); }
  double4 xyxw () const { return shuffle(0,1,0,3); }
  double4 xyyx () const { return shuffle(0,1,1,0); }
  double4 xyyy () const { return shuffle(0,1,1,1); }
  double4 xyyz () const { return shuffle(0,1,1,2); }
  double4 xyyw () const { return shuffle(0,1,1,3); }
  double4 xyzx () const { return shuffle(0,1,2,0); }
  double4 xyzy () const { return shuffle(0,1,2,1); }
  double4 xyzz () const { return shuffle(0,1,2,2); }
  double4 xyzw () const { return shuffle(0,1,2,3); }
  double4 xywx () const { return shuffle(0,1,3,0); }
  double4 xywy () const { return shuffle(0,1,3,1); }
  double4 xywz () const { return shuffle(0,1,3,2); }
  double4 xyww () const { return shuffle(0,1,3,3); }
  double4 xzxx () const { return shuffle(0,2,0,0); }
  double4 xzxy () const { return shuffle(0,2,0,1); }
  double4 xzxz () const { return shuffle(0,2,0,2); }
  double4 xzxw () const { return shuffle(0,2,0,3); }
  double4 xzyx () const { return shuffle(0,2,1,0); }
  double4 xzyy () const { return shuffle(0,2,1,1); }
  double4 xzyz () const { return shuffle(0,2,1,2); }
  double4 xzyw () const { return shuffle(0,2,1,3); }
  double4 xzzx () const { return shuffle(0,2,2,0); }
  double4 xzzy () const { return shuffle(0,2,2,1); }
  double4 xzzz () const { return shuffle(0,2,2,2); }
  double4 xzzw () const { return shuffle(0,2,2,3); }
  double4 xzwx () const { return shuffle(0,2,3,0); }
  double4 xzwy () const { return shuffle(0,2,3,1); }
  double4 xzwz () const { return shuffle(0,2,3,2); }
  double4 xzww () const { return shuffle(0,2,3,3); }
  double4 xwxx () const { return shuffle(0,3,0,0); }
  double4 xwxy () const { return shuffle(0,3,0,1); }
  double4 xwxz () const { return shuffle(0,3,0,2); }
  double4 xwxw () const { return shuffle(0,3,0,3); }
  double4 xwyx () const { return shuffle(0,3,1,0); }
  double4 xwyy () const { return shuffle(0,3,1,1); }
  double4 xwyz () const { return shuffle(0,3,1,2); }
  double4 xwyw () const { return shuffle(0,3,1,3); }
  double4 xwzx () const { return shuffle(0,3,2,0); }
  double4 xwzy () const { return shuffle(0,3,2,1); }
  double4 xwzz () const { return shuffle(0,3,2,2); }
  double4 xwzw () const { return shuffle(0,3,2,3); }
  double4 xwwx () const { return shuffle(0,3,3,0); }
  double4 xwwy () const { return shuffle(0,3,3,1); }
  double4 xwwz () const { return shuffle(0,3,3,2); }
  double4 xwww () const { return shuffle(0,3,3,3); }
  double4 yxxx () const { return shuffle(1,0,0,0); }
  double4 yxxy () const { return shuffle(1,0,0,1); }
  double4 yxxz () const { return shuffle(1,0,0,2); }
  double4 yxxw () const { return shuffle(1,0,0,3); }
  double4 yxyx () const { return shuffle(1,0,1,0); }
  double4 yxyy () const { return shuffle(1,0,1,1); }
  double4 yxyz () const { return shuffle(1,0,1,2); }
  double4 yxyw () const { return shuffle(1,0,1,3); }
  double4 yxzx () const { return shuffle(1,0,2,0); }
  double4 yxzy () const { return shuffle(1,0,2,1); }
  double4 yxzz () const { return shuffle(1,0,2,2); }
  double4 yxzw () const { return shuffle(1,0,2,3); }
  double4 yxwx () const { return shuffle(1,0,3,0); }
  double4 yxwy () const { return shuffle(1,0,3,1); }
  double4 yxwz () const { return shuffle(1,0,3,2); }
  double4 yxww () const { return shuffle(1,0,3,3); }
  double4 yyxx () const { return shuffle(1,1,0,0); }
  double4 yyxy () const { return shuffle(1,1,0,1); }
  double4 yyxz () const { return shuffle(1,1,0,2); }
  double4 yyxw () const { return shuffle(1,1,0,3); }
  double4 yyyx () const { return shuffle(1,1,1,0); }
  double4 yyyy () const { return shuffle(1,1,1,1); }
  double4 yyyz () const { return shuffle(1,1,1,2); }
  double4 yyyw () const { return shuffle(1,1,1,3); }
  double4 yyzx () const { return shuffle(1,1,2,0); }
  double4 yyzy () const { return shuffle(1,1,2,1); }
  double4 yyzz () const { return shuffle(1,1,2,2); }
  double4 yyzw () const { return shuffle(1,1,2,3); }
  double4 yywx () const { return shuffle(1,1,3,0); }
  double4 yywy () const { return shuffle(1,1,3,1); }
  double4 yywz () const { return shuffle(1,1,3,2); }
  double4 yyww () const { return shuffle(1,1,3,3); }
  double4 yzxx () const { return shuffle(1,2,0,0); }
  double4 yzxy () const { return shuffle(1,2,0,1); }
  double4 yzxz () const { return shuffle(1,2,0,2); }
  double4 yzxw () const { return shuffle(1,2,0,3); }
  double4 yzyx () const { return shuffle(1,2,1,0); }
  double4 yzyy () const { return shuffle(1,2,1,1); }
  double4 yzyz () const { return shuffle(1,2,1,2); }
  double4 yzyw () const { return shuffle(1,2,1,3); }
  double4 yzzx () const { return shuffle(1,2,2,0); }
  double4 yzzy () const { return shuffle(1,2,2,1); }
  double4 yzzz () const { return shuffle(1,2,2,2); }
  double4 yzzw () const { return shuffle(1,2,2,3); }
  double4 yzwx () const { return shuffle(1,2,3,0); }
  double4 yzwy () const { return shuffle(1,2,3,1); }
  double4 yzwz () const { return shuffle(1,2,3,2); }
  double4 yzww () const { return shuffle(1,2,3,3); }
  double4 ywxx () const { return shuffle(1,3,0,0); }
  double4 ywxy () const { return shuffle(1,3,0,1); }
  double4 ywxz () const { return shuffle(1,3,0,2); }
  double4 ywxw () const { return shuffle(1,3,0,3); }
  double4 ywyx () const { return shuffle(1,3,1,0); }
  double4 ywyy () const { return shuffle(1,3,1,1); }
  double4 ywyz () const { return shuffle(1,3,1,2); }
  double4 ywyw () const { return shuffle(1,3,1,3); }
  double4 ywzx () const { return shuffle(1,3,2,0); }
  double4 ywzy () const { return shuffle(1,3,2,1); }
  double4 ywzz () const { return shuffle(1,3,2,2); }
  double4 ywzw () const { return shuffle(1,3,2,3); }
  double4 ywwx () const { return shuffle(1,3,3,0); }
  double4 ywwy () const { return shuffle(1,3,3,1); }
  double4 ywwz () const { return shuffle(1,3,3,2); }
  double4 ywww () const { return shuffle(1,3,3,3); }
  double4 zxxx () const { return shuffle(2,0,0,0); }
  double4 zxxy () const { return shuffle(2,0,0,1); }
  double4 zxxz () const { return shuffle(2,0,0,2); }
  double4 zxxw () const { return shuffle(2,0,0,3); }
  double4 zxyx () const { return shuffle(2,0,1,0); }
  double4 zxyy () const { return shuffle(2,0,1,1); }
  double4 zxyz () const { return shuffle(2,0,1,2); }
  double4 zxyw () const { return shuffle(2,0,1,3); }
  double4 zxzx () const { return shuffle(2,0,2,0); }
  double4 zxzy () const { return shuffle(2,0,2,1); }
  double4 zxzz () const { return shuffle(2,0,2,2); }
  double4 zxzw () const { return shuffle(2,0,2,3); }
  double4 zxwx () const { return shuffle(2,0,3,0); }
  double4 zxwy () const { return shuffle(2,0,3,1); }
  double4 zxwz () const { return shuffle(2,0,3,2); }
  double4 zxww () const { return shuffle(2,0,3,3); }
  double4 zyxx () const { return shuffle(2,1,0,0); }
  double4 zyxy () const { return shuffle(2,1,0,1); }
  double4 zyxz () const { return shuffle(2,1,0,2); }
  double4 zyxw () const { return shuffle(2,1,0,3); }
  double4 zyyx () const { return shuffle(2,1,1,0); }
  double4 zyyy () const { return shuffle(2,1,1,1); }
  double4 zyyz () const { return shuffle(2,1,1,2); }
  double4 zyyw () const { return shuffle(2,1,1,3); }
  double4 zyzx () const { return shuffle(2,1,2,0); }
  double4 zyzy () const { return shuffle(2,1,2,1); }
  double4 zyzz () const { return shuffle(2,1,2,2); }
  double4 zyzw () const { return shuffle(2,1,2,3); }
  double4 zywx () const { return shuffle(2,1,3,0); }
  double4 zywy () const { return shuffle(2,1,3,1); }
  double4 zywz () const { return shuffle(2,1,3,2); }
  double4 zyww () const { return shuffle(2,1,3,3); }
  double4 zzxx () const { return shuffle(2,2,0,0); }
  double4 zzxy () const { return shuffle(2,2,0,1); }
  double4 zzxz () const { return shuffle(2,2,0,2); }
  double4 zzxw () const { return shuffle(2,2,0,3); }
  double4 zzyx () const { return shuffle(2,2,1,0); }
  double4 zzyy () const { return shuffle(2,2,1,1); }
  double4 zzyz () const { return shuffle(2,2,1,2); }
  double4 zzyw () const { return shuffle(2,2,1,3); }
  double4 zzzx () const { return shuffle(2,2,2,0); }
  double4 zzzy () const { return shuffle(2,2,2,1); }
  double4 zzzz () const { return shuffle(2,2,2,2); }
  double4 zzzw () const { return shuffle(2,2,2,3); }
  double4 zzwx () const { return shuffle(2,2,3,0); }
  double4 zzwy () const { return shuffle(2,2,3,1); }
  double4 zzwz () const { return shuffle(2,2,3,2); }
  double4 zzww () const { return shuffle(2,2,3,3); }
  double4 zwxx () const { return shuffle(2,3,0,0); }
  double4 zwxy () const { return shuffle(2,3,0,1); }
  double4 zwxz () const { return shuffle(2,3,0,2); }
  double4 zwxw () const { return shuffle(2,3,0,3); }
  double4 zwyx () const { return shuffle(2,3,1,0); }
  double4 zwyy () const { return shuffle(2,3,1,1); }
  double4 zwyz () const { return shuffle(2,3,1,2); }
  double4 zwyw () const { return shuffle(2,3,1,3); }
  double4 zwzx () const { return shuffle(2,3,2,0); }
  double4 zwzy () const { return shuffle(2,3,2,1); }
  double4 zwzz () const { return shuffle(2,3,2,2); }
  double4 zwzw () const { return shuffle(2,3,2,3); }
  double4 zwwx () const { return shuffle(2,3,3,0); }
  double4 zwwy () const { return shuffle(2,3,3,1); }
  double4 zwwz () const { return shuffle(2,3,3,2); }
  double4 zwww () const { return shuffle(2,3,3,3); }
  double4 wxxx () const { return shuffle(3,0,0,0); }
  double4 wxxy () const { return shuffle(3,0,0,1); }
  double4 wxxz () const { return shuffle(3,0,0,2); }
  double4 wxxw () const { return shuffle(3,0,0,3); }
  double4 wxyx () const { return shuffle(3,0,1,0); }
  double4 wxyy () const { return shuffle(3,0,1,1); }
  double4 wxyz () const { return shuffle(3,0,1,2); }
  double4 wxyw () const { return shuffle(3,0,1,3); }
  double4 wxzx () const { return shuffle(3,0,2,0); }
  double4 wxzy () const { return shuffle(3,0,2,1); }
  double4 wxzz () const { return shuffle(3,0,2,2); }
  double4 wxzw () const { return shuffle(3,0,2,3); }
  double4 wxwx () const { return shuffle(3,0,3,0); }
  double4 wxwy () const { return shuffle(3,0,3,1); }
  double4 wxwz () const { return shuffle(3,0,3,2); }
  double4 wxww () const { return shuffle(3,0,3,3); }
  double4 wyxx () const { return shuffle(3,1,0,0); }
  double4 wyxy () const { return shuffle(3,1,0,1); }
  double4 wyxz () const { return shuffle(3,1,0,2); }
  double4 wyxw () const { return shuffle(3,1,0,3); }
  double4 wyyx () const { return shuffle(3,1,1,0); }
  double4 wyyy () const { return shuffle(3,1,1,1); }
  double4 wyyz () const { return shuffle(3,1,1,2); }
  double4 wyyw () const { return shuffle(3,1,1,3); }
  double4 wyzx () const { return shuffle(3,1,2,0); }
  double4 wyzy () const { return shuffle(3,1,2,1); }
  double4 wyzz () const { return shuffle(3,1,2,2); }
  double4 wyzw () const { return shuffle(3,1,2,3); }
  double4 wywx () const { return shuffle(3,1,3,0); }
  double4 wywy () const { return shuffle(3,1,3,1); }
  double4 wywz () const { return shuffle(3,1,3,2); }
  double4 wyww () const { return shuffle(3,1,3,3); }
  double4 wzxx () const { return shuffle(3,2,0,0); }
  double4 wzxy () const { return shuffle(3,2,0,1); }
  double4 wzxz () const { return shuffle(3,2,0,2); }
  double4 wzxw () const { return shuffle(3,2,0,3); }
  double4 wzyx () const { return shuffle(3,2,1,0); }
  double4 wzyy () const { return shuffle(3,2,1,1); }
  double4 wzyz () const { return shuffle(3,2,1,2); }
  double4 wzyw () const { return shuffle(3,2,1,3); }
  double4 wzzx () const { return shuffle(3,2,2,0); }
  double4 wzzy () const { return shuffle(3,2,2,1); }
  double4 wzzz () const { return shuffle(3,2,2,2); }
  double4 wzzw () const { return shuffle(3,2,2,3); }
  double4 wzwx () const { return shuffle(3,2,3,0); }
  double4 wzwy () const { return shuffle(3,2,3,1); }
  double4 wzwz () const { return shuffle(3,2,3,2); }
  double4 wzww () const { return shuffle(3,2,3,3); }
  double4 wwxx () const { return shuffle(3,3,0,0); }
  double4 wwxy () const { return shuffle(3,3,0,1); }
  double4 wwxz () const { return shuffle(3,3,0,2); }
  double4 wwxw () const { return shuffle(3,3,0,3); }
  double4 wwyx () const { return shuffle(3,3,1,0); }
  double4 wwyy () const { return shuffle(3,3,1,1); }
  double4 wwyz () const { return shuffle(3,3,1,2); }
  double4 wwyw () const { return shuffle(3,3,1,3); }
  double4 wwzx () const { return shuffle(3,3,2,0); }
  double4 wwzy () const { return shuffle(3,3,2,1); }
  double4 wwzz () const { return shuffle(3,3,2,2); }
  double4 wwzw () const { return shuffle(3,3,2,3); }
  double4 wwwx () const { return shuffle(3,3,3,0); }
  double4 wwwy () const { return shuffle(3,3,3,1); }
  double4 wwwz () const { return shuffle(3,3,3,2); }
  double4 wwww () const { return shuffle(3,3,3,3); }
} __attribute__ ((aligned (16)));

