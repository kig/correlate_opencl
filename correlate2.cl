#define HADD(v) ((v).x+(v).y+(v).z+(v).w)

#define X 4

__kernel void correlate (
  __global float *correlation,
  const int corr_size,
  __constant float4 *base,
  __constant float4 *mask,
  const int sample_size,
  const int stride )
{
  int gid = get_global_id(0)*corr_size;
  int offset_y = get_global_id(0);
  for (int y=0; y < sample_size-offset_y; y++) {
   int mask_idx = y*stride;
   for (int offset_x=0; offset_x < corr_size; offset_x+=X) {
    float4 sums[X]; for (int i=0; i<X; i++) sums[i] = 0;
    int base_idx = (offset_y+y)*stride+offset_x;
    for (int x=0; x < sample_size-offset_x; x++) {
      for (int i=0; i<X; i++)
        sums[i] += base[base_idx+x+i] * mask[mask_idx+x];
    }
    for (int i=0; i<X; i++)
      correlation[gid+offset_x+i] += HADD(sums[i]);
   }
  }
}
