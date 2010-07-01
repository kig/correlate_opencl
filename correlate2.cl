__kernel void correlate (
  __global float8 *correlation,
  const int corr_size,
  __constant float4 *base,
  __constant float4 *mask,
  const int sample_size,
  const int stride )
{
  float4 l_base[32], l_mask[32];
  float8 sum = 0;
  int gid = get_global_id(0);
  int offset_y = gid*8 / corr_size;
  int offset_x = gid*8 - (offset_y*corr_size);
  for (int y=0; y < sample_size-offset_y; y++) {
    int mask_idx = y*stride;
    int base_idx = (offset_y+y)*stride + offset_x;
    for (int x=0; x < sample_size-offset_x; x+=8) {
      for (int i=0; i<16; ++i) {
        l_base[i] = base[base_idx+x+i];
        l_mask[i] = mask[mask_idx+x+i];
      }
      float f = dot(l_base[0], l_mask[0]);
      float v = ( f +
          dot(l_base[1], l_mask[1]) +
          dot(l_base[2], l_mask[2]) +
          dot(l_base[3], l_mask[3]) +
          dot(l_base[4], l_mask[4]) +
          dot(l_base[5], l_mask[5]) +
          dot(l_base[6], l_mask[6]) +
          dot(l_base[7], l_mask[7]) );
      sum.s0 += v; v = v - f + dot(l_base[8], l_mask[8]);
      f = dot(l_base[1], l_mask[1]);
      sum.s1 += v; v = v - f + dot(l_base[9], l_mask[9]);
      f = dot(l_base[2], l_mask[2]);
      sum.s2 += v; v = v - f + dot(l_base[10], l_mask[10]);
      f = dot(l_base[3], l_mask[3]);
      sum.s3 += v; v = v - f + dot(l_base[11], l_mask[11]);
      f = dot(l_base[4], l_mask[4]);
      sum.s4 += v; v = v - f + dot(l_base[12], l_mask[12]);
      f = dot(l_base[5], l_mask[5]);
      sum.s5 += v; v = v - f + dot(l_base[13], l_mask[13]);
      f = dot(l_base[6], l_mask[6]);
      sum.s6 += v; v = v - f + dot(l_base[14], l_mask[14]);
      f = dot(l_base[7], l_mask[7]);
      sum.s7 += v; v = v - f + dot(l_base[15], l_mask[15]);
    }
  }
  correlation[gid] = sum;
}
