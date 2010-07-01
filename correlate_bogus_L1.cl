__kernel void correlate (
  __global float8 *correlation,
  const int corr_size,
  __constant float4 *base,
  __constant float4 *mask,
  const int sample_size,
  const int stride )
{
  float4 l_mask[32], l_base[32];
  float8 sum = 0;
  int gid = get_global_id(0);
  int offset_y = gid*8 / corr_size;
  int offset_x = gid*8 - (offset_y*corr_size);
  int y = get_global_id(1);
  if (y < sample_size-offset_y) {
    int mask_idx = y*stride;
    int base_idx = y*stride;
    for (int x=0; x < sample_size-offset_x; x+=16) {
      for (int i=0; i<32; i++) {
        l_base[i] = base[base_idx+x+i];
        l_mask[i] = mask[mask_idx+x+i];
      }
      for (int i=0; i<16; i++) {
        sum.s0 += dot(l_base[i+0], l_mask[i+0]);
        sum.s1 += dot(l_base[i+1], l_mask[i+1]);
        sum.s2 += dot(l_base[i+2], l_mask[i+2]);
        sum.s3 += dot(l_base[i+3], l_mask[i+3]);
        sum.s4 += dot(l_base[i+4], l_mask[i+4]);
        sum.s5 += dot(l_base[i+5], l_mask[i+5]);
        sum.s6 += dot(l_base[i+6], l_mask[i+6]);
        sum.s7 += dot(l_base[i+7], l_mask[i+7]);
      }
    }
    correlation[gid] += sum;
  }
}
