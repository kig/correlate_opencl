#define SUM4(v) ((v).x+(v).y+(v).z+(v).w)

__kernel void correlate (
  __global float8 *correlation,
  const int corr_size,
  __constant float4 *base,
  __constant float4 *mask,
  const int sample_size,
  const int stride )
{
  float4 l_mask[24], l_base[24];
  float4 sums[8] = {0,0,0,0,0,0,0,0};
  float8 sum = 0;
  int gid = get_global_id(0);
  int offset_y = gid*8 / corr_size;
  int offset_x = gid*8 - (offset_y*corr_size);
  for (int y=0; y < sample_size-offset_y; y++) {
    int mask_idx = (offset_y+y)*(stride);
    int base_idx = mask_idx + offset_x;
    for (int x=0; x < sample_size-offset_x; x+=16) {
      for (int i=0; i<24; i++) {
        l_base[i] = base[base_idx+x+i];
        l_mask[i] = mask[mask_idx+x+i];
      }
      for (int i=0; i<16; i++) {
        sums[0] += (l_base[i+0] * l_mask[i+0]);
        sums[1] += (l_base[i+1] * l_mask[i+1]);
        sums[2] += (l_base[i+2] * l_mask[i+2]);
        sums[3] += (l_base[i+3] * l_mask[i+3]);
        sums[4] += (l_base[i+4] * l_mask[i+4]);
        sums[5] += (l_base[i+5] * l_mask[i+5]);
        sums[6] += (l_base[i+6] * l_mask[i+6]);
        sums[7] += (l_base[i+7] * l_mask[i+7]);
      }
    }
  }
  sum.s0 = SUM4(sums[0]);
  sum.s1 = SUM4(sums[1]);
  sum.s2 = SUM4(sums[2]);
  sum.s3 = SUM4(sums[3]);
  sum.s4 = SUM4(sums[4]);
  sum.s5 = SUM4(sums[5]);
  sum.s6 = SUM4(sums[6]);
  sum.s7 = SUM4(sums[7]);
  correlation[gid] = sum;
}
