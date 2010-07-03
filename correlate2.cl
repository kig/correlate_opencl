#define HADD(v) ((v).x+(v).y+(v).z+(v).w)

__kernel void correlate (
  __global float8 *correlation,
  const int corr_size,
  __constant float4 *base,
  __constant float4 *mask,
  const int sample_size,
  const int stride )
{
  float4 l_mults[16];
  int gid = get_global_id(0)*corr_size/8;
  int offset_y = get_global_id(0);
  for (int y=0; y < sample_size-offset_y; y++) {
   int mask_idx = y*stride;
   for (int offset_x=0; offset_x < corr_size; offset_x+=8) {
    float4 sums[8] = {0,0,0,0,0,0,0,0};
    int base_idx = (offset_y+y)*stride+offset_x;
    for (int i=0; i<8; i++) {
      l_mults[i] = base[base_idx+i] * mask[mask_idx+i];
    }
    for (int x=0; x < sample_size-offset_x; x+=8) {
      for (int i=8; i<16; i++) {
        l_mults[i] = base[base_idx+x+i] * mask[mask_idx+x+i];
      }
      for (int i=0; i<8; i++) {
        sums[0] += l_mults[i+0];
        sums[1] += l_mults[i+1];
        sums[2] += l_mults[i+2];
        sums[3] += l_mults[i+3];
        sums[4] += l_mults[i+4];
        sums[5] += l_mults[i+5];
        sums[6] += l_mults[i+6];
        sums[7] += l_mults[i+7];
      }
      for (int i=0; i<8; i++) {
        l_mults[i] = l_mults[i+8];
      }
    }
    correlation[gid+offset_x/8] += (float8)(
      HADD(sums[0]),
      HADD(sums[1]),
      HADD(sums[2]),
      HADD(sums[3]),
      HADD(sums[4]),
      HADD(sums[5]),
      HADD(sums[6]),
      HADD(sums[7])
    );
   }
  }
}
