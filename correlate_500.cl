#define HADD(v) ((v).x+(v).y+(v).z+(v).w)

#define W 1
#define SUMLEN (W*8)
#define STRIDE 16
#define CACHE (SUMLEN+STRIDE)

__attribute__((reqd_work_group_size(64,1,1)))
__kernel void correlate (
  __global float8 *correlation,
  const int corr_size,
  __constant float4 *base,
  __constant float4 *mask,
  const int sample_size,
  const int stride )
{
  float4 l_base[CACHE], l_mask[CACHE];
  float4 sums[SUMLEN]; for(int i=0; i<SUMLEN; i++) sums[i] = 0;
  int gid = get_global_id(0);
  int offset_y = gid*(W*8) / corr_size;
  int offset_x = gid*(W*8) - (offset_y*corr_size);
  for (int y=0; y < sample_size-offset_y; y++) {
    int mask_idx = y*stride - offset_x;
    int mi = y*stride;
    int base_idx = (offset_y+y)*stride;
    for (int x=offset_x%STRIDE; x < sample_size; x+=STRIDE) {
      if (x < offset_x) {
        for (int i=0; i<8; i++) {
          l_base[i] = base[base_idx+x+i];
          l_mask[i] = mask[mi+x+i];
        }
      } else {
        for (int i=0; i<CACHE; i++) {
          l_base[i] = base[base_idx+x+i];
          l_mask[i] = mask[mask_idx+x+i];
        }
        for (int i=0; i<STRIDE; i++)
          for (int j=0; j<SUMLEN; j++)
            sums[j] += l_base[i+j] * l_mask[i];
      }
    }
  }
  for (int i=0; i<W; i++)
  correlation[gid*W+i] = (float8)(
    HADD(sums[i*8+0]),
    HADD(sums[i*8+1]),
    HADD(sums[i*8+2]),
    HADD(sums[i*8+3]),
    HADD(sums[i*8+4]),
    HADD(sums[i*8+5]),
    HADD(sums[i*8+6]),
    HADD(sums[i*8+7])
  );
}
