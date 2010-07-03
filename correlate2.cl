#define HADD(v) ((v).x+(v).y+(v).z+(v).w)

#define W 4
#define SUMLEN (W*8)
#define CACHE (SUMLEN+8)

__kernel void correlate (
  __global float8 *correlation,
  const int corr_size,
  __constant float4 *base,
  __constant float4 *mask,
  const int sample_size,
  const int stride )
{
  float4 l_mults[CACHE];
  int gid = get_global_id(0)*corr_size/8;
  int offset_y = get_global_id(0);
  for (int y=0; y < sample_size-offset_y; y++) {
   int mask_idx = y*stride;
   for (int offset_x=0; offset_x < corr_size; offset_x+=SUMLEN) {
    float4 sums[SUMLEN]; for(int i=0; i<SUMLEN; i++) sums[i] = 0;
    int base_idx = (offset_y+y)*stride+offset_x;
    for (int i=0; i<SUMLEN; i++) {
      l_mults[i] = base[base_idx+i] * mask[mask_idx+i];
    }
    for (int x=0; x < sample_size-offset_x; x+=8) {
      for (int i=SUMLEN; i<CACHE; i++) {
        l_mults[i] = base[base_idx+x+i] * mask[mask_idx+x+i];
      }
      for (int j=0; j<SUMLEN; j++)
        for (int i=0; i<8; i++)
          sums[j] += l_mults[i+j];
      for (int i=0; i<SUMLEN; i++) {
        l_mults[i] = l_mults[i+8];
      }
    }
    for (int i=0; i<W; i++)
    correlation[gid+offset_x/8+i] += (float8)(
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
  }
}
