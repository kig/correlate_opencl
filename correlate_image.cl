#define HADD(v) ((v).x+(v).y+(v).z+(v).w)

__attribute__((reqd_work_group_size(64,1,1)))
__kernel void correlate (
  __global float *correlation,
  const int corr_size,
  __constant image2d_t base,
  __constant image2d_t mask,
  const int sample_size,
  const int stride )
{
  float4 sum = 0;
  sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int gid = get_global_id(0);
  int offset_y = gid / corr_size;
  int offset_x = gid - (offset_y*corr_size);
  for (int y=offset_y; y < sample_size; y+=2) {
    for (int x=offset_x; x < sample_size; x+=2) {
      for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
          sum += read_imagef(base, s, (int2)(x+i, y+j)) * 
                 read_imagef(mask, s, (int2)(x+i-offset_x, y+j-offset_y));
    }
  }
  
  correlation[gid] = HADD(sum);
}
