// run with corr_size*corr_size global work size

__kernel void correlate (
  __global float *correlation,
  const int corr_size,
  __constant float4 *base,
  __constant float4 *mask,
  const int sample_size )
{
  int gid = get_global_id(0);
  int offset_y = gid / corr_size;
  int offset_x = gid % corr_size;
  float4 sum = 0;
  for (int y=0; y < sample_size-offset_y; y++) {
    int mask_idx = y*sample_size;
    int base_idx = (offset_y+y)*sample_size + offset_x;
    for (int x=0; x < sample_size-offset_x; x++) {
      sum += base[base_idx+x] * mask[mask_idx+x];
    }
  }
  correlation[gid] = sum.x + sum.y + sum.z + sum.w;
}
