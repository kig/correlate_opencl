__kernel void correlate (
  __global float *correlation,
  const int corr_size,
  const __global float4 *base,
  const __global float4 *mask,
  const int sample_size )
{
  uint gid = get_global_id(0);
  uint offset_y = gid/corr_size;
  uint offset_x = gid-(offset_y*corr_size);
  float4 sum = (float4)(0.0f);
  for (int rows=0; rows < sample_size-offset_y; rows++) {
    int mask_index = (offset_y + rows) * sample_size;
    int base_index = mask_index + offset_x;
    for (int columns=0; columns < sample_size-offset_x; columns++) {
      sum += base[base_index+columns] * mask[mask_index+columns];
    }
  }
  correlation[gid] = sum.s0+sum.s1+sum.s2+sum.s3;
}
