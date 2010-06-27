__kernel void memsum (
  __global float4 *dst,
  __global float4 *src,
  const int buf_size )
{
  uint gid = get_global_id(0);
  float4 sum = (float4)0.0f;
  for (int i=0; i<buf_size; i++) {
      sum += src[i];
  }
  dst[gid] = sum;
}
