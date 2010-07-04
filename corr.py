def correlate(corr_size, base, mask, sample_size):
  correlation = range(corr_size*corr_size)
  for offset_y in xrange(corr_size):
    for offset_x in xrange(corr_size):
      sum = 0
      for y in xrange(sample_size-offset_y):
        mask_idx = 4*(y*sample_size)
        base_idx = 4*((y+offset_y)*sample_size + offset_x)
        for x in xrange(sample_size-offset_x):
          sum += (
            base[base_idx] * mask[mask_idx] +
            base[base_idx+1] * mask[mask_idx+1] +
            base[base_idx+2] * mask[mask_idx+2] +
            base[base_idx+3] * mask[mask_idx+3] )
          mask_idx += 4
          base_idx += 4
      correlation[offset_y*corr_size + offset_x] = sum
  return correlation

def mklist(ssz):
  return [ i * 0.000001 for i in xrange(ssz*ssz*4) ]

base = mklist(64)
mask = mklist(64)
correlate(32, base, mask, 64)
