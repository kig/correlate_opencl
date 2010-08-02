#version 120
#extension GL_ARB_texture_rectangle : enable

#define XBSZ 2
#define YBSZ 2

uniform sampler2DRect base;
uniform sampler2DRect mask;
uniform float sample_size;

varying vec2 texCoord0;
void main()
{
  vec4 sum = vec4(0.0);
  float offset_y = floor(texCoord0.y);
  float offset_x = floor(texCoord0.x);
  float last_y = sample_size-offset_y;
  float last_x = sample_size-offset_x;
  for (float yb=0; yb < last_y; yb += YBSZ) {
    for (float xb=0; xb < last_x; xb += XBSZ) {
      for (float y=0; y < YBSZ; y++) {
        float yby = yb + y;
        float ok = clamp(last_y - yby, 0.0, 1.0);
        for (float x=0; x < XBSZ; x++) {
          float xbx = xb + x;
          ok *= clamp(last_x - xbx, 0.0, 1.0);
          sum += ok * texture2DRect(mask, vec2(xbx,yby)) * texture2DRect(base, vec2(xbx+offset_x, yby+offset_y));
        }
      }
    }
  }
  gl_FragColor.x = sum.r + sum.g + sum.b + sum.a;
}
