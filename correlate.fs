#version 120

uniform sampler2DRect base;
uniform sampler2DRect mask;
uniform float sample_size;

varying vec2 texCoord0;
void main()
{
  vec4 sum = vec4(0.0);
  float offset_x = floor(texCoord0.s);
  float offset_y = floor(texCoord0.t);
  for (float y=0; y < sample_size-offset_y; y++) {
    for (float x=0; x < sample_size-offset_x; x++) {
      sum += texture2DRect(mask, vec2(x,y)) * texture2DRect(base, vec2(x+offset_x, y+offset_y));
    }
  }
  gl_FragColor.x = sum.x + sum.y + sum.z + sum.w;
}
