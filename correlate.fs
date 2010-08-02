#version 120
#extension GL_ARB_texture_rectangle : enable

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
  for (float y=0; y < last_y; y+=2) {
    float yf = clamp(last_y-y-1, 0.0, 1.0);
    for (float x=0; x < last_x; x+=2) {
      float xf = clamp(last_x-x-1, 0.0, 1.0);
      sum += texture2DRect(mask, vec2(x,y)) *
             texture2DRect(base, vec2(x+offset_x, y+offset_y));
      sum += xf * texture2DRect(mask, vec2(x+1,y)) *
                  texture2DRect(base, vec2(x+1+offset_x, y+offset_y));
      sum += yf * texture2DRect(mask, vec2(x,y+1)) *
                  texture2DRect(base, vec2(x+offset_x, y+1+offset_y));
      sum += xf * yf * texture2DRect(mask, vec2(x+1,y+1)) *
                       texture2DRect(base, vec2(x+1+offset_x, y+1+offset_y));
    }
  }
  gl_FragColor.x = sum.r + sum.g + sum.b + sum.a;
}
