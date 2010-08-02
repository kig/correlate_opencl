#version 120
attribute vec4 Vertex;
attribute vec2 TexCoord;

varying vec2 texCoord0;

void main()
{
  texCoord0 = TexCoord;
  gl_Position = Vertex;
}
