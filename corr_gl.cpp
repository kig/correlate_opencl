/*

g++ -O3 -msse3 -mfpmath=sse -fopenmp -lm -lGLEW -lGL -lGLU -lglut -o corr_gl corr_gl.cpp

*/
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <malloc.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>


#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glu.h>

#include "sse.h"

using namespace std;

double dtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (((double)t.tv_usec) / 1000000.0);
}

int align(int idx, int n) {
  return (n - idx%n) % n;
}

unsigned char *readFile (const char *filename, size_t *read_bytes)
{
  ifstream file;
  file.open(filename, ios::binary|ios::in|ios::ate);
  size_t sz = file.tellg();
  char *data = (char*)memalign(16, sz+1);
  data[sz] = 0;
  file.seekg(0, ios::beg);
  file.read(data, sz);
  file.close();
  *read_bytes = sz;
  return (unsigned char*)data;
}

bool checkFramebufferStatus() {
    GLenum status;
    status=(GLenum)glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            return true;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
            printf("Framebuffer incomplete,incomplete attachment\n");
            return false;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
            printf("Unsupported framebuffer format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
            printf("Framebuffer incomplete,missing attachment\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            printf("Framebuffer incomplete,attached images must have same dimensions\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
             printf("Framebuffer incomplete,attached images must have same format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
            printf("Framebuffer incomplete,missing draw buffer\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
            printf("Framebuffer incomplete,missing read buffer\n");
            return false;
    }
	return false;
}

void checkGLErrors(const char *label) {
    GLenum errCode;
    const GLubyte *errStr;
    if ((errCode = glGetError()) != GL_NO_ERROR) {
        errStr = gluErrorString(errCode);
        printf("OpenGL ERROR: ");
        printf((char*)errStr);
        printf("(Label: ");
        printf(label);
        printf(")\n.");
    }
}

/**
 * error checking for GLSL
 */
void printProgramInfoLog(GLuint obj) {
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;
    glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
    if (infologLength > 1) {
        infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf(infoLog);
        printf("\n");
        free(infoLog);
    }
}
void printShaderInfoLog(GLuint obj) {
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
    if (infologLength > 1) {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf(infoLog);
        printf("\n");
        free(infoLog);
    }
}

GLint GL_program = -1;

double correlate_openGL
(
 float *correlation, int corr_size,
 const float *base, const float *mask,
 int sample_size)
{
  glViewport(0,0, corr_size, corr_size);

  GLuint fbo, rbo, tex[3];
  glGenTextures(3, tex);

  double t0, buildTime;

  bool debug = false;

  if (debug) printf("init\n");

  t0 = dtime();

  if (GL_program == -1) {
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    size_t len;
    GLchar *vs = (GLchar*)readFile("correlate.vs", &len);
    GLchar *fs = (GLchar*)readFile("correlate.fs", &len);
    if (debug) printf("read shaders\n");
    const GLchar *vv = vs, *ff = fs;
    glShaderSource(v, 1, &vv, NULL);
    glCompileShader(v);
    if (debug) checkGLErrors("vs");
    if (debug) printShaderInfoLog(v);

    glShaderSource(f, 1, &ff, NULL);
    glCompileShader(f);
    if (debug) checkGLErrors("fs");
    if (debug) printShaderInfoLog(f);

    free(vs);
    free(fs);
    if (debug) printf("created shaders\n");
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    if (debug) checkGLErrors("linkProgram");
    if (debug) printProgramInfoLog(p);

    glUseProgram(p);
    if (debug) printf("program ok\n");

    if (debug) checkGLErrors("useProgram");
    GL_program = p;
  }

  buildTime = dtime() - t0;

  glGenFramebuffers(1, &fbo);
  if (debug) printf("genfbo %d!\n", fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  if (debug) printf("fbo!\n");


  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex[0]);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_LUMINANCE32F_ARB, corr_size, corr_size, 0, GL_LUMINANCE, GL_FLOAT, NULL);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_ARB, tex[0], 0);

  if (debug) printf("tex0\n");

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex[1]);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, sample_size, sample_size, 0, GL_RGBA, GL_FLOAT, base);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);

  if (debug) printf("tex1\n");

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex[2]);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, sample_size, sample_size, 0, GL_RGBA, GL_FLOAT, mask);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glActiveTexture(GL_TEXTURE0);

  if (debug) printf("tex2\n");

  if (debug) checkFramebufferStatus();

  if (debug) checkGLErrors("fbotex");

  if (debug) printf("fbotex!\n");

  GLfloat tc[12] = {
    0,0,
    corr_size,0,
    0,corr_size,
    corr_size,0,
    corr_size,corr_size,
    0,corr_size
  };
  GLfloat vc[18] = {
    -1, -1, 0,
    1, -1, 0,
    -1, 1, 0,
    1, -1, 0,
    1, 1, 0,
    -1, 1, 0
  };

  GLuint verta,texa;
  verta = glGetAttribLocation(GL_program, "Vertex");
  texa = glGetAttribLocation(GL_program, "TexCoord");

  if (debug) printf("uniforms\n");
  GLuint b_i, m_i, s_i;
  b_i = glGetUniformLocation(GL_program, "base");
  m_i = glGetUniformLocation(GL_program, "mask");
  s_i = glGetUniformLocation(GL_program, "sample_size");
  if (debug) printf("%d, %d, %d\n", b_i, m_i, s_i);
  glUniform1i(b_i, 1);
  glUniform1i(m_i, 2);
  glUniform1f(s_i, sample_size);
  if (debug) checkGLErrors("uniforms");

  if (debug) printf("vertexattribpointers\n");
  if (debug) printf("%d, %d\n", verta, texa);
  glVertexAttribPointer(texa, 2, GL_FLOAT, GL_FALSE, 0, tc);
  glEnableVertexAttribArray(texa);
  glVertexAttribPointer(verta, 3, GL_FLOAT, GL_FALSE, 0, vc);
  glEnableVertexAttribArray(verta);
  if (debug) checkGLErrors("vertexattribs");

  if (debug) printf("drawarrays\n");
  glDrawArrays(GL_TRIANGLES, 0, 6);

  if (debug) checkGLErrors("drawarrays");

  if (debug) printf("readpixels\n");
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex[0]);
  glGetTexImage(GL_TEXTURE_RECTANGLE_ARB, 0, GL_LUMINANCE, GL_FLOAT, correlation);

  if (debug) checkGLErrors("readPixels");

  if (debug) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glutSwapBuffers();
  }

  if (debug) printf("deletetex\n");
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
  glDeleteTextures(3, tex);
  if (debug) printf("deletefbo\n");
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);

  glFlush();

  return buildTime;
}

void correlate_optimized
(
 float *correlation, int corr_size,
 const float *basef, const float *maskf,
 int sample_size)
{
  const float4* base = (float4*) basef;
  const float4* mask = (float4*) maskf;
  #pragma omp parallel for
  for (int offset_y=0; offset_y < corr_size; offset_y++) {
    for (int rows=0; rows < sample_size-offset_y; rows++) {
      for (int offset_x=0; offset_x < corr_size; offset_x++) {
        float4 sum = float4(0.0);
        int mask_index = rows * sample_size;
        int base_index = (offset_y+rows) * sample_size + offset_x;
        for (int columns=0; columns < sample_size-offset_x; columns++) {
          sum += base[base_index+columns] * mask[mask_index+columns];
        }
        correlation[offset_y*corr_size + offset_x] += sum.sum();
      }
    }
  }
}


float* makeImage(int ssz, bool initialize)
{
  float *img = (float*)memalign(16, ssz*ssz*4*sizeof(float));
  if (initialize)
    for (int i=0; i<ssz*ssz*4; i++)
      img[i] = 0.00625*(i/(ssz*4)) + 0.00625*(i%(ssz*4));
  return img;
}



int main (int argc, char *argv[]) {
  double t0, t1;
  int ssz = 500;
  int csz = ssz/2;

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(ssz/2, ssz/2);
  int win = glutCreateWindow("GPU");

  glewInit();

  float *base = makeImage(ssz, true);
  float *mask = makeImage(ssz, true);
  // reverse mask
  float tmp;
  double tmpd;
  int len = ssz*ssz*4;
  for (int i=0; i<len/2; i++) {
    tmp = mask[len-1-i];
    mask[len-1-i] = mask[i];
    mask[i] = tmp;
  }

  float *corr1 = (float*)memalign(16, csz*csz*sizeof(float));
  float *corr3 = (float*)memalign(16, csz*csz*sizeof(float));
  memset((void*)corr1, 0, csz*csz*sizeof(float));
  memset((void*)corr3, 0, csz*csz*sizeof(float));


  fprintf(stderr, "Achieved bandwidth in GBps, divide by four for GFLOPS\n");
  printf("in_sz\tout_sz\tbw_used\tglsl\tsse_opt\n");


  for (int isz=ssz*ssz; isz<=ssz*ssz; isz+=20000) {
    int sz = sqrt(isz);
    double gb = 1e-9 * (2*sz*0.75*sz*0.75*4*4 * sz*0.5 * sz*0.5 + sz*0.5*sz*0.5);
    printf("%d\t%d\t%.2f", 2*(sz*sz)*16, (sz/2)*(sz/2)*4, gb);
    fflush(stdout);

    t0 = dtime();
    double buildTime = correlate_openGL(corr3, sz/2, base, mask, sz);
    t1 = dtime();
    printf("\t%.2f", gb/(t1-t0-buildTime));
    fflush(stdout);


    t0 = dtime();
    correlate_optimized(corr1, sz/2, base, mask, sz);
    t1 = dtime();
    printf("\t%.2f", gb/(t1-t0));
    fflush(stdout);

    printf("\n");

    for (int i=0; i<(sz/2)*(sz/2); i++) {
      if (
        fabs(corr3[i]-corr1[i]) > fabs(corr1[i]*0.001)
      ) {
        fprintf(stderr, "%d: discrepancy sse_opt %f glsl %f\n", i, corr1[i], corr3[i]);
        break;
      }
    }
  }

  return 0;
}
