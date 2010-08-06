g++ -O3 -msse3 -mfpmath=sse -fopenmp -lOpenCL -lm -o corr_500 corr_500.cpp
g++ -O3 -msse3 -mfpmath=sse -fopenmp -lOpenCL -lm -o corr corr.cpp
g++ -O3 -msse3 -mfpmath=sse -fopenmp -lm -lGLEW -lGL -lGLU -lglut -o corr_gl corr_gl.cpp
g++ -O3 -msse3 -mfpmath=sse -fopenmp -lOpenCL -lm -o corr_naive corr_naive.cpp
g++ -O3 -lOpenCL -lm -o memsum memsum.cpp
