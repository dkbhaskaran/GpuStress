all:
	hipcc -Wall -I. -lhipblas -L/opt/rocm/lib -o StressTest GpuStress.cpp

clean:
	rm -f StressTest
