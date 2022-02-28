#include <GpuStress.hpp>
#include <sched.h>
#include <unistd.h>
#include <vector>

Logger Logger::Glogger("Stress.log");

class GpuUtils {
public:
  static int getDeviceCount() {
    int DeviceCount = 0;
    checkError(hipGetDeviceCount(&DeviceCount));

    if (!DeviceCount)
      throw std::string("No CUDA devices");
    return DeviceCount;
  }

  static size_t getAvailableMemory() {
    size_t FreeMem, TotalMem;
    checkError(hipMemGetInfo(&FreeMem, &TotalMem));
    return FreeMem;
  }
};

class GpuGemmWorker {
public:
  GpuGemmWorker(int DevId, float *M1, float *M2, size_t Sz)
      : DeviceId(DevId), Size(Sz), MatA(M1), MatB(M2) {
    checkError(hipSetDevice(DeviceId));
    checkError(hipblasCreate(&Handle), "init");

#define PERCENT_MEM_TO_USE 80 // Try to allocate 80% of memory
    size_t MemoryToUse = (ssize_t)((double)GpuUtils::getAvailableMemory() *
                                   PERCENT_MEM_TO_USE / 100);

    size_t ResultSize = sizeof(float) * Size * Size;
    Iterations = (MemoryToUse - 2 * ResultSize) / ResultSize;

    Logger::Log(Logger::Debug, std::string("Thread id ") + "Results are " +
                                   std::to_string(ResultSize) +
                                   " bytes each, thus performing " +
                                   std::to_string(Iterations) +
                                   " iterations\n");

    checkError(hipMalloc(&DeviceMatA, ResultSize), "Device Source1 alloc");
    checkError(hipMalloc(&DeviceMatB, ResultSize), "Device Source2  alloc");
    checkError(hipMalloc(&DeviceMatC, Iterations * ResultSize),
               "Device Result alloc");

    // Populating matrices A and B
    checkError(hipMemcpyHtoD(DeviceMatA, MatA, ResultSize), "MatA -> device");
    checkError(hipMemcpyHtoD(DeviceMatB, MatB, ResultSize), "MatB -> device");

    checkError(hipMalloc(&DeviceErrors, sizeof(int)), "faulty data");
  }

  virtual ~GpuGemmWorker() {
    checkError(hipSetDevice(DeviceId));
    checkError(hipFree(DeviceMatA), "Free A");
    checkError(hipFree(DeviceMatB), "Free B");
    checkError(hipFree(DeviceMatC), "Free C");
    checkError(hipFree(DeviceErrors), "Free DeviceErrors");

    checkError(hipblasDestroy(Handle), "blas handle destroy");
  }

  template <class T>
  static __global__ void compareKernel(float *C, int *Errors,
                                       size_t Iterations) {
#define EPSILON 0.0001
    size_t Step = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    size_t Index = (blockIdx.y * blockDim.y + threadIdx.y) * // Y
                       gridDim.x * blockDim.x +              // W
                   blockIdx.x * blockDim.x +
                   threadIdx.x; // X

    int Faults = 0;
    for (size_t i = 1; i < Iterations; ++i)
      if (fabs(C[Index] - C[Index + i * Step]) > EPSILON)
        Faults++;
    atomicAdd(Errors, Faults);
  }

  void start() {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < Iterations; ++i) {
      checkError(hipSetDevice(DeviceId));
      checkError(hipblasSgemm(Handle, HIPBLAS_OP_N, HIPBLAS_OP_N, Size, Size,
                              Size, &alpha, DeviceMatA, Size, DeviceMatB, Size,
                              &beta, DeviceMatC + i * Size * Size, Size),
                 "SGEMM");
    }
  }

  size_t verify() {
    int Errors;
    const int BlockSize = 16;

    checkError(hipSetDevice(DeviceId));
    checkError(hipMemsetD32Async(DeviceErrors, 0, 1, 0), "memset");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(compareKernel<float>), Size / BlockSize,
                       Size / BlockSize, 0, 0, DeviceMatC, DeviceErrors,
                       Iterations);

    checkError(hipMemcpyDtoHAsync(&Errors, DeviceErrors, sizeof(int), 0),
               "Read Errors");
    TotalErrors += Errors;

    return TotalErrors;
  }

  void finish() {
    checkError(hipSetDevice(DeviceId));
    checkError(hipDeviceSynchronize());
  }

private:
  int DeviceId;
  size_t Size;
  int Iterations;
  int *DeviceErrors;
  size_t TotalErrors = 0;

  float *MatA, *MatB;
  float *DeviceMatA, *DeviceMatB, *DeviceMatC;
  hipblasHandle_t Handle = nullptr;
};

void StressTest(int DeviceId, size_t RunTime, float *MatA, float *MatB,
                size_t Size) {
  Logger::Log(Logger::Debug, std::string("Thread id ") +
                                 std::to_string(DeviceId) + " : started\n");

  GpuGemmWorker Worker(DeviceId, MatA, MatB, Size);
  Logger::Log(Logger::Debug, std::string("Thread id ") +
                                 std::to_string(DeviceId) +
                                 " : initialization Done\n");

  // returns the current time in seconds
  time_t StartTime = time(0);
  size_t Errors = -1;

  while (true) {
    Worker.start();
    Errors = Worker.verify();

    time_t CurrTime = time(0);
    Logger::Log(Logger::Debug, std::string("Thread id ") +
                                   std::to_string(DeviceId) +
                                   " : remaining time " +
                                   std::to_string(CurrTime - StartTime) + "\n");
    if (CurrTime - StartTime > RunTime) {
      Worker.finish();
      break;
    }
  }

  Logger::Log(Logger::Debug,
              std::string("Thread id ") + std::to_string(DeviceId) +
                  " : finished with Errors = " + std::to_string(Errors) + "\n");
}

void launchTest(int Runtime = 120, int Size = 2048) {
  float *MatA = (float *)malloc(sizeof(float) * Size * Size);
  float *MatB = (float *)malloc(sizeof(float) * Size * Size);

  assert(MatA && MatB);

  srand(10);
  for (size_t i = 0; i < Size * Size; ++i) {
    MatA[i] = (float)((double)(rand() % 1000000) / 100000.0);
    MatB[i] = (float)((double)(rand() % 1000000) / 100000.0);
  }

  checkError(hipSuccess);
  checkError(HIPBLAS_STATUS_SUCCESS);
  int DeviceCount = GpuUtils::getDeviceCount();
  if (!DeviceCount) {
    exit(EXIT_FAILURE);
  }

  std::vector<std::thread> Threads(DeviceCount);

  for (int i = 0; i < DeviceCount; i++) {
    Threads[i] = std::thread(StressTest, i, Runtime, MatA, MatB, Size);
    int Processors = sysconf(_SC_NPROCESSORS_ONLN);
    cpu_set_t Mask;

    CPU_ZERO(&Mask);
    CPU_SET(i % Processors, &Mask);
    int rc = pthread_setaffinity_np(Threads[i].native_handle(),
                                    sizeof(cpu_set_t), &Mask);

    if (rc != 0) {
      Logger::Log(Logger::Error,
                  std::string("Error calling pthread_setaffinity_np: ") +
                      std::to_string(rc) + "\n");
    }

#ifndef PARALLEL_EXECUTION
    // At present parallel execution triggers oom killer
    if (Threads[i].joinable())
      Threads[i].join();
#endif
  }

  for (auto &T : Threads)
    if (T.joinable())
      T.join();

  free(MatA);
  free(MatB);

  Logger::Log(Logger::Debug, "Stress test completed\n");
}

int main(int argc, char *argv[]) { launchTest(); }

