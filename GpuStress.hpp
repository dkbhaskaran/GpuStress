#ifndef _GPU_BURN_HPP_
#define _GPU_BURN_HPP_

#include <chrono>
#include <ctime>
#include <fstream>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <iostream>
#include <map>
#include <mutex>

class Logger {
public:
  enum LogLevel {
    Error,
    Debug,
  };

  Logger(std::string File) {
    Stream.open(File);
    if (Stream.fail()) {
      throw std::iostream::failure("Cannot open file: " + File);
    }
    Log(Debug, std::string("Logger Started\n"));
  }

  virtual ~Logger() { Stream.close(); }

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  void Write(const std::string &Message) {
    auto Now = std::chrono::system_clock::now();
    std::time_t Time = std::chrono::system_clock::to_time_t(Now);
    std::string TimeWithoutEndl = std::string(std::ctime(&Time)).substr(0, 24);

    std::lock_guard<std::mutex> lock(Mutex);
    Stream << TimeWithoutEndl << ": " << Message;
    Stream.flush();
  }

  static void Log(Logger::LogLevel Level, const std::string &Message) {
    Glogger.Write(Message);
  }

private:
  static Logger Glogger;
  std::ofstream Stream;
  std::mutex Mutex;
};

void checkError(hipError_t rCode, std::string desc = "") {
#define ADD_ERROR_STRING(_err)                                                 \
  g_errorStrings.insert(std::pair<hipError_t, std::string>(_err, #_err))

  static std::map<hipError_t, std::string> g_errorStrings;
  if (!g_errorStrings.size()) {
    ADD_ERROR_STRING(hipErrorInvalidValue);
    ADD_ERROR_STRING(hipErrorOutOfMemory);
    ADD_ERROR_STRING(hipErrorNotInitialized);
    ADD_ERROR_STRING(hipErrorDeinitialized);
    ADD_ERROR_STRING(hipErrorNoDevice);
    ADD_ERROR_STRING(hipErrorInvalidDevice);
    ADD_ERROR_STRING(hipErrorInvalidImage);
    ADD_ERROR_STRING(hipErrorInvalidContext);
    ADD_ERROR_STRING(hipErrorMapFailed);
    ADD_ERROR_STRING(hipErrorUnmapFailed);
    ADD_ERROR_STRING(hipErrorArrayIsMapped);
    ADD_ERROR_STRING(hipErrorAlreadyMapped);
    ADD_ERROR_STRING(hipErrorNoBinaryForGpu);
    ADD_ERROR_STRING(hipErrorAlreadyAcquired);
    ADD_ERROR_STRING(hipErrorNotMapped);
    ADD_ERROR_STRING(hipErrorNotMappedAsArray);
    ADD_ERROR_STRING(hipErrorNotMappedAsPointer);
    ADD_ERROR_STRING(hipErrorUnsupportedLimit);
    ADD_ERROR_STRING(hipErrorContextAlreadyInUse);
    ADD_ERROR_STRING(hipErrorInvalidSource);
    ADD_ERROR_STRING(hipErrorFileNotFound);
    ADD_ERROR_STRING(hipErrorSharedObjectSymbolNotFound);
    ADD_ERROR_STRING(hipErrorSharedObjectInitFailed);
    ADD_ERROR_STRING(hipErrorOperatingSystem);
    ADD_ERROR_STRING(hipErrorInvalidHandle);
    ADD_ERROR_STRING(hipErrorNotFound);
    ADD_ERROR_STRING(hipErrorNotReady);
    ADD_ERROR_STRING(hipErrorLaunchFailure);
    ADD_ERROR_STRING(hipErrorLaunchOutOfResources);
    ADD_ERROR_STRING(hipErrorLaunchTimeOut);
    ADD_ERROR_STRING(hipErrorSetOnActiveProcess);
    ADD_ERROR_STRING(hipErrorContextIsDestroyed);
    ADD_ERROR_STRING(hipErrorUnknown);
  }
#undef ADD_ERROR_STRING

  if (rCode != hipSuccess) {
    std::string Message =
        ((desc == "")
             ? std::string("Error: ")
             : (std::string("Error in \"") + desc + std::string("\": "))) +
        g_errorStrings[rCode];
    Logger::Log(Logger::Error, Message);
    exit(EXIT_FAILURE);
  }
}

void checkError(hipblasStatus_t rCode, std::string desc = "") {
#define ADD_ERROR_STRING(_err)                                                 \
  g_errorStrings.insert(std::pair<hipblasStatus_t, std::string>(_err, #_err))

  static std::map<hipblasStatus_t, std::string> g_errorStrings;
  if (!g_errorStrings.size()) {
    ADD_ERROR_STRING(HIPBLAS_STATUS_NOT_INITIALIZED);
    ADD_ERROR_STRING(HIPBLAS_STATUS_ALLOC_FAILED);
    ADD_ERROR_STRING(HIPBLAS_STATUS_INVALID_VALUE);
    ADD_ERROR_STRING(HIPBLAS_STATUS_ARCH_MISMATCH);
    ADD_ERROR_STRING(HIPBLAS_STATUS_MAPPING_ERROR);
    ADD_ERROR_STRING(HIPBLAS_STATUS_EXECUTION_FAILED);
    ADD_ERROR_STRING(HIPBLAS_STATUS_INTERNAL_ERROR);
  }
#undef ADD_ERROR_STRING

  if (rCode != HIPBLAS_STATUS_SUCCESS) {
    std::string Message =
        ((desc == "")
             ? std::string("Error: ")
             : (std::string("Error in \"") + desc + std::string("\": "))) +
        g_errorStrings[rCode];
    Logger::Log(Logger::Error, Message);
    exit(EXIT_FAILURE);
  }
}

#endif //_GPU_BURN_HPP_
