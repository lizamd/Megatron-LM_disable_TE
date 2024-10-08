
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>

#define VERSION_STRING "3.4"

#include "Stream.h"

#if defined(CUDA)
#include "CUDAStream.h"
#elif defined(HIP)
#include "HIPStream.h"
#elif defined(HC)
#include "HCStream.h"
#elif defined(OCL)
#include "OCLStream.h"
#elif defined(USE_RAJA)
#include "RAJAStream.hpp"
#elif defined(KOKKOS)
#include "KokkosStream.hpp"
#elif defined(ACC)
#include "ACCStream.h"
#elif defined(SYCL)
#include "SYCLStream.h"
#elif defined(OMP)
#include "OMPStream.h"
#endif
#include "ValidationStream.hpp"

// Default size of 2^25
const int num_tests = 7;
unsigned int ARRAY_SIZE = 33554432;
unsigned int num_times = 100;
unsigned int deviceIndex = 0;
bool error_per_iter = false;
bool use_float = false;
bool event_timing = false;
bool read_only = false;
bool write_only = false;
bool copy_only = false;
bool mul_only = false;
bool add_only = false;
bool triad_only = false;
bool dot_only = false;
bool output_as_csv = false;
bool mibibytes = false;
bool ew_err = false;
bool display_std = false;
std::string csv_separator = ",";

template <typename T>
bool check_solution(const std::vector<bool> &runmap, const unsigned int ntimes,
                    std::vector<T> &a, std::vector<T> &b, std::vector<T> &c,
                    T &sum);

template <typename T>
void run(const std::vector<bool>& runmap);

void parseArguments(int argc, char *argv[]);

template <typename T>
void print_header() {
  std::streamsize ss = std::cout.precision();
  if (!output_as_csv)
  {
    std::cout << "Running kernels " << num_times << " times" << std::endl;

    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;


    if (mibibytes)
    {
      // MiB = 2^20
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0) << " GiB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0) << " GiB)" << std::endl;
    }
    else
    {
      // MB = 10^6
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
    }
    std::cout.precision(ss);
  }
}


int main(int argc, char *argv[])
{

  parseArguments(argc, argv);

  if (!output_as_csv)
  {
    std::cout
      << "BabelStream" << std::endl
      << "Version: " << VERSION_STRING << std::endl
      << "Implementation: " << IMPLEMENTATION_STRING << std::endl;
  }

  if (use_float)
    print_header<float>();
  else
    print_header<double>();
  bool run_all = !(read_only || write_only || copy_only || mul_only || add_only || triad_only || dot_only);

  if (error_per_iter && run_all) {
    throw std::runtime_error("Error detection per iteration is not currently compatible with "
                             "running all kernels, use (e.g.,) --dot-only");
  }

  std::vector<bool> runmap = {
      read_only || run_all, write_only || run_all, copy_only || run_all,
      mul_only || run_all, add_only || run_all, triad_only || run_all,
      dot_only || run_all,
  };

  if (use_float)
    run<float>(runmap);
  else
    run<double>(runmap);
}

static double calculate_time_s(const bool evt_timing,
        const std::chrono::high_resolution_clock::time_point t1,
        const std::chrono::high_resolution_clock::time_point t2,
        const double kernel_time)
{
#if defined(HIP) || defined(OCL) || defined(CUDA) || defined(HC)
  if (!evt_timing)
    return std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  else
    return kernel_time/1000.;
#else
  return std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
#endif
}

template <typename T>
void run(const std::vector<bool>& runmap)
{

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  // Result of the Dot kernel
  T sum;

  Stream<T> *stream;

#if defined(CUDA)
  // Use the CUDA implementation
  stream = new CUDAStream<T>(ARRAY_SIZE, event_timing, deviceIndex);

#elif defined(HIP)
  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, event_timing, deviceIndex);

#elif defined(HC)
  // Use the HC implementation
  stream = new HCStream<T>(ARRAY_SIZE, event_timing, deviceIndex);

#elif defined(OCL)
  // Use the OpenCL implementation
  stream = new OCLStream<T>(ARRAY_SIZE, event_timing, deviceIndex);

#elif defined(USE_RAJA)
  // Use the RAJA implementation
  stream = new RAJAStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(KOKKOS)
  // Use the Kokkos implementation
  stream = new KokkosStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(ACC)
  // Use the OpenACC implementation
  stream = new ACCStream<T>(ARRAY_SIZE, a.data(), b.data(), c.data(), deviceIndex);

#elif defined(SYCL)
  // Use the SYCL implementation
  stream = new SYCLStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(OMP)
  // Use the OpenMP implementation
  stream = new OMPStream<T>(ARRAY_SIZE, a.data(), b.data(), c.data(), deviceIndex);
#endif

  stream->init_arrays(startA, startB, startC);

  // List of times
  std::vector<std::vector<double>> timings(num_tests);
  double kernel_time;

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;
#define time_and_run(fn)                                                       \
  do {                                                                         \
    t1 = std::chrono::high_resolution_clock::now();                            \
    kernel_time = fn();                                                        \
    t2 = std::chrono::high_resolution_clock::now();                            \
  } while (0);

  // Main loop
  for (unsigned int kernel = 0; kernel < num_tests; ++kernel) {
    if (!runmap[kernel])
      continue;
    for (unsigned int k = 0; k < num_times; k++) {
      switch(kernel) {
        case 0:
          // Execute Read-only
          time_and_run(stream->read);
          break;
        case 1:
          // Execute Write-only
          time_and_run(stream->write);
          break;
        case 2:
          // Execute Copy
          time_and_run(stream->copy);
          break;
        case 3:
          // Execute Mul
          time_and_run(stream->mul);
          break;
        case 4:
          // Execute Add
          time_and_run(stream->add);
          break;
        case 5:
          // Execute Triad
          time_and_run(stream->triad);
          break;
        case 6:
          // Execute Dot
          time_and_run(stream->dot);
          break;
        default:
          throw std::runtime_error("Unknown kernel type!");
      }
      timings[kernel].push_back(calculate_time_s(event_timing && kernel != 6, t1, t2, kernel_time));
      if (kernel == 6) {
        sum = kernel_time;
      }

      if (error_per_iter) {
        std::cout << "Checking solution for iteration " << k << std::endl;
        // Check solutions
        stream->read_arrays(a, b, c);
        if (!check_solution<T>(runmap, num_times, a, b, c, sum)) {
          std::cerr << "Iteration " << k << " failed, exiting..." << std::endl;
          exit(EXIT_FAILURE);
        }

      }
    }
  }

  // Check solutions
  if (!error_per_iter) {
    stream->read_arrays(a, b, c);
    check_solution<T>(runmap, num_times, a, b, c, sum);
  }

  // Display timing results
  if (output_as_csv)
  {
    std::cout
      << "function" << csv_separator
      << "num_times" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << ((mibibytes) ? "max_mibytes_per_sec" : "max_mbytes_per_sec") << csv_separator
      << "min_runtime" << csv_separator
      << "max_runtime" << csv_separator
      << "avg_runtime"
      << ((display_std) ? csv_separator : "")
      << ((display_std) ? "std_runtime" : "")
      << std::endl;
  }
  else
  {
    std::cout
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << ((mibibytes) ? "MiB/s" : "MB/s")
      << std::left << std::setw(12) << (std::string("Min (") + ((mibibytes) ? "MiB/s" : "MB/s") + ")")
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average" << std::fixed;
    if (display_std)
      std::cout << std::left << std::setw(12) << "Std. Dev.";
    std::cout
      << std::endl
      << std::fixed;
  }

  std::string labels[7] = {"Read", "Write", "Copy", "Mul", "Add", "Triad", "Dot"};
  size_t sizes[7] = {
    sizeof(T) * ARRAY_SIZE,
    sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE
  };

  for (int i = 0; i < num_tests; i++)
  {
    if (!runmap[i])
      continue;
    // the size of the array
    size_t size = ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i];

    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

    // Calculate std. dev; ignore the first result
    double stddev = 0;
    if (display_std)
    {
      const double average_bw = size / average;
      std::for_each(timings[i].begin()+1, timings[i].end(),
                    [&stddev, &average_bw, &size](const double& x){
        const double bw = size / x;
        stddev += (bw-average_bw) * (bw-average_bw);
      });
      stddev = std::sqrt(stddev / (timings[i].size()-2));
    }

    // Display results
    if (output_as_csv)
    {
      std::cout
        << labels[i] << csv_separator
        << num_times << csv_separator
        << ARRAY_SIZE << csv_separator
        << sizeof(T) << csv_separator
        << size / (*minmax.first) << csv_separator
        << size / (*minmax.second) << csv_separator
        << size / (*minmax.first) << csv_separator
        << size / average;
      if (display_std)
      {
        std::cout << csv_separator << stddev;
      }
      std::cout << std::endl;
    }
    else
    {
      std::cout
        << std::left << std::setw(12) << labels[i]
        << std::left << std::setw(12) << std::setprecision(3) << size / (*minmax.first)
        << std::left << std::setw(12) << std::setprecision(3) << size / (*minmax.second)
        << std::left << std::setw(12) << std::setprecision(3) << size / (*minmax.first)
        << std::left << std::setw(12) << std::setprecision(3) << size / average;
      if (display_std)
      {
        std::cout << std::left << std::setw(12) << std::setprecision(3) << stddev;
      }
      std::cout << std::endl;
    }
  }

  delete stream;

}

template <typename T>
bool check_solution(const std::vector<bool> &runmap, const unsigned int ntimes,
                    std::vector<T> &a, std::vector<T> &b, std::vector<T> &c,
                    T &sum) {

  ValidationStream<T> stream(ARRAY_SIZE);
  stream.init_arrays(startA, startB, startC);

  T goldSum;

  for (int kernel = 0; kernel < num_tests; ++kernel) {
    if (!runmap[kernel])
      continue;
    for (int i = 0; i < ntimes; ++i) {
      switch(kernel) {
        case 0:
          // Execute Read-only
          stream.read();
          break;
        case 1:
          // Execute Write-only
          stream.write();
          break;
        case 2:
          // Execute Copy
          stream.copy();
          break;
        case 3:
          // Execute Mul
          stream.mul();
          break;
        case 4:
          // Execute Add
          stream.add();
          break;
        case 5:
          // Execute Triad
          stream.triad();
          break;
        case 6:
          // Execute Dot
          goldSum = stream.dot();
          break;
        default:
          throw std::runtime_error("Unknown kernel type!");
      }
    }
  }

  std::vector<T> goldA(1);
  std::vector<T> goldB(1);
  std::vector<T> goldC(1);
  stream.read_arrays(goldA, goldB, goldC);
  double epsi = std::numeric_limits<T>::epsilon() * 100.0;

  if (ew_err) {
    std::cout << std::endl;
    std::cout << "Element-wise validation:" << std::endl;
    std::cout << std::left << std::setw(14) << "i" << std::left
                           << std::setw(14) << "|err(a)[i]|" << std::left
                           << std::setw(14) << "|err(b)[i]|" << std::left
                           << std::setw(14) << "|err(c)[i]|" << std::endl;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
      double err = std::max(std::max(std::abs(a[i] - goldA[0]), std::abs(b[i] - goldB[0])), std::abs(c[i] - goldC[0]));
      if (err > epsi) {
        std::cout << std::left << std::setw(14) << std::setprecision(1) << i
        << std::left << std::setw(14) << std::setprecision(3) << std::abs(a[i] - goldA[0])
        << std::left << std::setw(14) << std::setprecision(3) << std::abs(b[i] - goldB[0])
        << std::left << std::setw(14) << std::setprecision(3) << std::abs(c[i] - goldC[0]) << std::endl;
      }
    }
    std::cout << std::endl;
  }

  // Calculate the average error
  double errA = std::accumulate(a.begin(), a.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldA[0]); });
  errA /= a.size();
  double errB = std::accumulate(b.begin(), b.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldB[0]); });
  errB /= b.size();
  double errC = std::accumulate(c.begin(), c.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldC[0]); });
  errC /= c.size();
  double errSum = fabs(sum - goldSum) / fabs(goldSum);

  bool passed = true;

  if (errA > epsi) {
    std::cerr
      << "Validation failed on a[]. Average error " << errA
      << std::endl;
      passed = false;
  }
  if (errB > epsi) {
    std::cerr
      << "Validation failed on b[]. Average error " << errB
      << std::endl;
      passed = false;
  }
  if (errC > epsi) {
    std::cerr
      << "Validation failed on c[]. Average error " << errC
      << std::endl;
      passed = false;
  }
  // Check sum to 8 decimal places
  if (runmap[6] && errSum > 1.0E-8) {
    std::cerr
      << "Validation failed on sum. Error " << errSum
      << std::endl << std::setprecision(15)
      << "Sum was " << sum << " but should be " << goldSum
      << std::endl;
    passed = false;
  }
  return passed;
}

int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--list").compare(argv[i]))
    {
      listDevices();
      exit(EXIT_SUCCESS);
    }
    else if (!std::string("--device").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
      {
        std::cerr << "Invalid device index." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--arraysize").compare(argv[i]) ||
             !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &ARRAY_SIZE))
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
             !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--float").compare(argv[i]))
    {
      use_float = true;
    }
    else if (!std::string("--read-only").compare(argv[i]))
    {
      read_only = true;
    }
    else if (!std::string("--write-only").compare(argv[i]))
    {
      write_only = true;
    }
    else if (!std::string("--copy-only").compare(argv[i]))
    {
      copy_only = true;
    }
    else if (!std::string("--mul-only").compare(argv[i]))
    {
      mul_only = true;
    }
    else if (!std::string("--add-only").compare(argv[i]))
    {
      add_only = true;
    }
    else if (!std::string("--triad-only").compare(argv[i]))
    {
      triad_only = true;
    }
    else if (!std::string("--dot-only").compare(argv[i]))
    {
      dot_only = true;
    }
#if defined(HIP) || defined(CUDA) || defined(OCL) || defined(HC)
    else if (!std::string("--event-timing").compare(argv[i]) ||
             !std::string("-e").compare(argv[i]))
    {
      event_timing = true;
    }
#endif
    else if (!std::string("--std").compare(argv[i]))
    {
        display_std = true;
    }
    else if (!std::string("--csv").compare(argv[i]))
    {
      output_as_csv = true;
    }
    else if (!std::string("--mibibytes").compare(argv[i]))
    {
      mibibytes = true;
    }
    else if (!std::string("--elementwise-error").compare(argv[i]))
    {
      ew_err = true;
    }
    else if (!std::string("--error-per-iter").compare(argv[i])) {
      error_per_iter = true;
    }
    else if (!std::string("--help").compare(argv[i]) ||
             !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "      --list               List available devices" << std::endl;
      std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << "      --float              Use floats (rather than doubles)" << std::endl;
      std::cout << "      --read-only          Only run read" << std::endl;
      std::cout << "      --write-only         Only run write" << std::endl;
      std::cout << "      --copy-only          Only run copy" << std::endl;
      std::cout << "      --mul-only           Only run mul" << std::endl;
      std::cout << "      --add-only           Only run add" << std::endl;
      std::cout << "      --triad-only         Only run triad" << std::endl;
      std::cout << "      --dot-only           Only run dot" << std::endl;
#if defined(HIP) || defined(CUDA) || defined(OCL) || defined(HC)
      std::cout << "  -e  --event-timing       Use event timing instead of host-side timing" << std::endl;
#endif
      std::cout << "      --std                Display the standard deviation" << std::endl;
      std::cout << "      --csv                Output as csv table" << std::endl;
      std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << "      --elementwise-error  Display the element-wise error of the computed values" << std::endl;
      std::cout << "      --error_per_iter     Validate the results after each kernel iteration" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
