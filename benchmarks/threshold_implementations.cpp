#include "common.hpp"

#include "../src/threshold.hpp"

#include <hayai.hpp>

#define THRESHOLD_MONOTHREADED_BENCH(fixture)                                                      \
	BENCHMARK_F(fixture, mono_threaded, 10, 100) {                                                 \
		threshold(image, 127, intensity);                                                          \
	}

#define THRESHOLD_GNU_PARALLEL_BENCH(fixture)                                                      \
	BENCHMARK_F(fixture, gnu_parallel, 10, 100) {                                                  \
		threshold_gnupar(image, 127, intensity);                                                   \
	}

#define THRESHOLD_OPENCL_BENCH(fixture)                                                            \
	BENCHMARK_F(fixture, opencl, 10, 100) {                                                        \
		threshold_cl(image4, 127);                                                                 \
	}

#define THRESHOLD_OPENCV_BENCH(fixture)                                                            \
	BENCHMARK_F(fixture, opencv, 10, 100) {                                                        \
		threshold_cv(image, 127);                                                                  \
	}

#define THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(fixture)                                          \
	THRESHOLD_MONOTHREADED_BENCH(fixture)                                                          \
	THRESHOLD_GNU_PARALLEL_BENCH(fixture)                                                          \
	THRESHOLD_OPENCL_BENCH(fixture)                                                                \
	THRESHOLD_OPENCV_BENCH(fixture)

// Random images
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(BlockRandomImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(EDTV480RandomImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(EDTV576RandomImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(HDTV720RandomImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(HDTV1080RandomImageFixture);

// Given images
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(AirplaneFixedImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(BaboonFixedImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(CameramanFixedImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(LenaFixedImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(LogoNoiseFixedImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(LogoFixedImageFixture);
THRESHOLD_ALL_IMPLEMENTATIONS_BENCHMARKS(PeppersFixedImageFixture);
