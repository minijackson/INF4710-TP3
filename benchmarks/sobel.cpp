#include "common.hpp"

#include "../src/sobel.hpp"

#include <hayai.hpp>

#define SOBEL_BASE_BENCHMARK(fixture)                                                              \
	BENCHMARK_F(fixture, mono_threaded, 10, 100) {                                                 \
		sobel(image);                                                                              \
	}

#define SOBEL_OPENCL_BENCHMARK(fixture)                                                            \
	BENCHMARK_F(fixture, opencl, 10, 100) {                                                        \
		sobel_cl(image4);                                                                          \
	}

#define SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(fixture)                                              \
	SOBEL_BASE_BENCHMARK(fixture)                                                                  \
	SOBEL_OPENCL_BENCHMARK(fixture)

SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(EDTV480RandomImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(EDTV576RandomImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(HDTV720RandomImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(HDTV1080RandomImageFixture);

SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(AirplaneFixedImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(BaboonFixedImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(CameramanFixedImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(LenaFixedImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(LogoNoiseFixedImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(LogoFixedImageFixture);
SOBEL_ALL_IMPLEMENTATIONS_BENCHMARKS(PeppersFixedImageFixture);
