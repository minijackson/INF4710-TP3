#include "common.hpp"

#include "../src/threshold.hpp"

#include <hayai.hpp>

#define THRESHOLD_BASE_BENCHMARK(fixture, light_component)                                         \
	BENCHMARK_F(fixture, light_component, 10, 100) {                                               \
		threshold(image, 127, light_component);                                                    \
	}

#define THRESHOLD_ALL_COMPONENTS_BENCHMARKS(fixture)                                               \
	THRESHOLD_BASE_BENCHMARK(fixture, intensity)                                                   \
	THRESHOLD_BASE_BENCHMARK(fixture, value)                                                       \
	THRESHOLD_BASE_BENCHMARK(fixture, lightness)                                                   \
	THRESHOLD_BASE_BENCHMARK(fixture, luma)                                                        \
	THRESHOLD_BASE_BENCHMARK(fixture, luma_rounded)

// Random images
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(BlockRandomImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(EDTV480RandomImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(EDTV576RandomImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(HDTV720RandomImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(HDTV1080RandomImageFixture);

// Given images
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(AirplaneFixedImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(BaboonFixedImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(CameramanFixedImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(LenaFixedImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(LogoNoiseFixedImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(LogoFixedImageFixture);
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(PeppersFixedImageFixture);
