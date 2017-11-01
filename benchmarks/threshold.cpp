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
	THRESHOLD_BASE_BENCHMARK(fixture, luma)

THRESHOLD_ALL_COMPONENTS_BENCHMARKS(SmallRandomImageFixture)
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(MediumRandomImageFixture)
THRESHOLD_ALL_COMPONENTS_BENCHMARKS(BigRandomImageFixture)
