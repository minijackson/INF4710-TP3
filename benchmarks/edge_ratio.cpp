#include "common.hpp"

#include "../src/edge_ratio.hpp"

#include "../src/dilation.hpp"
#include "../src/threshold.hpp"

#include <hayai.hpp>

#define CREATE_SUCC_IMAGE_FIXTURE(ClassName, frame_id, next_frame_id)                              \
	class ClassName : public ::hayai::Fixture {                                                    \
	public:                                                                                        \
		cv::Mat_<uint8_t> dilated_frame;                                                           \
		cv::Mat_<uint8_t> threshed_next_frame;                                                     \
                                                                                                   \
		virtual void SetUp() {                                                                     \
			cl_singletons::setup();                                                                \
                                                                                                   \
			auto frame = cv::imread("../data/mpv-shot000" #frame_id ".jpg", cv::IMREAD_COLOR);     \
			auto threshed_frame = threshold_gnupar(frame, 170);                                    \
			dilated_frame       = dilate_omp(threshed_frame, 10);                                  \
                                                                                                   \
			auto next_frame =                                                                      \
			        cv::imread("../data/mpv-shot000" #next_frame_id ".jpg", cv::IMREAD_COLOR);     \
			threshed_next_frame = threshold_gnupar(next_frame, 170);                               \
		}                                                                                          \
	};

CREATE_SUCC_IMAGE_FIXTURE(RoadFixture, 1, 2);
CREATE_SUCC_IMAGE_FIXTURE(WaterFixture, 3, 4);
CREATE_SUCC_IMAGE_FIXTURE(BicycleFixture, 5, 6);

#define EDGE_RATIO_MONOTHREADED_BENCH(fixture)                                                     \
	BENCHMARK_F(fixture, mono_threaded, 10, 500) {                                                 \
		edge_ratio(dilated_frame, threshed_next_frame);                                            \
	}

#define EDGE_RATIO_OPENMP_BENCH(fixture)                                                           \
	BENCHMARK_F(fixture, gnu_parallel, 10, 500) {                                                  \
		edge_ratio_omp(dilated_frame, threshed_next_frame);                                        \
	}

#define EDGE_RATIO_ALL_IMPLEMENTATIONS_BENCHMARKS(fixture)                                         \
	EDGE_RATIO_MONOTHREADED_BENCH(fixture)                                                         \
	EDGE_RATIO_OPENMP_BENCH(fixture)

EDGE_RATIO_ALL_IMPLEMENTATIONS_BENCHMARKS(RoadFixture);
EDGE_RATIO_ALL_IMPLEMENTATIONS_BENCHMARKS(WaterFixture);
EDGE_RATIO_ALL_IMPLEMENTATIONS_BENCHMARKS(BicycleFixture);
