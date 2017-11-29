#include "common.hpp"

#include "../src/dilation.hpp"
#include "../src/threshold.hpp"

#include <hayai.hpp>

#define CREATE_THRESHED_IMAGE_FIXTURE(ParentClass)                                                 \
	class Threshed##ParentClass : public ParentClass {                                             \
	public:                                                                                        \
		cv::Mat_<uint8_t> threshed_image;                                                          \
                                                                                                   \
		virtual void SetUp() {                                                                     \
			ParentClass::SetUp();                                                                  \
			threshed_image = threshold_gnupar(image, 170);                                         \
		}                                                                                          \
	};

/*
CREATE_THRESHED_IMAGE_FIXTURE(BlockRandomImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(EDTV480RandomImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(EDTV576RandomImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(HDTV720RandomImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(HDTV1080RandomImageFixture);
*/

CREATE_THRESHED_IMAGE_FIXTURE(AirplaneFixedImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(BaboonFixedImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(CameramanFixedImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(LenaFixedImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(LogoNoiseFixedImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(LogoFixedImageFixture);
CREATE_THRESHED_IMAGE_FIXTURE(PeppersFixedImageFixture);

#define DILATION_BASE_BENCH(fixture)                                                               \
	BENCHMARK_F(fixture, mono_threaded, 10, 50) {                                                  \
		dilate(threshed_image, 10);                                                                \
	}

#define DILATION_OPENMP_BENCH(fixture)                                                             \
	BENCHMARK_F(fixture, openmp, 10, 50) {                                                         \
		dilate_omp(threshed_image, 10);                                                            \
	}

#define DILATION_OPENCV_BENCH(fixture)                                                             \
	BENCHMARK_F(fixture, opencv, 10, 50) {                                                         \
		dilate_cv(threshed_image, 10);                                                             \
	}

#define DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(fixture)                                           \
	DILATION_BASE_BENCH(Threshed##fixture)                                                         \
	DILATION_OPENMP_BENCH(Threshed##fixture)                                                       \
	DILATION_OPENCV_BENCH(Threshed##fixture)

/*
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(BlockRandomImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(EDTV480RandomImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(EDTV576RandomImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(HDTV720RandomImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(HDTV1080RandomImageFixture);
*/

DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(AirplaneFixedImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(BaboonFixedImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(CameramanFixedImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(LenaFixedImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(LogoNoiseFixedImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(LogoFixedImageFixture);
DILATION_ALL_IMPLEMENTATIONS_BENCHMARKS(PeppersFixedImageFixture);
