#pragma once

#include <hayai.hpp>
#include <opencv2/opencv.hpp>

#include <random>

#define CREATE_RANDOM_IMAGE_FIXTURE(ClassName, width, height)                                      \
	class ClassName : public ::hayai::Fixture {                                                    \
	public:                                                                                        \
		static constexpr std::pair<size_t, size_t> image_size = {width, height};                   \
		static constexpr size_t data_size = image_size.first * image_size.second * 3;              \
		cv::Mat image;                                                                             \
		uint8_t image_data[data_size];                                                             \
		virtual void SetUp() {                                                                     \
			std::random_device r;                                                                  \
			std::seed_seq seed{r(), r(), r(), r(), r(), r()};                                      \
			std::mt19937 rand_engine(seed);                                                        \
			std::uniform_int_distribution<uint8_t> distribution;                                   \
			for(size_t i = 0; i < data_size; ++i) {                                                \
				image_data[i] = distribution(rand_engine);                                         \
			}                                                                                      \
			image = cv::Mat(image_size.first, image_size.second, CV_8UC3, image_data);             \
		}                                                                                          \
	};

CREATE_RANDOM_IMAGE_FIXTURE(SmallRandomImageFixture, 3, 3);
CREATE_RANDOM_IMAGE_FIXTURE(MediumRandomImageFixture, 800, 600);
CREATE_RANDOM_IMAGE_FIXTURE(BigRandomImageFixture, 1920, 1080);
