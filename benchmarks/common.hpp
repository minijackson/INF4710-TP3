#pragma once

#include <hayai.hpp>
#include <opencv2/opencv.hpp>

#include <random>

#define CREATE_RANDOM_IMAGE_FIXTURE(ClassName, width, height)                                      \
	class ClassName : public ::hayai::Fixture {                                                    \
	public:                                                                                        \
		static constexpr std::pair<size_t, size_t> image_size = {width, height};                   \
		static constexpr size_t data_size = image_size.first * image_size.second * 3;              \
                                                                                                   \
		cv::Mat image;                                                                             \
		uint8_t image_data[data_size];                                                             \
                                                                                                   \
		virtual void SetUp() {                                                                     \
			std::random_device r;                                                                  \
			std::seed_seq seed{r(), r(), r(), r(), r(), r()};                                      \
			std::mt19937 rand_engine(seed);                                                        \
			std::uniform_int_distribution<uint8_t> distribution;                                   \
                                                                                                   \
			for(size_t i = 0; i < data_size; ++i) {                                                \
				image_data[i] = distribution(rand_engine);                                         \
			}                                                                                      \
                                                                                                   \
			image = cv::Mat(image_size.first, image_size.second, CV_8UC3, image_data);             \
		}                                                                                          \
	};

CREATE_RANDOM_IMAGE_FIXTURE(BlockRandomImageFixture, 3, 3);
CREATE_RANDOM_IMAGE_FIXTURE(EDTV480RandomImageFixture, 720, 480);
CREATE_RANDOM_IMAGE_FIXTURE(EDTV576RandomImageFixture, 720, 576);
CREATE_RANDOM_IMAGE_FIXTURE(HDTV720RandomImageFixture, 1280, 720);
CREATE_RANDOM_IMAGE_FIXTURE(HDTV1080RandomImageFixture, 1920, 1080);

#define CREATE_FIXED_IMAGE_FIXTURE(ClassName, image_name)                                          \
	class ClassName : public ::hayai::Fixture {                                                    \
	public:                                                                                        \
		cv::Mat image;                                                                             \
                                                                                                   \
		virtual void SetUp() {                                                                     \
			image = cv::imread("../data/" #image_name, cv::IMREAD_COLOR);                          \
		}                                                                                          \
	};

CREATE_FIXED_IMAGE_FIXTURE(AirplaneFixedImageFixture, airplane.png);
CREATE_FIXED_IMAGE_FIXTURE(BaboonFixedImageFixture, baboon.png);
CREATE_FIXED_IMAGE_FIXTURE(CameramanFixedImageFixture, cameraman.tif);
CREATE_FIXED_IMAGE_FIXTURE(LenaFixedImageFixture, lena.png);
CREATE_FIXED_IMAGE_FIXTURE(LogoNoiseFixedImageFixture, logo_noise.tif);
CREATE_FIXED_IMAGE_FIXTURE(LogoFixedImageFixture, logo.tif);
CREATE_FIXED_IMAGE_FIXTURE(PeppersFixedImageFixture, peppers.png);
