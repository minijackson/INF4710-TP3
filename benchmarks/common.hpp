#pragma once

#include "../src/opencl.hpp"

#include <hayai.hpp>
#include <opencv2/opencv.hpp>

#include <random>

#define CREATE_RANDOM_IMAGE_FIXTURE(ClassName, width, height)                                      \
	class ClassName : public ::hayai::Fixture {                                                    \
	public:                                                                                        \
		static constexpr size_t data_size = width * height;                                        \
                                                                                                   \
		cv::Mat_<cv::Vec3b> image;                                                                 \
		cv::Mat_<cv::Vec4b> image4;                                                                \
		cv::Vec3b image_data[data_size];                                                           \
                                                                                                   \
		virtual void SetUp() {                                                                     \
			cl_singletons::setup();                                                                \
                                                                                                   \
			std::random_device r;                                                                  \
			std::seed_seq seed{r(), r(), r(), r(), r(), r()};                                      \
			std::mt19937 rand_engine(seed);                                                        \
			std::uniform_int_distribution<uint8_t> distribution;                                   \
                                                                                                   \
			for(size_t i = 0; i < data_size; ++i) {                                                \
				image_data[i][0] = distribution(rand_engine);                                      \
				image_data[i][1] = distribution(rand_engine);                                      \
				image_data[i][2] = distribution(rand_engine);                                      \
			}                                                                                      \
                                                                                                   \
			image = cv::Mat_<cv::Vec3b>(width, height, image_data);                                \
			cv::cvtColor(image, image4, CV_BGR2RGBA, 4);                                           \
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
		cv::Mat image4;                                                                            \
                                                                                                   \
		virtual void SetUp() {                                                                     \
			cl_singletons::setup();                                                                \
                                                                                                   \
			image = cv::imread("../data/" #image_name, cv::IMREAD_COLOR);                          \
			cv::cvtColor(image, image4, CV_BGR2RGBA, 4);                                           \
		}                                                                                          \
	};

CREATE_FIXED_IMAGE_FIXTURE(AirplaneFixedImageFixture, airplane.png);
CREATE_FIXED_IMAGE_FIXTURE(BaboonFixedImageFixture, baboon.png);
CREATE_FIXED_IMAGE_FIXTURE(CameramanFixedImageFixture, cameraman.tif);
CREATE_FIXED_IMAGE_FIXTURE(LenaFixedImageFixture, lena.png);
CREATE_FIXED_IMAGE_FIXTURE(LogoNoiseFixedImageFixture, logo_noise.tif);
CREATE_FIXED_IMAGE_FIXTURE(LogoFixedImageFixture, logo.tif);
CREATE_FIXED_IMAGE_FIXTURE(PeppersFixedImageFixture, peppers.png);
