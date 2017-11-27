#include "common.hpp"

#include "../src/opencl.hpp"
#include "../src/threshold.hpp"

#include "gtest/gtest.h"

cv::Vec3b test1_data[12] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
cv::Mat_<cv::Vec3b> test1(2, 2, test1_data);
cv::Mat_<uint8_t> expected1 = (cv::Mat_<uint8_t>(2, 2) << 0, 0, 0, 0),
                  expected2 = (cv::Mat_<uint8_t>(2, 2) << 255, 255, 255, 255),
                  expected3 = (cv::Mat_<uint8_t>(2, 2) << 0, 0, 255, 255);

TEST(Threshold, MonoThreaded) {
	EXPECT_TRUE(assert_mat_equal(threshold(test1, 40), expected1));
	EXPECT_TRUE(assert_mat_equal(threshold(test1, 0), expected2));
}

TEST(Threshold, MonoThreadedWithComponents) {
	EXPECT_TRUE(assert_mat_equal(threshold(test1, 2, LightnessComponent::intensity), expected3));
	EXPECT_TRUE(assert_mat_equal(threshold(test1, 2, LightnessComponent::lightness), expected3));
	EXPECT_TRUE(assert_mat_equal(threshold(test1, 2, LightnessComponent::luma), expected3));
	EXPECT_TRUE(assert_mat_equal(threshold(test1, 2, LightnessComponent::value), expected3));
}

TEST(Threshold, GnuParallel) {
	EXPECT_TRUE(
	        assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::intensity), expected3));
	EXPECT_TRUE(
	        assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::lightness), expected3));
	EXPECT_TRUE(assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::luma), expected3));
	EXPECT_TRUE(assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::value), expected3));
}

TEST(Threshold, OpenCL) {
	cl_singletons::setup();

	cv::Vec4b test2_data[12] = {{1, 1, 1, 1}, {2, 2, 2, 1}, {3, 3, 3, 1}, {4, 4, 4, 1}};
	cv::Mat_<cv::Vec4b> test2(2, 2, test2_data);

	EXPECT_TRUE(assert_mat_equal(threshold_cl(test2, 127).get(), expected1));
	EXPECT_TRUE(assert_mat_equal(threshold_cl(test2, 0).get(), expected2));
	EXPECT_TRUE(assert_mat_equal(threshold_cl(test2, 2).get(), expected3));
}

TEST(Threshold, DISABLED_Lena) {
	cv::Mat_<cv::Vec3b> lena = cv::imread("../data/lena.png", cv::IMREAD_COLOR);
	cv::Mat_<cv::Vec4b> lena4;
	cv::cvtColor(lena, lena4, CV_BGR2RGBA, 4);

	cv::imshow("Original", lena);
	cv::imshow("Lena (Single Threaded) Intensity", threshold(lena, 127, intensity));
	cv::imshow("Lena (Single Threaded) Value", threshold(lena, 127, value));
	cv::imshow("Lena (Single Threaded) Lightness", threshold(lena, 127, lightness));
	cv::imshow("Lena (Single Threaded) Luma", threshold(lena, 127, luma));
	cv::imshow("Lena (Single Threaded) Luma Rounded", threshold(lena, 127, luma_rounded));
	cv::imshow("Lena (OpenCV)", threshold_cv(lena, 127));
	cv::imshow("Lena (OpenCV + OpenCL)", threshold_cvcl(lena, 127));
	cv::imshow("Lena (OpenCL)", threshold_cl(lena4, 127).get());

	cv::waitKey(0);
}

TEST(Threshold, DISABLED_AllOpenCL) {
	const std::vector<std::string> images = {
	        "../data/airplane.png",
	        "../data/baboon.png",
	        "../data/cameraman.tif",
	        "../data/lena.png",
	        "../data/logo.tif",
	        "../data/logo_noise.tif",
	        "../data/peppers.png",
	};

	for(std::string const& path : images) {
		const cv::Mat_<cv::Vec3b> input = cv::imread(path, cv::IMREAD_COLOR);
		if(input.empty()) {
			std::cerr << "Could not load image at '" << path << "'; exiting..."
			          << std::endl;
			std::exit(-1);
		}

		cv::Mat_<cv::Vec4b> input4;
		cv::cvtColor(input, input4, CV_BGR2RGBA, 4);

		cv::imshow("Original", input);
		cv::imshow("Threshold (OpenCL)", threshold_cl(input4, 127).get());

		cv::waitKey(0);
	}
}
