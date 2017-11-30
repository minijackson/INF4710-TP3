#include <gtest/gtest.h>

#include "common.hpp"

#include "../src/dilation.hpp"
#include "../src/threshold.hpp"

cv::Mat_<uint8_t> test1 = (cv::Mat_<uint8_t>(5, 5) <<
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0),
                  test2 = (cv::Mat_<uint8_t>(5, 5) <<
						  0, 0, 0,   0, 0,
						  0, 0, 0,   0, 0,
						  0, 0, 255, 0, 0,
						  0, 0, 0,   0, 0,
						  0, 0, 0,   0, 0),
                  test3 = (cv::Mat_<uint8_t>(5, 5) <<
						  0, 0,   0, 0, 0,
						  0, 255, 0, 0, 0,
						  0, 0,   0, 0, 0,
						  0, 0,   0, 0, 0,
						  0, 0,   0, 0, 0),

                  test4 = (cv::Mat_<uint8_t>(5, 5) <<
						  0, 0,   0,   0,   0,
						  0, 255, 255, 255, 0,
						  0, 255, 255, 255, 0,
						  0, 255, 255, 255, 0,
						  0, 0,   0,   0,   0),
                  test5 = (cv::Mat_<uint8_t>(5, 5) <<
						  0, 0,   0,   0, 0,
						  0, 255, 255, 0, 0,
						  0, 255, 255, 0, 0,
						  0, 0,   0,   0, 0,
						  0, 0,   0,   0, 0),
                  test6 = (cv::Mat_<uint8_t>(5, 5) <<
						  0, 0,   0,   0, 0,
						  0, 255, 0,   0, 0,
						  0, 0,   255, 0, 0,
						  0, 0,   0,   0, 0,
						  0, 0,   0,   0, 0),
                  test7 = (cv::Mat_<uint8_t>(5, 5) <<
						  255, 255, 255, 255, 255,
						  255, 255, 255, 255, 255,
						  255, 255, 255, 255, 255,
						  255, 255, 255, 255, 255,
						  255, 255, 255, 255, 255),
                  test8 = (cv::Mat_<uint8_t>(5, 5) <<
						  255, 255, 255, 0, 0,
						  255, 255, 255, 0, 0,
						  255, 255, 255, 0, 0,
						  0,   0,   0,   0, 0,
						  0,   0,   0,   0, 0),
                  test9 = (cv::Mat_<uint8_t>(5, 5) <<
						  255, 255, 255, 255, 0,
						  255, 255, 255, 255, 0,
						  255, 255, 255, 255, 0,
						  255, 255, 255, 255, 0,
						  0,   0,   0,   0,   0);

TEST(Dilation, MonoThreaded) {
	EXPECT_TRUE(assert_mat_equal(dilate(test1, 1), test1));

	EXPECT_TRUE(assert_mat_equal(dilate(test2, 0), test2));
	EXPECT_TRUE(assert_mat_equal(dilate(test2, 1), test4));
	EXPECT_TRUE(assert_mat_equal(dilate(test2, 2), test2));
	EXPECT_TRUE(assert_mat_equal(dilate(test2, 1337), test2));

	EXPECT_TRUE(assert_mat_equal(dilate(test3, 0), test3));
	EXPECT_TRUE(assert_mat_equal(dilate(test3, 1), test5));
	EXPECT_TRUE(assert_mat_equal(dilate(test3, 2), test6));
	EXPECT_TRUE(assert_mat_equal(dilate(test3, 1337), test3));
}

TEST(Dilation, OpenMP) {
	EXPECT_TRUE(assert_mat_equal(dilate_omp(test1, 1), test1));

	EXPECT_TRUE(assert_mat_equal(dilate_omp(test2, 0), test2));
	EXPECT_TRUE(assert_mat_equal(dilate_omp(test2, 1), test4));
	EXPECT_TRUE(assert_mat_equal(dilate_omp(test2, 2), test2));
	EXPECT_TRUE(assert_mat_equal(dilate_omp(test2, 1337), test2));

	EXPECT_TRUE(assert_mat_equal(dilate_omp(test3, 0), test3));
	EXPECT_TRUE(assert_mat_equal(dilate_omp(test3, 1), test5));
	EXPECT_TRUE(assert_mat_equal(dilate_omp(test3, 2), test6));
	EXPECT_TRUE(assert_mat_equal(dilate_omp(test3, 1337), test3));
}

TEST(Dilation, OpenCL) {
	cl_singletons::setup();

	EXPECT_TRUE(assert_mat_equal(dilate_cl(test1, 1).get(), test1));

	EXPECT_TRUE(assert_mat_equal(dilate_cl(test2, 0).get(), test2));
	EXPECT_TRUE(assert_mat_equal(dilate_cl(test2, 1).get(), test4));
	EXPECT_TRUE(assert_mat_equal(dilate_cl(test2, 2).get(), test7));
	EXPECT_TRUE(assert_mat_equal(dilate_cl(test2, 1337).get(), test7));

	EXPECT_TRUE(assert_mat_equal(dilate_cl(test3, 0).get(), test3));
	EXPECT_TRUE(assert_mat_equal(dilate_cl(test3, 1).get(), test8));
	EXPECT_TRUE(assert_mat_equal(dilate_cl(test3, 2).get(), test9));
	EXPECT_TRUE(assert_mat_equal(dilate_cl(test3, 1337).get(), test7));
}

TEST(Dilation, DISABLED_AllImplementations) {
	cl_singletons::setup();

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
			std::cerr << "Could not load image at '" << path << "'; exiting..." << std::endl;
			std::exit(-1);
		}

		auto threshed = threshold_gnupar(input, 180);

		cv::imshow("Original", input);
		cv::imshow("Threshold (GNU Parallel)", threshed);
		cv::imshow("Dilation (Naive)", dilate(threshed, 5));
		cv::imshow("Dilation (OpenMP)", dilate_omp(threshed, 5));
		cv::imshow("Dilation (OpenCL)", dilate_cl(threshed, 5).get());
		cv::imshow("Dilation (OpenCV)", dilate_cv(threshed, 5));

		cv::waitKey(0);
	}
}
