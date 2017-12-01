#include "common.hpp"

#include "gtest/gtest.h"

#include "../src/sobel.hpp"

cv::Mat_<cv::Vec3b> expectedResult =  (cv::Mat_<cv::Vec3b>(5,5) <<	cv::Vec3b{0, 0, 0}, cv::Vec3b{0 , 0 , 0 }, cv::Vec3b{0 , 0 , 0 }, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0},
																	cv::Vec3b{0, 0, 0}, cv::Vec3b{40, 40, 40}, cv::Vec3b{40, 40, 40}, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0},
																	cv::Vec3b{0, 0, 0}, cv::Vec3b{40, 40, 40}, cv::Vec3b{40, 40, 40}, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0},
																	cv::Vec3b{0, 0, 0}, cv::Vec3b{40, 40, 40}, cv::Vec3b{40, 40, 40}, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0},
																	cv::Vec3b{0, 0, 0}, cv::Vec3b{0 , 0 , 0 }, cv::Vec3b{0 , 0 , 0 }, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0});

cv::Mat_<int8_t> test = (cv::Mat_<int8_t>(5, 5) <<	0, 0, 10, 10, 10,
													0, 0, 10, 10, 10,
													0, 0, 10, 10, 10,
													0, 0, 10, 10, 10,
													0, 0, 10, 10, 10);

cv::Mat_<int8_t> testbis = (cv::Mat_<int8_t>(5, 5) <<	1, 1, 1, 1, 1,
														1, 1, 1, 1, 1,
														1, 1, 1, 1, 1,
														1, 1, 1, 1, 1,
														1, 1, 1, 1, 1);

cv::Mat_<int8_t> gx = (cv::Mat_<int8_t>(5, 5) <<	0, 30, 30, 0, -30,
													0, 40, 40, 0, -40,
													0, 40, 40, 0, -40,
													0, 40, 40, 0, -40,
													0, 30, 30, 0, -30);

cv::Mat_<int8_t> gy = (cv::Mat_<int8_t>(5, 5) <<	0, -10, -30, -40, -30,
													0,	 0,	  0,   0,   0,
													0,   0,   0,   0,   0,
													0,   0,   0,   0,   0,
													0, -10, -30, -40, -30);

TEST(Sobel, MonoThreaded) {

	cv::Mat_<int8_t> zeroRow = (cv::Mat_<int8_t>(1,7) << 0, 0, 0, 0, 0, 0, 0);
	cv::Mat_<int8_t> zeroCol = (cv::Mat_<int8_t>(5,1) << 0, 0, 0, 0, 0);

	cv::Mat_<cv::Vec3b> testIMG(5, 5);

	for(int row = 0; row < 5; ++row)
	{
		for(int col = 0; col < 5; ++col)
		{
			testIMG.at<cv::Vec3b>(row, col)[0] = test(row, col);
			testIMG.at<cv::Vec3b>(row, col)[1] = test(row, col);
			testIMG.at<cv::Vec3b>(row, col)[2] = test(row, col);
		}
	}

	cv::Mat_<cv::Vec3b> myResult = sobel(testIMG);

	EXPECT_TRUE(assert_mat_equal(myResult, expectedResult));
}

TEST(Sobel, DISABLED_AllImplementations) {
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

		cv::Mat_<cv::Vec4b> input4;
		cv::cvtColor(input, input4, CV_RGB2RGBA, 4);

		cv::imshow("Original", input);
		cv::imshow("Sobel (Naive)",  sobel(input));
		cv::imshow("Sobel (Naive Fast)",  sobel(input));
		//cv::imshow("Sobel (OpenMP)", sobel_omp(input));
		cv::imshow("Sobel (OpenCL)", sobel_cl(input4).get());
		//cv::imshow("Sobel (OpenCV)", sobel_cv(input));

		cv::waitKey(0);
	}
}
