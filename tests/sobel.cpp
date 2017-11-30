#include "gtest/gtest.h"

#include "../src/sobel.hpp"

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
		//cv::imshow("Sobel (Naive)",  sobel(input));
		//cv::imshow("Sobel (OpenMP)", sobel_omp(input));
		cv::imshow("Sobel (OpenCL)", sobel_cl(input4).get());
		//cv::imshow("Sobel (OpenCV)", sobel_cv(input));

		cv::waitKey(0);
	}
}
