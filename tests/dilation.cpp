#include "gtest/gtest.h"

#include "../src/dilation.hpp"
#include "../src/threshold.hpp"

TEST(Threshold, DISABLED_AllImplementations) {
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
