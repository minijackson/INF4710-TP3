#include "../src/opencl.hpp"

#include "../src/dilation.hpp"
#include "../src/sobel.hpp"
#include "../src/threshold.hpp"

#include <gtest/gtest.h>

#include <iostream>

cv::Mat_<cv::Vec4b> load_image(std::string const& path) {
	const cv::Mat_<cv::Vec3b> input = cv::imread(path, cv::IMREAD_COLOR);
	if(input.empty()) {
		std::cerr << "Could not load image at '" << path << "', exiting." << std::endl;
		std::exit(-1);
	}

	const cv::Mat_<cv::Vec4b> input4(input.rows, input.cols);
	cv::cvtColor(input, input4, CV_RGB2RGBA);

	return input4;
}

TEST(Integration, DISABLED_Pipeline) {
	cl_singletons::setup();

	const std::vector<std::string> images{
	        "../data/airplane.png",
	        "../data/baboon.png",
	        "../data/cameraman.tif",
	        "../data/lena.png",
	        "../data/logo.tif",
	        "../data/logo_noise.tif",
	        "../data/peppers.png",
	};

	for(std::string const& path : images) {
		const auto input4 = load_image(path);
		std::cout << "\nProcessing image at '" << path << "'...\n";

		std::cout << "Sobeling..." << std::endl;
		const auto sobel_res = sobel_cl(input4);
		std::cout << "Thresholding..." << std::endl;
		const auto thresh_res = threshold_gnupar(sobel_res.get(), 170);
		std::cout << "Dilating..." << std::endl;
		const auto dilation_res = dilate_cl(thresh_res, 5);

		cv::imshow("Original", input4);
		cv::imshow("Sobel result", sobel_res.get());
		cv::imshow("Threshold result", thresh_res);
		cv::imshow("Dilation result", dilation_res.get());

		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

TEST(Integration, DISABLED_AdjacentPipeline) {
	const std::vector<std::pair<std::string, std::string>> images{
	        {"../data/mpv-shot0001.png", "../data/mpv-shot0002.png"},
	        {"../data/mpv-shot0003.png", "../data/mpv-shot0004.png"},
	        {"../data/mpv-shot0005.png", "../data/mpv-shot0006.png"},
	        {"../data/mpv-shot0007.png", "../data/mpv-shot0008.png"}};

	for(auto const& frames : images) {
		std::string first_frame_path, second_frame_path;
		std::tie(first_frame_path, second_frame_path) = frames;

		const auto first_frame  = load_image(first_frame_path);
		const auto second_frame = load_image(second_frame_path);

		std::cout << "\nProcessing image at '" << first_frame_path << "', '" << second_frame_path
		          << "'...\n";

		std::cout << "Sobeling..." << std::endl;
		const auto first_sobel_res  = sobel_cl(first_frame);
		const auto second_sobel_res = sobel_cl(second_frame);
		std::cout << "Thresholding..." << std::endl;
		const auto first_thresh_res  = threshold_gnupar(first_sobel_res.get(), 170);
		const auto second_thresh_res = threshold_gnupar(second_sobel_res.get(), 170);
		std::cout << "Dilating..." << std::endl;
		const auto first_dilation_res = dilate_cl(first_thresh_res, 1);

		cv::imshow("Original 1", first_frame);
		cv::imshow("Original 2", second_frame);

		cv::imshow("Sobel 1", first_sobel_res.get());
		cv::imshow("Sobel 2", second_sobel_res.get());
		cv::imshow("Threshold 1", first_thresh_res);
		cv::imshow("Result 1", first_dilation_res.get());
		cv::imshow("Result 2", second_thresh_res);

		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}
