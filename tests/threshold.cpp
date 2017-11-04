#include "common.hpp"

#include "../src/threshold.hpp"

#include <cassert>

int main() {
	uint8_t test1_data[12]      = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
	cv::Mat test1               = cv::Mat(2, 2, CV_8UC3, test1_data);
	cv::Mat_<uint8_t> expected1 = (cv::Mat_<uint8_t>(2, 2) << 0, 0, 0, 0),
	                  expected2 = (cv::Mat_<uint8_t>(2, 2) << 255, 255, 255, 255),
	                  expected3 = (cv::Mat_<uint8_t>(2, 2) << 0, 0, 255, 255);

	std::cout << "Testing Threshold 1..." << std::endl;
	assert_mat_equal(threshold(test1, 40), expected1);
	std::cout << "Testing Threshold 2..." << std::endl;
	assert_mat_equal(threshold(test1, 0), expected2);

	std::cout << "Testing Threshold 3 (intensity)..." << std::endl;
	assert_mat_equal(threshold(test1, 2, LightnessComponent::intensity), expected3);
	std::cout << "Testing Threshold 3 (lightness)..." << std::endl;
	assert_mat_equal(threshold(test1, 2, LightnessComponent::lightness), expected3);
	std::cout << "Testing Threshold 3 (luma)..." << std::endl;
	assert_mat_equal(threshold(test1, 2, LightnessComponent::luma), expected3);
	std::cout << "Testing Threshold 3 (value)..." << std::endl;
	assert_mat_equal(threshold(test1, 2, LightnessComponent::value), expected3);

	std::cout << "Testing Parallel Threshold 3 (intensity)..." << std::endl;
	assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::intensity), expected3);
	std::cout << "Testing Parallel Threshold 3 (lightness)..." << std::endl;
	assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::lightness), expected3);
	std::cout << "Testing Parallel Threshold 3 (luma)..." << std::endl;
	assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::luma), expected3);
	std::cout << "Testing Parallel Threshold 3 (value)..." << std::endl;
	assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::value), expected3);

	//cv::Mat cameraman = cv::imread("../data/cameraman.tif", cv::IMREAD_COLOR);
	//std::cout << "Testing cameraman..." << std::endl;
	//cv::imshow("Originale", cameraman);
	//cv::imshow("Cameraman", threshold(cameraman, 127, luma_rounded));

	//cv::waitKey(0);
}
