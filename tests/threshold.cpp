#include "common.hpp"

#include "../src/opencl.hpp"
#include "../src/threshold.hpp"

#include <cassert>

int main() {
	cv::Vec3b test1_data[12] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
	cv::Mat_<cv::Vec3b> test1(2, 2, test1_data);
	cv::Mat_<uint8_t> expected1 = (cv::Mat_<uint8_t>(2, 2) << 0, 0, 0, 0),
	                  expected2 = (cv::Mat_<uint8_t>(2, 2) << 255, 255, 255, 255),
	                  expected3 = (cv::Mat_<uint8_t>(2, 2) << 0, 0, 255, 255);

	std::cout << "Testing Threshold 1..." << std::endl;
	assert_mat_equal(threshold(test1, 40), expected1);
	std::cout << "Testing Threshold 2..." << std::endl;
	assert_mat_equal(threshold(test1, 0), expected2);

	std::cout << "Testing Threshold 3 (intensity)..." << std::endl;
	assert_mat_equal(threshold(test1, 2, LightnessComponent::intensity), expected3);
	std::cout << "Testing Threshold 4 (lightness)..." << std::endl;
	assert_mat_equal(threshold(test1, 2, LightnessComponent::lightness), expected3);
	std::cout << "Testing Threshold 5 (luma)..." << std::endl;
	assert_mat_equal(threshold(test1, 2, LightnessComponent::luma), expected3);
	std::cout << "Testing Threshold 6 (value)..." << std::endl;
	assert_mat_equal(threshold(test1, 2, LightnessComponent::value), expected3);

	std::cout << "Testing Parallel Threshold 7 (intensity)..." << std::endl;
	assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::intensity), expected3);
	std::cout << "Testing Parallel Threshold 8 (lightness)..." << std::endl;
	assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::lightness), expected3);
	std::cout << "Testing Parallel Threshold 9 (luma)..." << std::endl;
	assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::luma), expected3);
	std::cout << "Testing Parallel Threshold 10 (value)..." << std::endl;
	assert_mat_equal(threshold_gnupar(test1, 2, LightnessComponent::value), expected3);

	cv::Vec4b test2_data[12] = {{1, 1, 1, 1}, {2, 2, 2, 1}, {3, 3, 3, 1}, {4, 4, 4, 1}};
	cv::Mat_<cv::Vec4b> test2(2, 2, test2_data);

	cl_singletons::setup();

	std::cout << "Testing OpenCL Threshold 11..." << std::endl;
	assert_mat_equal(threshold_cl(test2, 127), expected1);
	std::cout << "Testing OpenCL Threshold 12..." << std::endl;
	assert_mat_equal(threshold_cl(test2, 0), expected2);
	std::cout << "Testing OpenCL Threshold 13..." << std::endl;
	assert_mat_equal(threshold_cl(test2, 2), expected3);

	//cv::Mat_<cv::Vec3b> lena = cv::imread("../data/lena.png", cv::IMREAD_COLOR);
	//std::cout << "Testing lena..." << std::endl;
	//cv::imshow("Originale", lena);
	//cv::imshow("Lena 1", threshold(lena, 127, intensity));
	//cv::imshow("Lena 2", threshold(lena, 127, value));
	//cv::imshow("Lena 3", threshold(lena, 127, lightness));
	//cv::imshow("Lena 4", threshold(lena, 127, luma));
	//cv::imshow("Lena 5", threshold(lena, 127, luma_rounded));
	//cv::imshow("Lena OpenCV", threshold_cv(lena, 127, intensity));
	//cv::imshow("Lena OpenCV + OpenCL", threshold_cvcl(lena, 127, intensity));

	//cv::waitKey(0);
}
