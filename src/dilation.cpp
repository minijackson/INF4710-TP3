#include "dilation.hpp"

#include <algorithm>
#include <iostream>

#include <omp.h>

cv::Mat_<uint8_t> dilate(cv::Mat_<uint8_t> const& input, const size_t radius) {
	cv::Mat_<uint8_t> output(input.rows, input.cols, static_cast<uint8_t>(0));

	int iradius = std::min({static_cast<int>(radius), input.cols, input.rows});

	for(int i = iradius; i < input.rows - iradius; ++i) {
		for(int j = iradius; j < input.cols - iradius; ++j) {

			for(int kernel_i = -iradius; kernel_i <= iradius; ++kernel_i) {
				for(int kernel_j = -iradius; kernel_j <= iradius; ++kernel_j) {
					size_t x = i + kernel_i;
					size_t y = j + kernel_j;

					if(input(x, y) != 0) {
						output(i, j) = 255;
						break;
					}
				}
			}
		}
	}

	return output;
}

cv::Mat_<uint8_t> dilate_omp(cv::Mat_<uint8_t> const& input, const size_t radius) {
	cv::Mat_<uint8_t> output(input.rows, input.cols, static_cast<uint8_t>(0));

	int iradius = std::min({static_cast<int>(radius), input.cols, input.rows});

#pragma omp parallel for collapse(2)
	for(int i = iradius; i < input.rows - iradius; ++i) {
		for(int j = iradius; j < input.cols - iradius; ++j) {

			//#pragma omp parallel for collapse(2) reduction(+:sum)
			for(int kernel_i = -iradius; kernel_i <= iradius; ++kernel_i) {
				for(int kernel_j = -iradius; kernel_j <= iradius; ++kernel_j) {
					size_t x = i + kernel_i;
					size_t y = j + kernel_j;

					if(input(x, y) != 0) {
						output(i, j) = 255;
						break;
					}

				}
			}

		}
	}

	return output;
}

cv::Mat_<uint8_t> dilate_cv(cv::Mat_<uint8_t> const& input, const size_t radius) {
	cv::Mat_<uint8_t> output(input.rows, input.cols);

	cv::Mat_<uint8_t> kernel(radius * 2 + 1, radius * 2 + 1, 1);

	cv::dilate(input, output, kernel);

	return output;
}
