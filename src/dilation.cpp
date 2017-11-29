#include "dilation.hpp"

#include "opencl.hpp"

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
						goto outtahere;
					}
				}
			}
outtahere:
			// Noop
			(void)0;
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
						goto outtahere;
					}

				}
			}
outtahere:
			// Noop
			(void)0;
		}
	}

	return output;
}

CLMat<uint8_t> dilate_cl(cv::Mat_<uint8_t> const& input, const size_t radius) {
	// Create buffers that represents variables in the device's side
	cl::Image2D input_buffer(cl_singletons::context,
			CL_MEM_READ_ONLY,
			cl::ImageFormat(CL_INTENSITY, CL_UNSIGNED_INT8),
			input.rows,
			input.cols,
			/* image_row_pitch = */ 0,
			input.data);

	cl::Image2D output_buffer(cl_singletons::context,
			CL_MEM_WRITE_ONLY,
			cl::ImageFormat(CL_INTENSITY, CL_UNSIGNED_INT8),
			input.rows,
			input.cols,
			/* image_row_pitch = */ 0);

	cl::size_t<3> origin;
	origin.push_back(0);
	origin.push_back(0);
	origin.push_back(0);

	cl::size_t<3> region;
	region.push_back(input.rows);
	region.push_back(input.cols);
	region.push_back(1);

	// Send the image to the device
	cl_singletons::queue.enqueueWriteImage(input_buffer,
			/* blocking = */ CL_FALSE,
			origin,
			region,
			/* row_pitch = */ 0,
			/* slice_pitch = */ 0,
			input.data);

	// Create a functor (object that can be called) that will "call" the OpenCL function
	cl::KernelFunctor dilate_cl_functor(
			cl_singletons::dilation_kernel,
			cl_singletons::queue,
			/* global_work_offset = */ cl::NullRange,
			/* global_work_size = */ cl::NDRange(input.rows, input.cols),
			/* local_work_size = */ cl::NullRange);

	// Call it
	dilate_cl_functor(input_buffer, static_cast<int>(radius), output_buffer);

	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	return CLMat<uint8_t>(input.rows, input.cols, output_buffer);
}

cv::Mat_<uint8_t> dilate_cv(cv::Mat_<uint8_t> const& input, const size_t radius) {
	cv::Mat_<uint8_t> output(input.rows, input.cols);

	cv::Mat_<uint8_t> kernel(radius * 2 + 1, radius * 2 + 1, 1);

	cv::dilate(input, output, kernel);

	return output;
}
