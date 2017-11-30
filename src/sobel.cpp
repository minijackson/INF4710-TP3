#include "sobel.hpp"

#include "opencl.hpp"

cv::Mat_<cv::Vec3b> sobel(cv::Mat_<cv::Vec3b> const& RGB) {

}

CLMat<cv::Vec4b> sobel_cl(cv::Mat_<cv::Vec4b> const& input) {
	// Create buffers that represents variables in the device's side
	cl::Image2D input_buffer(cl_singletons::context,
			CL_MEM_READ_ONLY,
			cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
			input.rows,
			input.cols,
			/* image_row_pitch = */ 0,
			input.data);

	cl::Image2D output_buffer(cl_singletons::context,
			CL_MEM_WRITE_ONLY,
			cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
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
	cl::KernelFunctor sobel_cl_functor(
			cl_singletons::sobel_kernel,
			cl_singletons::queue,
			/* global_work_offset = */ cl::NullRange,
			/* global_work_size = */ cl::NDRange(input.rows, input.cols),
			/* local_work_size = */ cl::NullRange);

	// Call it
	sobel_cl_functor(input_buffer, output_buffer);

	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	return CLMat<cv::Vec4b>(input.rows, input.cols, output_buffer);
}
