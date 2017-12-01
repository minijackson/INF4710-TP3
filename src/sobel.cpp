#include "sobel.hpp"

#include "opencl.hpp"

#include <algorithm>

#include <cmath>

const cv::Mat_<int8_t> sobelx = (cv::Mat_<int8_t>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
const cv::Mat_<int8_t> sobely = (cv::Mat_<int8_t>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

cv::Mat_<cv::Vec4b> sobel(cv::Mat_<cv::Vec4b> RGB) {

	cv::Vec4b nullVec{0, 0, 0, 255};
	cv::Mat_<cv::Vec4b> FGRGB(RGB.rows, RGB.cols, nullVec);

	for(int x = 1; x < RGB.rows - 1; ++x) {
		for(int y = 1; y < RGB.cols - 1; ++y) {

			int redGx = (sobelx(0, 0) * RGB.at<cv::Vec4b>(x - 1, y - 1)[2]) +
			            (sobelx(0, 1) * RGB.at<cv::Vec4b>(x, y - 1)[2]) +
			            (sobelx(0, 2) * RGB.at<cv::Vec4b>(x + 1, y - 1)[2]) +
			            (sobelx(1, 0) * RGB.at<cv::Vec4b>(x - 1, y)[2]) +
			            (sobelx(1, 1) * RGB.at<cv::Vec4b>(x, y)[2]) +
			            (sobelx(1, 2) * RGB.at<cv::Vec4b>(x + 1, y)[2]) +
			            (sobelx(2, 0) * RGB.at<cv::Vec4b>(x - 1, y + 1)[2]) +
			            (sobelx(2, 1) * RGB.at<cv::Vec4b>(x, y + 1)[2]) +
			            (sobelx(2, 2) * RGB.at<cv::Vec4b>(x + 1, y + 1)[2]);

			int redGy = (sobely(0, 0) * RGB.at<cv::Vec4b>(x - 1, y - 1)[2]) +
			            (sobely(0, 1) * RGB.at<cv::Vec4b>(x, y - 1)[2]) +
			            (sobely(0, 2) * RGB.at<cv::Vec4b>(x + 1, y - 1)[2]) +
			            (sobely(1, 0) * RGB.at<cv::Vec4b>(x - 1, y)[2]) +
			            (sobely(1, 1) * RGB.at<cv::Vec4b>(x, y)[2]) +
			            (sobely(1, 2) * RGB.at<cv::Vec4b>(x + 1, y)[2]) +
			            (sobely(2, 0) * RGB.at<cv::Vec4b>(x - 1, y + 1)[2]) +
			            (sobely(2, 1) * RGB.at<cv::Vec4b>(x, y + 1)[2]) +
			            (sobely(2, 2) * RGB.at<cv::Vec4b>(x + 1, y + 1)[2]);

			int greenGx = (sobelx(0, 0) * RGB.at<cv::Vec4b>(x - 1, y - 1)[1]) +
			              (sobelx(0, 1) * RGB.at<cv::Vec4b>(x, y - 1)[1]) +
			              (sobelx(0, 2) * RGB.at<cv::Vec4b>(x + 1, y - 1)[1]) +
			              (sobelx(1, 0) * RGB.at<cv::Vec4b>(x - 1, y)[1]) +
			              (sobelx(1, 1) * RGB.at<cv::Vec4b>(x, y)[1]) +
			              (sobelx(1, 2) * RGB.at<cv::Vec4b>(x + 1, y)[1]) +
			              (sobelx(2, 0) * RGB.at<cv::Vec4b>(x - 1, y + 1)[1]) +
			              (sobelx(2, 1) * RGB.at<cv::Vec4b>(x, y + 1)[1]) +
			              (sobelx(2, 2) * RGB.at<cv::Vec4b>(x + 1, y + 1)[1]);

			int greenGy = (sobely(0, 0) * RGB.at<cv::Vec4b>(x - 1, y - 1)[1]) +
			              (sobely(0, 1) * RGB.at<cv::Vec4b>(x, y - 1)[1]) +
			              (sobely(0, 2) * RGB.at<cv::Vec4b>(x + 1, y - 1)[1]) +
			              (sobely(1, 0) * RGB.at<cv::Vec4b>(x - 1, y)[1]) +
			              (sobely(1, 1) * RGB.at<cv::Vec4b>(x, y)[1]) +
			              (sobely(1, 2) * RGB.at<cv::Vec4b>(x + 1, y)[1]) +
			              (sobely(2, 0) * RGB.at<cv::Vec4b>(x - 1, y + 1)[1]) +
			              (sobely(2, 1) * RGB.at<cv::Vec4b>(x, y + 1)[1]) +
			              (sobely(2, 2) * RGB.at<cv::Vec4b>(x + 1, y + 1)[1]);

			int blueGx = (sobelx(0, 0) * RGB.at<cv::Vec4b>(x - 1, y - 1)[0]) +
			             (sobelx(0, 1) * RGB.at<cv::Vec4b>(x, y - 1)[0]) +
			             (sobelx(0, 2) * RGB.at<cv::Vec4b>(x + 1, y - 1)[0]) +
			             (sobelx(1, 0) * RGB.at<cv::Vec4b>(x - 1, y)[0]) +
			             (sobelx(1, 1) * RGB.at<cv::Vec4b>(x, y)[0]) +
			             (sobelx(1, 2) * RGB.at<cv::Vec4b>(x + 1, y)[0]) +
			             (sobelx(2, 0) * RGB.at<cv::Vec4b>(x - 1, y + 1)[0]) +
			             (sobelx(2, 1) * RGB.at<cv::Vec4b>(x, y + 1)[0]) +
			             (sobelx(2, 2) * RGB.at<cv::Vec4b>(x + 1, y + 1)[0]);

			int blueGy = (sobely(0, 0) * RGB.at<cv::Vec4b>(x - 1, y - 1)[0]) +
			             (sobely(0, 1) * RGB.at<cv::Vec4b>(x, y - 1)[0]) +
			             (sobely(0, 2) * RGB.at<cv::Vec4b>(x + 1, y - 1)[0]) +
			             (sobely(1, 0) * RGB.at<cv::Vec4b>(x - 1, y)[0]) +
			             (sobely(1, 1) * RGB.at<cv::Vec4b>(x, y)[0]) +
			             (sobely(1, 2) * RGB.at<cv::Vec4b>(x + 1, y)[0]) +
			             (sobely(2, 0) * RGB.at<cv::Vec4b>(x - 1, y + 1)[0]) +
			             (sobely(2, 1) * RGB.at<cv::Vec4b>(x, y + 1)[0]) +
			             (sobely(2, 2) * RGB.at<cv::Vec4b>(x + 1, y + 1)[0]);

			double redFG   = std::sqrt(redGx * redGx + redGy * redGy);
			double greenFG = std::sqrt(greenGx * greenGx + greenGy * greenGy);
			double blueFG  = std::sqrt(blueGx * blueGx + blueGy * blueGy);

			FGRGB.at<cv::Vec4b>(x, y)[0] = std::clamp(blueFG, 0.0, 255.0);
			FGRGB.at<cv::Vec4b>(x, y)[1] = std::clamp(greenFG, 0.0, 255.0);
			FGRGB.at<cv::Vec4b>(x, y)[2] = std::clamp(redFG, 0.0, 255.0);
			FGRGB.at<cv::Vec4b>(x, y)[3] = 255;
		}
	}

	return FGRGB;
}

CLMat<cv::Vec4b> sobel_cl(cv::Mat_<cv::Vec4b> const& input) {
	// Create buffers that represents variables in the device's side
	cl::Image2D input_buffer(cl_singletons::context,
	                         CL_MEM_READ_ONLY,
	                         cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
	                         input.cols,
	                         input.rows,
	                         /* image_row_pitch = */ 0,
	                         input.data);

	cl::Image2D output_buffer(cl_singletons::context,
	                          CL_MEM_WRITE_ONLY,
	                          cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
	                          input.cols,
	                          input.rows,
	                          /* image_row_pitch = */ 0);

	cl::size_t<3> origin;
	origin.push_back(0);
	origin.push_back(0);
	origin.push_back(0);

	cl::size_t<3> region;
	region.push_back(input.cols);
	region.push_back(input.rows);
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
	cl::KernelFunctor sobel_cl_functor(cl_singletons::sobel_kernel,
	                                   cl_singletons::queue,
	                                   /* global_work_offset = */ cl::NullRange,
	                                   /* global_work_size = */ cl::NDRange(input.cols, input.rows),
	                                   /* local_work_size = */ cl::NullRange);

	// Call it
	sobel_cl_functor(input_buffer, output_buffer);

	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	return CLMat<cv::Vec4b>(input.rows, input.cols, output_buffer);
}
