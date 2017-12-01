#include "threshold.hpp"

#include "opencl.hpp"
#include "cl_mat.hpp"

#include <opencv2/core/ocl.hpp>

#include <algorithm>
#include <parallel/algorithm>

namespace {
	uint8_t getComponent(LightnessComponent component, uint8_t red, uint8_t green, uint8_t blue) {
		switch(component) {
			case intensity:
				return (red + green + blue) / 3;
			case value:
				return std::max({red, green, blue});
			case lightness: {
				auto minmax = std::minmax({red, green, blue});
				return (minmax.first + minmax.second) / 2;
			}
			case luma:
				return 0.299 * red + 0.587 * green + 0.114 * blue;
			case luma_rounded:
				return std::round(0.299 * red + 0.587 * green + 0.114 * blue);
			default:
				throw std::runtime_error("Unknown lightness component");
		}
	}
}

cv::Mat_<uint8_t> threshold(cv::Mat_<cv::Vec4b> const& input,
                            uint8_t limit,
                            LightnessComponent component) {
	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	std::transform(
	        input.begin(), input.end(), output.begin(), [limit, component](cv::Vec4b values) {
		        uint8_t lightness_component =
		                getComponent(component, values[0], values[1], values[2]);
		        if(lightness_component > limit) {
			        return 255;
		        } else {
			        return 0;
		        }
	        });

	return output;
}

cv::Mat_<uint8_t> threshold_gnupar(cv::Mat_<cv::Vec4b> const& input,
                                   uint8_t limit,
                                   LightnessComponent component) {
	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	__gnu_parallel::transform(
	        input.begin(), input.end(), output.begin(), [limit, component](cv::Vec4b values) {
		        uint8_t lightness_component =
		                getComponent(component, values[0], values[1], values[2]);
		        if(lightness_component > limit) {
			        return 255;
		        } else {
			        return 0;
		        }
	        });

	return output;
}

CLMat<uint8_t> threshold_cl(cv::Mat_<cv::Vec4b> const& input, uint8_t limit) {
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
	cl::KernelFunctor threshold_cl_functor(
	        cl_singletons::threshold_kernel,
	        cl_singletons::queue,
	        /* global_work_offset = */ cl::NullRange,
	        /* global_work_size = */ cl::NDRange(input.rows, input.cols),
	        /* local_work_size = */ cl::NullRange);

	// Call it
	threshold_cl_functor(input_buffer, limit, output_buffer);

	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	// Get the result back
	// cl_singletons::queue.enqueueReadImage(output_buffer,
	//                                      /* blocking = */ CL_TRUE,
	//                                      origin,
	//                                      region,
	//                                      /* row_pitch = */ 0,
	//                                      /* slice_pitch = */ 0,
	//                                      output.data);


	//void* mapped_memory = cl_singletons::queue.enqueueMapImage(output_buffer,
	//                                                           /* blocking = */ CL_TRUE,
	//                                                           CL_MAP_READ,
	//                                                           origin,
	//                                                           region,
	//                                                           /* row_pitch = */ 0,
	//                                                           /* slice_pitch = */ 0);

	//memcpy(output.data, mapped_memory, sizeof(uint8_t) * input.rows * input.cols);
	//cl_singletons::queue.enqueueUnmapMemObject(output_buffer, mapped_memory);

	//return output;

	return CLMat<uint8_t>(input.rows, input.cols, output_buffer);
}

cv::Mat_<uint8_t> threshold_cv(cv::Mat_<cv::Vec4b> const& input, uint8_t limit) {
	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	cv::cvtColor(input, output, CV_BGRA2GRAY);
	cv::threshold(output, output, limit, 255, 0);

	return output;
}

cv::Mat_<uint8_t> threshold_cvcl(cv::Mat_<cv::Vec4b> const& input, uint8_t limit) {
	cv::ocl::setUseOpenCL(true);

	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	cv::cvtColor(input, output, CV_BGRA2GRAY);
	cv::threshold(output, output, limit, 255, 0);

	cv::ocl::setUseOpenCL(false);

	return output;
}
