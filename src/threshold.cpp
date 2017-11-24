#include "threshold.hpp"

#include "opencl.hpp"

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

cv::Mat_<uint8_t> threshold(cv::Mat_<cv::Vec3b> const& input,
                            uint8_t limit,
                            LightnessComponent component) {
	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	std::transform(
	        input.begin(), input.end(), output.begin(), [limit, component](cv::Vec3b values) {
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

cv::Mat_<uint8_t> threshold_gnupar(cv::Mat_<cv::Vec3b> const& input,
                                   uint8_t limit,
                                   LightnessComponent component) {
	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	__gnu_parallel::transform(
	        input.begin(), input.end(), output.begin(), [limit, component](cv::Vec3b values) {
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

cv::Mat_<uint8_t> threshold_cl(cv::Mat_<cv::Vec4b> const& input, uint8_t limit) {
	static bool done_setup = false;
	static cl::Kernel kernel;
	if(!done_setup) {
		cl_singletons::setup();

		cl::Program program = cl_singletons::program_from_file("../src/opencl/threshold.cl");
		cl_int error;

		kernel = cl::Kernel(program, "threshold", &error);
		if(error != 0) {
			perror("Failed to create the threshold kernel");
			std::exit(error);
		}
		std::cout << "Created Kernel: " << kernel.getInfo<CL_KERNEL_FUNCTION_NAME>() << std::endl;

		done_setup = true;
	}

	cl::CommandQueue queue(cl_singletons::context, cl_singletons::devices[0]);

	// Create buffers that represents variables in the device's side
	cl::Image2D input_buffer(cl_singletons::context,
	                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
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
	queue.enqueueWriteImage(input_buffer,
	                        /* blocking = */ CL_TRUE,
	                        origin,
	                        region,
	                        /* row_pitch = */ 0,
	                        /* slice_pitch = */ 0,
	                        input.data);

	// Create a functor (object that can be called) that will "call" the OpenCL function
	cl::KernelFunctor threshold_cl_functor(
	        kernel,
	        queue,
	        /* global_work_offset = */ cl::NullRange,
	        /* global_work_size = */ cl::NDRange(input.rows, input.cols),
	        /* local_work_size = */ cl::NullRange);

	// Call it
	threshold_cl_functor(input_buffer, limit, output_buffer);

	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	// Get the result back
	queue.enqueueReadImage(output_buffer,
	                       /* blocking = */ CL_TRUE,
	                       origin,
	                       region,
	                       /* row_pitch = */ 0,
	                       /* slice_pitch = */ 0,
	                       output.data);

	return output;
}
