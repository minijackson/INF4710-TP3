#pragma once

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
#else
#	include <CL/cl.hpp>
#endif

#include <vector>

namespace cl_singletons {

	extern cl::Kernel threshold_kernel;
	extern cl::Kernel dilation_kernel;

	extern cl::CommandQueue queue;
	extern cl::Context context;
	extern std::vector<cl::Platform> platforms;
	extern std::vector<cl::Device> devices;

	void setup();
	cl::Kernel kernel_from_file(char const* filename, char const* name);

}
