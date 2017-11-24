#pragma once

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
#else
#	include <CL/cl.hpp>
#endif

#include <vector>

namespace cl_singletons {

	extern cl::Context context;
	extern std::vector<cl::Platform> platforms;
	extern std::vector<cl::Device> devices;

	void setup();
	cl::Program program_from_file(char const* filename);

}
