#include "opencl.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cl_singletons {

	cl::CommandQueue queue;
	cl::Context context;
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;

	void setup() {
		static bool done = false;
		if(done) {
			return;
		}
		done = true;

		cl::Platform::get(&platforms);
		if(platforms.size() == 0) {
			throw std::runtime_error("No OpenCL compatible platform found on this system");
		}

		std::cout << "Found " << platforms.size() << " OpenCL compatible platforms\n"
			"Using: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;

		platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if(devices.size() == 0) {
			throw std::runtime_error("No OpenCL compatible devices found on this system");
		}

		std::cout << "Found " << devices.size()
		          << " OpenCL compatible devices\n"
		             "Using: "
		          << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

		context = cl::Context(devices);
		std::cout << "OpenCL context created" << std::endl;

		queue = cl::CommandQueue(context, devices[0]);
		std::cout << "OpenCL command queue created" << std::endl;
	}

	cl::Program program_from_file(char const* filename) {
		std::ifstream source_file(filename);
		std::string source_file_content{std::istreambuf_iterator<char>(source_file),
		                                std::istreambuf_iterator<char>()};
		cl::Program::Sources source(
		        1, std::make_pair(source_file_content.c_str(), source_file_content.length() + 1));

		cl_int error;
		cl::Program program(context, source, &error);
		if(error != 0) {
			perror("Failed to create program");
			std::exit(error);
		}

		error = program.build(devices);
		if(error != 0) {
			std::cerr << "Failed to build program:\n";
			for(auto const& device : devices) {
				std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
			}
			std::exit(error);
		}

		return program;
	}
}
