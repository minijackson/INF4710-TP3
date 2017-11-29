hayai = subproject('hayai').get_variable('hayai_dep')

# Create executables

sobel_bench = executable(
	'sobel_bench.bin',
	['sobel.cpp'],
	cpp_pch: ['pch/pch.hpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, hayai]
)

threshold_components_bench = executable(
	'threshold_components_bench.bin',
	['threshold_components.cpp'],
	cpp_pch: ['pch/pch.hpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, hayai]
)

threshold_implementations_bench = executable(
	'threshold_implementations_bench.bin',
	['threshold_implementations.cpp'],
	cpp_pch: ['pch/pch.hpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, opencl_dep, hayai]
)

dilation_bench = executable(
	'dilation_bench.bin',
	['dilation.cpp'],
	cpp_pch: ['pch/pch.hpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, opencl_dep, hayai]
)

# Register benchmarks

benchmark(
	'Sobel',
	sobel_bench,
	args: ['-o', 'json:sobel.json', '-o', 'console'],
	timeout: 1200
)

benchmark(
	'Threshold (Components)',
	threshold_components_bench,
	args: ['-o', 'json:threshold_components.json', '-o', 'console'],
	timeout: 1200
)

benchmark(
	'Threshold (Implementations)',
	threshold_implementations_bench,
	args: ['-o', 'json:threshold_implementations.json', '-o', 'console'],
	timeout: 1200
)

benchmark(
	'Dilation',
	dilation_bench,
	args: ['-o', 'json:dilation.json', '-o', 'console'],
	timeout: 1200
)

# vim: ft=cmake
