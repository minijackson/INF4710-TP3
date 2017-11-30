gtest_dep = dependency('gtest', main: true, fallback: ['gtest', 'gtest_dep'])

sobel = executable(
	'sobel_test.bin',
	['sobel.cpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, gtest_dep, opencl_dep]
)
threshold = executable(
	'threshold_test.bin',
	['threshold.cpp'],
	cpp_pch: ['pch/pch.hpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, gtest_dep, opencl_dep]
)
dilation = executable(
	'dilation_test.bin',
	['dilation.cpp'],
	cpp_pch: ['pch/pch.hpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, gtest_dep, opencl_dep]
)
edge_ratio = executable(
	'edge_ratio_test.bin',
	['edge_ratio.cpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, gtest_dep]
)
transition_detection = executable(
	'transition_detection_test.bin',
	['transition_detection.cpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, gtest_dep]
)

test('Sobel', sobel)
test('Threshold', threshold)
test('Dilation', dilation)
test('Edge Ration', edge_ratio)
test('Transition Detection', transition_detection)

# vim: ft=cmake
