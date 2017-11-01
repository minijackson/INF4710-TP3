project('Multimedia TP3', 'cpp', default_options: ['cpp_std=c++1z'], license: 'MIT')

opencv_dep = dependency('opencv')

main = executable('main.bin', ['src/main.cpp', 'src/sobel.cpp', 'src/threshold.cpp'], dependencies: opencv_dep)
main_lib = library('main', ['src/sobel.cpp', 'src/threshold.cpp'], dependencies: opencv_dep)

sobel = executable('sobel_test.bin', ['tests/sobel.cpp'], link_with: main_lib, dependencies: opencv_dep)
threshold = executable('threshold_test.bin', ['tests/threshold.cpp'], link_with: main_lib, dependencies: opencv_dep)
dilation = executable('dilation_test.bin', ['tests/dilation.cpp'], link_with: main_lib, dependencies: opencv_dep)
edge_ratio = executable('edge_ratio_test.bin', ['tests/edge_ratio.cpp'], link_with: main_lib, dependencies: opencv_dep)
transition_detection = executable('transition_detection_test.bin', ['tests/transition_detection.cpp'], link_with: main_lib, dependencies: opencv_dep)

test('Sobel', sobel)
test('Threshold', threshold)
test('Dilation', dilation)
test('Edge Ration', edge_ratio)
test('Transition Detection', transition_detection)

hayai = subproject('hayai').get_variable('hayai_dep')

sobel_bench = executable('sobel_bench.bin', ['benchmarks/sobel.cpp'], link_with: main_lib, dependencies: [opencv_dep, hayai])
threshold_bench = executable('threshold_bench.bin', ['benchmarks/threshold.cpp'], link_with: main_lib, dependencies: [opencv_dep, hayai])

benchmark('Sobel', sobel_bench, args: ['-o', 'json:sobel.json'], timeout: 1200)
benchmark('Threshold', threshold_bench, args: ['-o', 'json:threshold.json'], timeout: 1200)

# vim: ft=cmake
