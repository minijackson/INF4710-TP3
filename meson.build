project('Multimedia TP3', 'cpp', default_options: ['cpp_std=c++1z'], license: 'MIT')

opencv_dep = dependency('opencv')

main = executable('main.bin', ['src/main.cpp', 'src/sobel.cpp', 'src/threshold.cpp'], dependencies: opencv_dep)
main_lib = shared_library('main', ['src/sobel.cpp', 'src/threshold.cpp'], dependencies: opencv_dep)

sobel = executable('sobel.bin', ['tests/sobel.cpp'], link_with: main_lib, dependencies: opencv_dep)
threshold = executable('threshold.bin', ['tests/threshold.cpp'], link_with: main_lib, dependencies: opencv_dep)
dilation = executable('dilation.bin', ['tests/dilation.cpp'], link_with: main_lib, dependencies: opencv_dep)
edge_ratio = executable('edge_ratio.bin', ['tests/edge_ratio.cpp'], link_with: main_lib, dependencies: opencv_dep)
transition_detection = executable('transition_detection.bin', ['tests/transition_detection.cpp'], link_with: main_lib, dependencies: opencv_dep)

test('Sobel', sobel)
test('Threshold', threshold)
test('Dilation', dilation)
test('Edge Ration', edge_ratio)
test('Transition Detection', transition_detection)

# vim: ft=cmake
