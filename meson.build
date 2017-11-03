project('Multimedia TP3', 'cpp', default_options: ['cpp_std=c++1z'], license: 'MIT')

opencv_dep = dependency('opencv')
openmp_dep = declare_dependency(
	compile_args: '-fopenmp',
	link_args: '-fopenmp'
)

main = executable(
	'main.bin',
	['src/main.cpp',
	 'src/sobel.cpp',
	 'src/threshold.cpp'],
	dependencies: [opencv_dep, openmp_dep]
)

main_lib = library(
	'main',
	['src/sobel.cpp',
	 'src/threshold.cpp'],
	dependencies: [opencv_dep, openmp_dep]
)

subdir('./tests')
subdir('./benchmarks')

# vim: ft=cmake
