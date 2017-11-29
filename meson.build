project('Multimedia TP3', 'cpp', default_options: ['cpp_std=c++1z'], license: 'MIT')

opencv_dep = dependency('opencv')
openmp_dep = declare_dependency(
	compile_args: '-fopenmp',
	link_args: '-fopenmp'
)
opencl_dep = declare_dependency(link_args: '-lOpenCL')

sources = ['src/sobel.cpp',
           'src/threshold.cpp',
           'src/dilation.cpp',
		   'src/opencl.cpp']

main = executable(
	'main.bin',
	sources + ['src/main.cpp'],
	dependencies: [opencv_dep, openmp_dep, opencl_dep]
)

main_lib = static_library(
	'main',
	sources,
	dependencies: [opencv_dep, openmp_dep, opencl_dep]
)

subdir('./tests')
subdir('./benchmarks')

# vim: ft=cmake
