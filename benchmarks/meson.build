hayai = subproject('hayai').get_variable('hayai_dep')

sobel_bench = executable(
	'sobel_bench.bin',
	['sobel.cpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, hayai]
)
threshold_bench = executable(
	'threshold_bench.bin',
	['threshold.cpp'],
	link_with: main_lib,
	dependencies: [opencv_dep, hayai]
)

benchmark(
	'Sobel',
	sobel_bench,
	args: ['-o', 'json:sobel.json', '-o', 'console'],
	timeout: 1200
)
benchmark(
	'Threshold',
	threshold_bench,
	args: ['-o', 'json:threshold.json', '-o', 'console'],
	timeout: 1200
)

# vim: ft=cmake
