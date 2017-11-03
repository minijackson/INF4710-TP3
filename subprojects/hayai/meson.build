project('Hayai', 'cpp', default_options: ['cpp_std=c++1z'])

inc = include_directories('hayai/src/')
lib = library('hayai', ['hayai/src/hayai_posix_main.cpp'], )

hayai_dep = declare_dependency(include_directories: inc, link_with: lib)

# vim: ft=cmake
