#pragma once

#include "cl_mat.hpp"

#include <opencv2/opencv.hpp>

cv::Mat_<cv::Vec3b> sobel(cv::Mat_<cv::Vec3b> const& RGB);
CLMat<cv::Vec4b> sobel_cl(cv::Mat_<cv::Vec4b> const& input);
