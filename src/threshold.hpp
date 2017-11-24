#pragma once

#include <opencv2/opencv.hpp>

enum LightnessComponent { intensity, value, lightness, luma, luma_rounded };

cv::Mat_<uint8_t> threshold(cv::Mat_<cv::Vec3b> const& input,
                            uint8_t limit,
                            LightnessComponent component = intensity);

cv::Mat_<uint8_t> threshold_gnupar(cv::Mat_<cv::Vec3b> const& input,
                                   uint8_t limit,
                                   LightnessComponent component = intensity);

cv::Mat_<uint8_t> threshold_cl(cv::Mat_<cv::Vec4b> const& input, uint8_t limit);
