#pragma once

#include <opencv2/opencv.hpp>

enum LightnessComponent { intensity, value, lightness, luma, luma_rounded };

cv::Mat_<uint8_t> threshold(cv::Mat const& input,
                            uint8_t limit,
                            LightnessComponent component = intensity);

cv::Mat_<uint8_t> threshold_gnupar(cv::Mat const& input,
                                   uint8_t limit,
                                   LightnessComponent component = intensity);
