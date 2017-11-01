#pragma once

#include <opencv2/opencv.hpp>

enum LightnessComponent {
	intensity,
	value,
	lightness,
	luma,
};

cv::Mat_<uint8_t> threshold(cv::Mat const& input,
                            uint8_t limit,
                            LightnessComponent component = intensity);
