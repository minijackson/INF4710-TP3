#include "threshold.hpp"

#include <algorithm>

namespace {
	uint8_t getComponent(LightnessComponent component, uint8_t red, uint8_t green, uint8_t blue) {
		switch(component) {
			case intensity:
				return (red + green + blue) / 3;
			case value:
				return std::max({red, green, blue});
			case lightness: {
				auto minmax = std::minmax({red, green, blue});
				return (minmax.first + minmax.second) / 2;
			}
			case luma:
				return 0.299 * red + 0.587 * green + 0.114 * blue;
			case luma_rounded:
				return std::round(0.299 * red + 0.587 * green + 0.114 * blue);
			default:
				throw std::runtime_error("Unknown lightness component");
		}
	}
}

cv::Mat_<uint8_t> threshold(cv::Mat const& input, uint8_t limit, LightnessComponent component) {
	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	std::transform(input.begin<cv::Vec3b>(),
	               input.end<cv::Vec3b>(),
	               output.begin(),
	               [limit, component](cv::Vec3b values) {
		               uint8_t lightness_component =
		                       getComponent(component, values[0], values[1], values[2]);
		               if(lightness_component > limit) {
			               return 255;
		               } else {
			               return 0;
		               }
	               });

	return output;
}
