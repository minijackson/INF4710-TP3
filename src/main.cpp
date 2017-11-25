#include <iostream>
#include <opencv2/opencv.hpp>

#include "sobel.hpp"

int main(int argc, char const* argv[]) {

	try {
		const std::vector<std::pair<std::string, int>> vImages = {
		        {"../data/airplane.png", cv::IMREAD_COLOR},
		        {"../data/baboon.png", cv::IMREAD_COLOR},
		        {"../data/cameraman.tif", cv::IMREAD_COLOR},
		        {"../data/lena.png", cv::IMREAD_COLOR},
		        {"../data/logo.tif", cv::IMREAD_COLOR},
		        {"../data/logo_noise.tif", cv::IMREAD_COLOR},
		        {"../data/peppers.png", cv::IMREAD_COLOR},
		};

		for(const std::pair<std::string, int>& oImagePathFlag : vImages) {
			const cv::Mat_<cv::Vec3b> oInput =
			        cv::imread(oImagePathFlag.first, oImagePathFlag.second);
			if(oInput.empty()) {
				std::cerr << "Could not load image at '" << oImagePathFlag.first << "'; exiting..."
				          << std::endl;
				std::exit(-1);
			}
			std::cout << "\nProcessing image at '" << oImagePathFlag.first << "'..." << std::endl;

			sobel(oInput);
		}

	} catch(...) {
		std::cerr << "Caught unhandled exception." << std::endl;
	}
}
