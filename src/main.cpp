#include "cl_mat.hpp"
#include "opencl.hpp"

#include "dilation.hpp"
#include "edge_ratio.hpp"
#include "sobel.hpp"
#include "threshold.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <optional>

constexpr const double LIMIT    = 0.5;
constexpr const double FX_LIMIT = 0.99;

constexpr const int PROGRESS_BAR_SIZE = 42;

cv::Mat_<uint8_t> pipeline(cv::Mat_<cv::Vec4b> input) {
	const auto sobel_res  = sobel_cl(input);
	const auto thresh_res = threshold_gnupar(sobel_res.get(), 170);

	return thresh_res;
}

int main() {
	cl_singletons::setup();

	std::cout << "Loading video..." << std::endl;
	cv::VideoCapture input_video("../data/INF4710_TP3_A2017_video.avi");

	size_t frame_count = input_video.get(CV_CAP_PROP_FRAME_COUNT), frame_id = 1;

	std::cout << "Processing " << frame_count << " frames\n\n";

	cv::Mat first_frame, second_frame;
	cv::Mat_<cv::Vec4b> first_frame4, second_frame4;

	input_video >> first_frame;
	input_video >> second_frame;

	std::vector<cv::Mat_<cv::Vec3b>> scenes{first_frame.clone()};

	cv::cvtColor(first_frame, first_frame4, CV_RGB2RGBA);

	cv::Mat_<uint8_t> first_thresh_res = pipeline(first_frame4);

	std::optional<cv::Mat_<uint8_t>> saved_dilated_buffer;

	while(!second_frame.empty()) {
		if(frame_id % 50 == 0) {
			size_t progress = (frame_id / (double)frame_count) * PROGRESS_BAR_SIZE;
			std::cout << "[1000D[" << std::string(progress, '#')
			          << std::string(PROGRESS_BAR_SIZE - progress, ' ') << "]" << std::flush;
		}
		++frame_id;

		cv::cvtColor(second_frame, second_frame4, CV_RGB2RGBA);

		cv::Mat_<uint8_t> second_thresh_res = pipeline(second_frame4);

		auto first_dilation_res = dilate_cl(first_thresh_res, 2);

		const double ratio = edge_ratio_omp(first_dilation_res.get(), second_thresh_res);

		if(ratio < LIMIT) {
			std::cout << "ratio over 9000!\n";
			scenes.push_back(second_frame.clone());
		} else if(ratio < FX_LIMIT) {
			if(saved_dilated_buffer) {

				const double saved_ratio = edge_ratio_omp(*saved_dilated_buffer, second_thresh_res);

				if(saved_ratio < LIMIT) {
					std::cout << "saved ratio over 9000!\n";
					scenes.push_back(second_frame.clone());
					saved_dilated_buffer = std::nullopt;
				}

			} else {
				std::cout << "Saving current dilated frame\n";
				saved_dilated_buffer = first_dilation_res.get().clone();
			}
		} else if(saved_dilated_buffer) {
			std::cout << "Abandon saved dilated frame\n";
			saved_dilated_buffer = std::nullopt;
		}

		first_thresh_res = second_thresh_res;
		input_video >> second_frame;
	}
	std::cout << "[1000D[" << std::string(PROGRESS_BAR_SIZE, '#') << "]" << std::flush;
	std::cout << "\n" << std::endl;

	for(auto const& scene : scenes) {
		cv::imshow("Scene", scene);
		cv::waitKey(0);
	}
}
