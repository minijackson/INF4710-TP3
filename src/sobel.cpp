#include "sobel.hpp"

#include "opencl.hpp"

#include <cmath>

const cv::Mat_<int8_t> sobelx = (cv::Mat_<int8_t>(3,3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
const cv::Mat_<int8_t> sobely = (cv::Mat_<int8_t>(3,3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

void initializeBordersToZero(cv::Mat_<cv::Vec3b>& FGRGB) {
	int rowMax = FGRGB.rows-1;
	int colMax = FGRGB.cols-1;

	for(int row = 0; row < FGRGB.rows; ++row) {
		for(int i = 0; i < 3; ++i) {

			FGRGB.at<cv::Vec3b>(row, 0)[i]      = 0;
			FGRGB.at<cv::Vec3b>(row, colMax)[i] = 0;
		}
	}

	for(int col = 0; col < FGRGB.cols; ++col) {
		for(int i = 0; i < 3; ++i) {
			FGRGB.at<cv::Vec3b>(0, col)[i]      = 0;
			FGRGB.at<cv::Vec3b>(rowMax, col)[i] = 0;
		}
	}
}

cv::Mat_<cv::Vec3b> sobel(cv::Mat_<cv::Vec3b> RGB) {

	cv::Mat_<cv::Vec3b> FGRGB(RGB.rows, RGB.cols);

	double redFG, greenFG, blueFG;
	int redGx, redGy, greenGx, greenGy, blueGx, blueGy;

	cv::Vec3b nullVec{0, 0, 0};
	cv::Mat_<cv::Vec3b> zeroCol(RGB.rows,1, nullVec);
	cv::Mat_<cv::Vec3b> zeroRow(1, RGB.cols + 2, nullVec);

	// Exemples concaténation horizontale et verticale
	// hconcat : https://docs.opencv.org/trunk/d2/de8/group__core__array.html#gaab5ceee39e0580f879df645a872c6bf7
	// vconcat : https://docs.opencv.org/trunk/d2/de8/group__core__array.html#ga744f53b69f6e4f12156cdde4e76aed27

	cv::hconcat(zeroCol, RGB, RGB);
	cv::hconcat(RGB, zeroCol, RGB);
	cv::vconcat(zeroRow, RGB, RGB);
	cv::vconcat(RGB, zeroRow, RGB);

	for(int x = 1; x < RGB.rows - 1; ++x) {
		for(int y = 1; y < RGB.cols - 1; ++y) {

			redGx =		(sobelx(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[2]) + (sobelx(0,1) * RGB.at<cv::Vec3b>(x,y-1)[2]) + (sobelx(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[2])	+
						(sobelx(1,0) * RGB.at<cv::Vec3b>(x-1,y)[2]	) + (sobelx(1,1) * RGB.at<cv::Vec3b>(x,y)[2]  ) + (sobelx(1,2) * RGB.at<cv::Vec3b>(x+1,y)[2]  )	+
						(sobelx(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[2]) + (sobelx(2,1) * RGB.at<cv::Vec3b>(x,y+1)[2]) + (sobelx(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[2]);

			redGy =		(sobely(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[2]) + (sobely(0,1) * RGB.at<cv::Vec3b>(x,y-1)[2]) + (sobely(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[2]) +
						(sobely(1,0) * RGB.at<cv::Vec3b>(x-1,y)[2]  ) + (sobely(1,1) * RGB.at<cv::Vec3b>(x,y)[2]  ) + (sobely(1,2) * RGB.at<cv::Vec3b>(x+1,y)[2]  ) +
						(sobely(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[2]) + (sobely(2,1) * RGB.at<cv::Vec3b>(x,y+1)[2]) + (sobely(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[2]);

			greenGx =	(sobelx(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[1]) + (sobelx(0,1) * RGB.at<cv::Vec3b>(x,y-1)[1]) + (sobelx(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[1])	+
						(sobelx(1,0) * RGB.at<cv::Vec3b>(x-1,y)[1]  ) + (sobelx(1,1) * RGB.at<cv::Vec3b>(x,y)[1]  ) + (sobelx(1,2) * RGB.at<cv::Vec3b>(x+1,y)[1]  )	+
						(sobelx(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[1]) + (sobelx(2,1) * RGB.at<cv::Vec3b>(x,y+1)[1]) + (sobelx(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[1]);

			greenGy =	(sobely(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[1]) + (sobely(0,1) * RGB.at<cv::Vec3b>(x,y-1)[1]) + (sobely(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[1])	+
						(sobely(1,0) * RGB.at<cv::Vec3b>(x-1,y)[1]  ) + (sobely(1,1) * RGB.at<cv::Vec3b>(x,y)[1]  ) + (sobely(1,2) * RGB.at<cv::Vec3b>(x+1,y)[1]  )	+
						(sobely(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[1]) + (sobely(2,1) * RGB.at<cv::Vec3b>(x,y+1)[1]) + (sobely(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[1]);

			blueGx =	(sobelx(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[0]) + (sobelx(0,1) * RGB.at<cv::Vec3b>(x,y-1)[0]) + (sobelx(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[0])	+
						(sobelx(1,0) * RGB.at<cv::Vec3b>(x-1,y)[0]  ) + (sobelx(1,1) * RGB.at<cv::Vec3b>(x,y)[0]  ) + (sobelx(1,2) * RGB.at<cv::Vec3b>(x+1,y)[0]  ) +
						(sobelx(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[0]) + (sobelx(2,1) * RGB.at<cv::Vec3b>(x,y+1)[0]) + (sobelx(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[0]);

			blueGy =	(sobely(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[0]) + (sobely(0,1) * RGB.at<cv::Vec3b>(x,y-1)[0]) + (sobely(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[0])	+
						(sobely(1,0) * RGB.at<cv::Vec3b>(x-1,y)[0]  ) + (sobely(1,1) * RGB.at<cv::Vec3b>(x,y)[0]  ) + (sobely(1,2) * RGB.at<cv::Vec3b>(x+1,y)[0]  )	+
						(sobely(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[0]) + (sobely(2,1) * RGB.at<cv::Vec3b>(x,y+1)[0]) + (sobely(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[0]);

			redFG   = std::sqrt(redGx	* redGx		+ redGy		* redGy);
			greenFG = std::sqrt(greenGx * greenGx	+ greenGy	* greenGy);
			blueFG	= std::sqrt(blueGx	* blueGx	+ blueGy	* blueGy);

			if(redFG > 255)		FGRGB.at<cv::Vec3b>(x - 1, y - 1)[2] = 255;
			else				FGRGB.at<cv::Vec3b>(x - 1, y - 1)[2] = redFG;
			if(greenFG > 255)	FGRGB.at<cv::Vec3b>(x - 1, y - 1)[1] = 255;
			else				FGRGB.at<cv::Vec3b>(x - 1, y - 1)[1] = greenFG;
			if(blueFG > 255)	FGRGB.at<cv::Vec3b>(x - 1, y - 1)[0] = 255;
			else				FGRGB.at<cv::Vec3b>(x - 1, y - 1)[0] = blueFG;
		}
	}
	return FGRGB;
}

cv::Mat_<cv::Vec3b> fastsobel(cv::Mat_<cv::Vec3b> RGB) {

	cv::Mat_<cv::Vec3b> FGRGB(RGB.rows, RGB.cols);

	double redFG, greenFG, blueFG;
	int redGx, redGy, greenGx, greenGy, blueGx, blueGy;

	initializeBordersToZero(FGRGB);

	for(int x = 1; x < RGB.rows - 1; ++x) {
		for(int y = 1; y < RGB.cols - 1; ++y) {

			redGx =		(sobelx(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[2]) + (sobelx(0,1) * RGB.at<cv::Vec3b>(x,y-1)[2]) + (sobelx(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[2])	+
						(sobelx(1,0) * RGB.at<cv::Vec3b>(x-1,y)[2]	) + (sobelx(1,1) * RGB.at<cv::Vec3b>(x,y)[2]  ) + (sobelx(1,2) * RGB.at<cv::Vec3b>(x+1,y)[2]  )	+
						(sobelx(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[2]) + (sobelx(2,1) * RGB.at<cv::Vec3b>(x,y+1)[2]) + (sobelx(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[2]);

			redGy =		(sobely(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[2]) + (sobely(0,1) * RGB.at<cv::Vec3b>(x,y-1)[2]) + (sobely(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[2]) +
						(sobely(1,0) * RGB.at<cv::Vec3b>(x-1,y)[2]  ) + (sobely(1,1) * RGB.at<cv::Vec3b>(x,y)[2]  ) + (sobely(1,2) * RGB.at<cv::Vec3b>(x+1,y)[2]  ) +
						(sobely(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[2]) + (sobely(2,1) * RGB.at<cv::Vec3b>(x,y+1)[2]) + (sobely(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[2]);

			greenGx =	(sobelx(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[1]) + (sobelx(0,1) * RGB.at<cv::Vec3b>(x,y-1)[1]) + (sobelx(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[1])	+
						(sobelx(1,0) * RGB.at<cv::Vec3b>(x-1,y)[1]  ) + (sobelx(1,1) * RGB.at<cv::Vec3b>(x,y)[1]  ) + (sobelx(1,2) * RGB.at<cv::Vec3b>(x+1,y)[1]  )	+
						(sobelx(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[1]) + (sobelx(2,1) * RGB.at<cv::Vec3b>(x,y+1)[1]) + (sobelx(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[1]);

			greenGy =	(sobely(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[1]) + (sobely(0,1) * RGB.at<cv::Vec3b>(x,y-1)[1]) + (sobely(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[1])	+
						(sobely(1,0) * RGB.at<cv::Vec3b>(x-1,y)[1]  ) + (sobely(1,1) * RGB.at<cv::Vec3b>(x,y)[1]  ) + (sobely(1,2) * RGB.at<cv::Vec3b>(x+1,y)[1]  )	+
						(sobely(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[1]) + (sobely(2,1) * RGB.at<cv::Vec3b>(x,y+1)[1]) + (sobely(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[1]);	

			blueGx =	(sobelx(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[0]) + (sobelx(0,1) * RGB.at<cv::Vec3b>(x,y-1)[0]) + (sobelx(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[0])	+
						(sobelx(1,0) * RGB.at<cv::Vec3b>(x-1,y)[0]  ) + (sobelx(1,1) * RGB.at<cv::Vec3b>(x,y)[0]  ) + (sobelx(1,2) * RGB.at<cv::Vec3b>(x+1,y)[0]  ) +
						(sobelx(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[0]) + (sobelx(2,1) * RGB.at<cv::Vec3b>(x,y+1)[0]) + (sobelx(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[0]);

			blueGy =	(sobely(0,0) * RGB.at<cv::Vec3b>(x-1,y-1)[0]) + (sobely(0,1) * RGB.at<cv::Vec3b>(x,y-1)[0]) + (sobely(0,2) * RGB.at<cv::Vec3b>(x+1,y-1)[0])	+
						(sobely(1,0) * RGB.at<cv::Vec3b>(x-1,y)[0]  ) + (sobely(1,1) * RGB.at<cv::Vec3b>(x,y)[0]  ) + (sobely(1,2) * RGB.at<cv::Vec3b>(x+1,y)[0]  )	+
						(sobely(2,0) * RGB.at<cv::Vec3b>(x-1,y+1)[0]) + (sobely(2,1) * RGB.at<cv::Vec3b>(x,y+1)[0]) + (sobely(2,2) * RGB.at<cv::Vec3b>(x+1,y+1)[0]);

			redFG   = std::sqrt(redGx	* redGx		+ redGy		* redGy);
			greenFG = std::sqrt(greenGx * greenGx	+ greenGy	* greenGy);
			blueFG	= std::sqrt(blueGx	* blueGx	+ blueGy	* blueGy);

			FGRGB.at<cv::Vec3b>(x, y)[0] = blueFG;
			FGRGB.at<cv::Vec3b>(x, y)[1] = greenFG;
			FGRGB.at<cv::Vec3b>(x, y)[2] = redFG;
		}
	}
	return FGRGB;
}

CLMat<cv::Vec4b> sobel_cl(cv::Mat_<cv::Vec4b> const& input) {
	// Create buffers that represents variables in the device's side
	cl::Image2D input_buffer(cl_singletons::context,
			CL_MEM_READ_ONLY,
			cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
			input.rows,
			input.cols,
			/* image_row_pitch = */ 0,
			input.data);

	cl::Image2D output_buffer(cl_singletons::context,
			CL_MEM_WRITE_ONLY,
			cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
			input.rows,
			input.cols,
			/* image_row_pitch = */ 0);

	cl::size_t<3> origin;
	origin.push_back(0);
	origin.push_back(0);
	origin.push_back(0);

	cl::size_t<3> region;
	region.push_back(input.rows);
	region.push_back(input.cols);
	region.push_back(1);

	// Send the image to the device
	cl_singletons::queue.enqueueWriteImage(input_buffer,
			/* blocking = */ CL_FALSE,
			origin,
			region,
			/* row_pitch = */ 0,
			/* slice_pitch = */ 0,
			input.data);

	// Create a functor (object that can be called) that will "call" the OpenCL function
	cl::KernelFunctor sobel_cl_functor(
			cl_singletons::sobel_kernel,
			cl_singletons::queue,
			/* global_work_offset = */ cl::NullRange,
			/* global_work_size = */ cl::NDRange(input.rows, input.cols),
			/* local_work_size = */ cl::NullRange);

	// Call it
	sobel_cl_functor(input_buffer, output_buffer);

	cv::Mat_<uint8_t> output;
	output.create(input.rows, input.cols);

	return CLMat<cv::Vec4b>(input.rows, input.cols, output_buffer);
}
