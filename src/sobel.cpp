#include "sobel.hpp"
#include <cmath>

const cv::Mat_<int8_t> sobelx = (cv::Mat_<int8_t>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
const cv::Mat_<int8_t> sobely = (cv::Mat_<int8_t>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

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

			FGRGB.at<cv::Vec3b>(x - 1, y - 1)[0] = blueFG;
			FGRGB.at<cv::Vec3b>(x - 1, y - 1)[1] = greenFG;
			FGRGB.at<cv::Vec3b>(x - 1, y - 1)[2] = redFG;
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
