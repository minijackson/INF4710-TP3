#include "common.hpp"

#include "gtest/gtest.h"

#include "../src/sobel.hpp"

TEST(Sobel, MonoThreaded) {

	//Mat lena = imread("../data/lena.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat_<cv::Vec3b> lena =  cv::imread("../data/lena.png", cv::IMREAD_COLOR);


	cv::Mat_<cv::Vec3b> expectedFastResult =  (cv::Mat_<cv::Vec3b>(5,5) <<	cv::Vec3b{0, 0, 0}, cv::Vec3b{0 , 0 , 0 }, cv::Vec3b{0 , 0 , 0 }, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0},
																			cv::Vec3b{0, 0, 0}, cv::Vec3b{40, 40, 40}, cv::Vec3b{40, 40, 40}, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0},
																			cv::Vec3b{0, 0, 0}, cv::Vec3b{40, 40, 40}, cv::Vec3b{40, 40, 40}, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0},
																			cv::Vec3b{0, 0, 0}, cv::Vec3b{40, 40, 40}, cv::Vec3b{40, 40, 40}, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0},
																			cv::Vec3b{0, 0, 0}, cv::Vec3b{0 , 0 , 0 }, cv::Vec3b{0 , 0 , 0 }, cv::Vec3b{0, 0, 0}, cv::Vec3b{0, 0, 0});

	cv::Mat_<int8_t> test = (cv::Mat_<int8_t>(5, 5) <<		0, 0, 10, 10, 10,
															0, 0, 10, 10, 10,
															0, 0, 10, 10, 10,
															0, 0, 10, 10, 10,
															0, 0, 10, 10, 10);

	cv::Mat_<int8_t> testbis = (cv::Mat_<int8_t>(5, 5) <<	1, 1, 1, 1, 1,
															1, 1, 1, 1, 1,
															1, 1, 1, 1, 1,
															1, 1, 1, 1, 1,
															1, 1, 1, 1, 1);

	cv::Mat_<int8_t> gx = (cv::Mat_<int8_t>(5, 5) <<	0, 30, 30, 0, -30,
														0, 40, 40, 0, -40,
														0, 40, 40, 0, -40,
														0, 40, 40, 0, -40,
														0, 30, 30, 0, -30);

	cv::Mat_<int8_t> gy = (cv::Mat_<int8_t>(5, 5) <<	0, -10, -30, -40, -30,
														0,	 0,	  0,   0,   0,
														0,   0,   0,   0,   0,
														0,   0,   0,   0,   0,
														0, -10, -30, -40, -30);

	cv::Mat_<int8_t> zeroRow = (cv::Mat_<int8_t>(1,7) << 0, 0, 0, 0, 0, 0, 0);
	cv::Mat_<int8_t> zeroCol = (cv::Mat_<int8_t>(5,1) << 0, 0, 0, 0, 0);

	cv::Mat_<cv::Vec3b> expectedResult(5, 5);

	for(int row = 0; row < 5; ++row) {
		for(int col = 0; col < 5; ++col) {
			expectedResult.at<cv::Vec3b>(row, col)[0] = std::sqrt(gx(row, col) * gx(row, col) + gy(row, col) * gy(row, col));
			expectedResult.at<cv::Vec3b>(row, col)[1] = std::sqrt(gx(row, col) * gx(row, col) + gy(row, col) * gy(row, col));
			expectedResult.at<cv::Vec3b>(row, col)[2] = std::sqrt(gx(row, col) * gx(row, col) + gy(row, col) * gy(row, col));
		}
	}

	cv::Mat_<cv::Vec3b> testIMG(5, 5);

	for(int row = 0; row < 5; ++row)
	{
		for(int col = 0; col < 5; ++col)
		{
			testIMG.at<cv::Vec3b>(row, col)[0] = test(row, col);
			testIMG.at<cv::Vec3b>(row, col)[1] = test(row, col);
			testIMG.at<cv::Vec3b>(row, col)[2] = test(row, col);
		}
	}

	cv::Mat_<cv::Vec3b> myResult = sobel(testIMG);
	cv::Mat_<cv::Vec3b> myFastResult = fastsobel(testIMG);

	//std::cout << "Résultats obtenus :\n" << myResult << "\n";

	//std::cout << "Résultats attendus :\n" << expectedResult << "\n";

	//std::cout << "Résultats rapides obtenus :\n" << myFastResult << "\n";

	//std::cout << "Résultats rapides attendus :\n" << expectedFastResult << std::endl;

	EXPECT_TRUE(assert_mat_equal(myResult, expectedResult));
	EXPECT_TRUE(assert_mat_equal(myFastResult, expectedFastResult));

	cv::imshow("lenasobel", sobel(lena));
	cv::waitKey(0);
}
