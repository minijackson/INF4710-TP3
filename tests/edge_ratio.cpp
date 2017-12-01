#include <gtest/gtest.h>

#include "../src/edge_ratio.hpp"

cv::Mat_<uint8_t> test1 = (cv::Mat_<uint8_t>(2, 3) << 0, 0, 0, 0, 0, 0),
                  test2 = (cv::Mat_<uint8_t>(2, 3) << 255, 0, 255, 0, 0, 0),
                  test3 = (cv::Mat_<uint8_t>(2, 3) << 255, 0, 0, 0, 0, 0),
                  test4 = (cv::Mat_<uint8_t>(2, 3) << 255, 255, 255, 255, 0, 0);

TEST(EdgeRatio, MonoThreaded) {
	EXPECT_EQ(edge_ratio(test1, test1), 1);
	EXPECT_EQ(edge_ratio(test1, test2), 0);
	EXPECT_EQ(edge_ratio(test1, test3), 0);
	EXPECT_EQ(edge_ratio(test1, test4), 0);

	EXPECT_EQ(edge_ratio(test2, test2), 1);
	EXPECT_EQ(edge_ratio(test2, test3), 1);
	EXPECT_EQ(edge_ratio(test2, test4), 0.5);

	EXPECT_EQ(edge_ratio(test3, test3), 1);
	EXPECT_EQ(edge_ratio(test3, test4), 0.25);

	EXPECT_EQ(edge_ratio(test4, test1), 1);
	EXPECT_EQ(edge_ratio(test4, test2), 1);
	EXPECT_EQ(edge_ratio(test4, test3), 1);
	EXPECT_EQ(edge_ratio(test4, test4), 1);
}

TEST(EdgeRatio, GnuParallel) {
	EXPECT_EQ(edge_ratio_omp(test1, test1), 1);
	EXPECT_EQ(edge_ratio_omp(test1, test2), 0);
	EXPECT_EQ(edge_ratio_omp(test1, test3), 0);
	EXPECT_EQ(edge_ratio_omp(test1, test4), 0);

	EXPECT_EQ(edge_ratio_omp(test2, test2), 1);
	EXPECT_EQ(edge_ratio_omp(test2, test3), 1);
	EXPECT_EQ(edge_ratio_omp(test2, test4), 0.5);

	EXPECT_EQ(edge_ratio_omp(test3, test3), 1);
	EXPECT_EQ(edge_ratio_omp(test3, test4), 0.25);

	EXPECT_EQ(edge_ratio_omp(test4, test1), 1);
	EXPECT_EQ(edge_ratio_omp(test4, test2), 1);
	EXPECT_EQ(edge_ratio_omp(test4, test3), 1);
	EXPECT_EQ(edge_ratio_omp(test4, test4), 1);
}

