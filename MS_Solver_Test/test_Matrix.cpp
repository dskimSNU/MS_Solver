#pragma once
#include "gtest/gtest.h"
#include "../MS_Solver/INC/Matrix.h"

GTEST_TEST(Matrix, operator_mv_1) {
	const Euclidean_Vector v = { 1,2 };
	const Matrix<2,2> m = { 1,2,3,4 };
	const auto result = m * v;

	const Euclidean_Vector ref = { 5,11 };
	EXPECT_EQ(result, ref);
}

GTEST_TEST(Matrix, operator_mv_2) {
	const Matrix<1, 2> m = { 1,1 };
	const Euclidean_Vector v = { 1,2 };
	
	const auto result = (m+m) * v;

	const Euclidean_Vector ref = 6;
	EXPECT_EQ(result, ref);
}