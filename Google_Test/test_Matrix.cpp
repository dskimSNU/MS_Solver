#pragma once
#include "gtest/gtest.h"
#include "../MS_Solver/INC/Matrix.h"

TEST(Matrix, construct_diagonal_matrix_1) {
	std::array<double, 2> ar = { 2,2 };
	Matrix<2, 2> result = ar;

	Matrix<2, 2> ref = { 2,0,0,2 };
	EXPECT_EQ(result, ref);
}
TEST(Matrix, construct_diagonal_matrix_2) {
	std::array<double, 2> ar = { 2,2 };
	Matrix result = ar;

	Matrix<2, 2> ref = { 2,0,0,2 };
	EXPECT_EQ(result, ref);
}

TEST(Matrix, inverse_1) {
	Matrix<2,2> m = { 1,2,3,4 };
	m.be_inverse();

	Matrix<2, 2> ref = { -2,1,1.5,-0.5 };

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			EXPECT_DOUBLE_EQ(m.at(i, j), ref.at(i, j));
}


TEST(Matrix, operator_plus_1) {
	std::array<double, 16> ar;
	ar.fill(1);

	const Matrix<4, 4> m1 = ar;
	const Matrix<4, 4> m2 = ar;
	const auto result = m1 + m2;

	std::array<double, 16> ref_val;
	ref_val.fill(2);

	const Matrix<4, 4> ref = ref_val;
	EXPECT_EQ(result, ref);
}
TEST(Matrix, operator_plus_2) {
	std::array<double, 36> ar;
	ar.fill(1);

	const Matrix<6, 6> m1 = ar;
	const Matrix<6, 6> m2 = ar;
	const auto result = m1 + m2;

	std::array<double, 36> ref_val;
	ref_val.fill(2);

	const Matrix<6, 6> ref = ref_val;
	EXPECT_EQ(result, ref);
}

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

TEST(Matrix, scalar_multiplication_1) {
	std::array<double, 9> ar;
	ar.fill(1);
	
	const Matrix<3, 3> m(ar);
	const auto result = m * 10;

	std::array<double, 9> ref_val;
	ref_val.fill(10);

	const Matrix<3, 3> ref(ref_val);
	EXPECT_EQ(result, ref);
}
TEST(Matrix, scalar_multiplication_2) {
	std::array<double, 100> ar;
	ar.fill(1);

	const Matrix<10, 10> m(ar);
	const auto result = m * 10;

	std::array<double, 100> ref_val;
	ref_val.fill(10);

	const Matrix<10, 10> ref(ref_val);
	EXPECT_EQ(result, ref);
}

TEST(Matrix, gemm_1) {
	Matrix<1, 2> m = { 1,2 };
	const Matrix<2, 2> m2 = { 2,2,2,2 };
	m *= m2;

	const Matrix<1, 2> ref = { 6,6 };
	EXPECT_EQ(m, ref);
}
TEST(Matrix, gemm_2) {
	std::array<double, 3> ar = { 1, 0, 0 };
	Matrix m = ar;

	const Matrix<3, 3> m2 = { 1,2,3,4,5,6,7,8,9 };
	const auto result = m * m2;

	const Matrix<3, 3> ref = { 1,2,3,0,0,0,0,0,0 };
	EXPECT_EQ(result, ref);
}
TEST(Matrix, gemm_3) {
	std::array<double, 3> ar = { 1, 0, 0 };
	Matrix m = ar;

	const Matrix<3, 3> m2 = { 1,2,3,4,5,6,7,8,9 };
	const auto result = m2 * m;

	const Matrix<3, 3> ref = { 1,0,0,4,0,0,7,0,0 };
	EXPECT_EQ(result, ref);
}

TEST(Matrix, column_1) {
	const Matrix<2, 3> m = { 1,2,3,4,5,6 };
	const auto result = m.column(0);

	const Euclidean_Vector ref = { 1,4 };
	EXPECT_EQ(result, ref);
}
TEST(Matrix, column_2) {
	const Matrix<2, 3> m = { 1,2,3,4,5,6 };
	const auto result = m.column(2);

	const Euclidean_Vector ref = { 3,6 };
	EXPECT_EQ(result, ref);
}

TEST(Matrix, change_columns_1){
	Matrix<2, 3> m = { 1,2,3,4,5,6 };
	const Matrix<2, 2> m1 = { 1,1,4,4 };
	m.change_columns(1, m1);

	const Matrix<2, 3> ref = { 1,1,1,4,4,4 };
	EXPECT_EQ(m, ref);
}
TEST(Matrix, change_columns_2) {
	Matrix<2, 3> m = { 1,2,3,4,5,6 };
	const Dynamic_Matrix m1 = { 2,2,{ 1,1,4,4 } };
	m.change_columns(1, m1);

	const Matrix<2, 3> ref = { 1,1,1,4,4,4 };
	EXPECT_EQ(m, ref);
}

TEST(Dynamic_Matrix, construct_diagonal_matrix_1) {
	Dynamic_Matrix m(2, {3, 3});
	Dynamic_Matrix ref(2, 2, { 3,0,0,3 });

	EXPECT_EQ(m, ref);
}

TEST(Dynamic_Matrix, change_row_1) {
	Dynamic_Matrix m(2, 2, { 1,2,3,4 });

	Euclidean_Vector v = { 5,6 };
	m.change_row(0, v);

	Dynamic_Matrix ref(2, 2, {5, 6, 3, 4});
	EXPECT_EQ(m, ref);
}
TEST(Dynamic_Matrix, change_row_2) {
	Dynamic_Matrix m(2, 2, { 1,2,3,4 });

	Dynamic_Euclidean_Vector v = { 5,6 };
	m.change_row(0, v);

	Dynamic_Matrix ref(2, 2, { 5, 6, 3, 4 });
	EXPECT_EQ(m, ref);
}

TEST(Dynamic_Matrix, change_rows_1) {
	const Matrix<2, 2> m = { 1,2,3,4 };

	Dynamic_Matrix dm(4, 2);
	dm.change_rows(0, m);

	const Dynamic_Matrix ref(4, 2, { 1,2,3,4,0,0,0,0 });
	EXPECT_EQ(dm, ref);
}

TEST(Dynamic_Matrix, change_columns_1) {
	const Matrix<2, 2> m = { 1,2,3,4 };

	Dynamic_Matrix dm(2, 4);
	dm.change_columns(0, m);

	const Dynamic_Matrix ref(2, 4, { 1,2,0,0,3,4,0,0 });
	EXPECT_EQ(dm, ref);
}

TEST(Dynamic_Matrix, row_1) {
	Dynamic_Matrix dm(2, 3, { 1,2,3,4,5,6 });
	const auto result = dm.row(0);

	const Dynamic_Euclidean_Vector ref = { 1,2,3 };
	EXPECT_EQ(result, ref);
}