#pragma once
#include "Euclidean_Vector.h"

class Dynamic_Matrix;


// Matrix template class
template<size_t num_row, size_t num_column>
class Matrix
{
private:
	template<size_t num_other_row, size_t num_other_column>
	friend class Matrix;
	friend class Dynamic_Matrix;
	friend class ms::BLAS;

	static constexpr size_t num_value_ = num_row * num_column;

private:
	std::array<double, num_value_> values_ = { 0 };

public:
	Matrix(void) = default;				
	Matrix(const std::array<double, num_value_>& values) : values_{ values } {};
	Matrix(const Dynamic_Matrix& dynamic_matrix);

	template <typename... Args>
	Matrix(Args... args);

	// diagonal matrix constructor
	// SFINAE to resolve 1x1 matrix constructor ambiguity
	template<size_t temp = num_row * num_column, std::enable_if_t<temp != 1, bool> = true>
	Matrix(const std::array<double, num_row>& values);

public:
	Matrix& operator+=(const Matrix& A);
	Matrix& operator-=(const Matrix& A);
	Matrix& operator*=(const double scalar);
	Matrix& operator*=(const Matrix<num_column, num_column>& A);

	Matrix operator+(const Matrix& A) const;
	Matrix operator*(const double scalar) const;
	Euclidean_Vector<num_row> operator*(const Euclidean_Vector<num_column>& x) const;
	Dynamic_Matrix operator*(const Dynamic_Matrix& dynamic_matrix) const;

	template <size_t other_num_column>
	Matrix<num_row, other_num_column> operator*(const Matrix<num_column, other_num_column>& A) const;

	bool operator==(const Matrix & A) const;

public:
	Matrix& be_inverse(void) {
		static_require(num_row == num_column, "invertable matrix should be square matrix");

		const auto ipiv = this->PLU_decomposition();

		const int matrix_layout = LAPACK_ROW_MAJOR;
		const lapack_int n = static_cast<int>(num_row);
		const lapack_int lda = n;
		const lapack_int info = LAPACKE_dgetri(matrix_layout, n, this->values_.data(), lda, ipiv.data());

		dynamic_require(info == 0, "info should be 0 when success matrix inverse");
		//info > 0 "U is singular matrix in L-U decomposition"
		//info < 0 "fail to inverse the matrix"

		return *this;
	}
	Matrix inverse(void) const {
		auto result = *this;
		return result.be_inverse();
	}

	void change_columns(const size_t column_index, const Dynamic_Matrix& A);

	template <size_t num_other_column>
	void change_columns(const size_t column_index, const Matrix<num_row, num_other_column>& A);

public:
	double at(const size_t row_index, const size_t column_index) const;
	Euclidean_Vector<num_row> column(const size_t column_index) const;
	std::string to_string(void) const;

private:
	double& value_at(const size_t row_index, const size_t column_index);
	std::vector<int> PLU_decomposition(void) {
		const int matrix_layout = LAPACK_ROW_MAJOR;
		const lapack_int m = static_cast<int>(num_row);
		const lapack_int n = static_cast<int>(num_column);
		const lapack_int lda = n;
		std::vector<int> ipiv(std::min(m, n));
		lapack_int info = LAPACKE_dgetrf(matrix_layout, m, n, this->values_.data(), lda, ipiv.data());

		dynamic_require(0 <= info, "info should be greater than 0 when sucess PLU decomposition");
		return ipiv;
	};
};

template <size_t matrix_order>
Matrix(const std::array<double, matrix_order>& values)->Matrix<matrix_order, matrix_order>;

template<size_t num_row, size_t num_column>
Matrix<num_row, num_column> operator*(const double scalar, const Matrix<num_row, num_column>& A);

template<size_t num_row, size_t num_column>
std::ostream& operator<<(std::ostream& os, const Matrix<num_row, num_column>& m);


// Dynamic matrix 
class Dynamic_Matrix
{
private:
	template<size_t num_row, size_t num_column>
	friend class Matrix;
	friend class ms::BLAS;


private:
	CBLAS_TRANSPOSE transpose_type_ = CBLAS_TRANSPOSE::CblasNoTrans;
	size_t num_row_ = 0;
	size_t num_column_ = 0;
	std::vector<double> values_;

public:
	Dynamic_Matrix(const size_t matrix_order);
	Dynamic_Matrix(const size_t matrix_order, const std::vector<double>& value);
	Dynamic_Matrix(const size_t num_row, const size_t num_column);
	Dynamic_Matrix(const size_t num_row, const size_t num_column, std::vector<double>&& value);

public:
	Dynamic_Matrix operator*(const Dynamic_Matrix& other) const;
	bool operator==(const Dynamic_Matrix& other) const;
		
public:
	double at(const size_t row, const size_t column) const;
	Dynamic_Matrix transpose(void) const;
	std::string to_string(void) const;
	Dynamic_Matrix inverse(void) const;
	std::pair<size_t, size_t> size(void) const;
	bool is_finite(void) const;

public:
	Dynamic_Matrix& be_transpose(void);
	Dynamic_Matrix& be_inverse(void);
	void change_column(const size_t column_index, const Dynamic_Euclidean_Vector& vec);
	void change_row(const size_t start_row_index, const Dynamic_Euclidean_Vector& vec);
	void change_rows(const size_t start_row_index, const Dynamic_Matrix& A);

public:
	template <size_t num_row>
	Euclidean_Vector<num_row> column(const size_t column_index) const;

	Dynamic_Euclidean_Vector row(const size_t row_index) const;

	template <size_t dimension>
	void change_row(const size_t row_index, const Euclidean_Vector<dimension>& vec);

	template <size_t dimension>
	void change_column(const size_t column_index, const Euclidean_Vector<dimension>& vec);

	template <size_t num_row, size_t num_column>
	void change_rows(const size_t start_row_index, const Matrix<num_row, num_column>& A);

	template <size_t num_row, size_t num_column>
	void change_columns(const size_t start_column_index, const Matrix<num_row, num_column>& A);


private:
	bool is_square_matrix(void) const;
	bool is_transposed(void) const;
	bool is_in_range(const size_t irow, const size_t jcolumn) const;
	size_t leading_dimension(void) const;

	double& value_at(const size_t row, const size_t column);
	std::vector<int> PLU_decomposition(void);
};


std::ostream& operator<<(std::ostream& os, const Dynamic_Matrix& m);


namespace ms {
	class BLAS
	{
	private:
		BLAS(void) = delete;

	public:
		template<size_t num_row, size_t num_column>
		static void gemm(const Dynamic_Matrix& A, const Dynamic_Matrix& B, Matrix<num_row, num_column>& C) { BLAS::gemm(A, B, C.values_.data()); };
		static void gemm(const Dynamic_Matrix& A, const Dynamic_Matrix& B, double* output_ptr);
		static void gemvpv(const Dynamic_Matrix& A, const Dynamic_Euclidean_Vector& v1, Dynamic_Euclidean_Vector& v2);
	};

	inline constexpr ushort blas_dscal_criteria = 10;
	inline constexpr ushort blas_mv_criteria = 50;

	inline void gemm(const Dynamic_Matrix& A, const Dynamic_Matrix& B, double* output_ptr) {
		ms::BLAS::gemm(A, B, output_ptr);
	}

	inline void gemvpv(const Dynamic_Matrix& A, const Dynamic_Euclidean_Vector& v1, Dynamic_Euclidean_Vector& v2) {
		ms::BLAS::gemvpv(A, v1, v2);
	}

	template<size_t num_row, size_t num_column>
	void gemm(const Dynamic_Matrix& A, const Dynamic_Matrix& B, Matrix<num_row, num_column>& result) {
		ms::BLAS::gemm(A, B, result);
	}
}


//template definition part
template<size_t num_row, size_t num_column>
template<size_t temp, std::enable_if_t<temp != 1, bool>>
Matrix<num_row, num_column>::Matrix(const std::array<double, num_row>& values) {
	static_require(num_row == num_column, "It should be square matrix");
	for (size_t i = 0; i < num_row; ++i)
		this->value_at(i, i) = values[i];
}

template<size_t num_row, size_t num_column>
Matrix<num_row,num_column>::Matrix(const Dynamic_Matrix& dynamic_matrix) {
	dynamic_require(dynamic_matrix.size() == std::make_pair(num_row, num_column), "both matrix should be same size");
	std::copy(dynamic_matrix.values_.begin(), dynamic_matrix.values_.end(), this->values_.begin());
}

template<size_t num_row, size_t num_column>
template <typename... Args>
Matrix<num_row, num_column>::Matrix(Args... args) : values_{ static_cast<double>(args)... } {
	static_require(sizeof...(Args) == this->num_value_, "Number of arguments should be same with (num row * num column)");
	static_require(ms::are_arithmetics<Args...>, "every arguments should be arithmetics");
};

template<size_t num_row, size_t num_column>
Matrix<num_row, num_column>& Matrix<num_row, num_column>::operator+=(const Matrix& A) {
	if constexpr (this->num_value_ < ms::blas_axpy_criteria) {
		for (size_t i = 0; i < this->num_value_; ++i)
			this->values_[i] += A.values_[i];
	}
	else 
		cblas_daxpy(this->num_value_, 1.0, A.values_.data(), 1, this->values_.data(), 1);

	return *this;
}

template<size_t num_row, size_t num_column>
Matrix<num_row, num_column>& Matrix<num_row, num_column>::operator-=(const Matrix& A) {
	if constexpr (this->num_value_ < ms::blas_axpy_criteria) {
		for (size_t i = 0; i < this->num_value_; ++i)
			this->values_[i] -= A.values_[i];
	}
	else
		cblas_daxpy(this->num_value_, -1.0, A.values_.data(), 1, this->values_.data(), 1);

	return *this;
}

template<size_t num_row, size_t num_column>
Matrix<num_row, num_column>& Matrix<num_row, num_column>::operator*=(const double scalar) {
	if constexpr (this->num_value_ < ms::blas_dscal_criteria) {
		for (size_t i = 0; i < num_row * num_column; ++i)
			this->values_[i] *= scalar;
	}
	else
		cblas_dscal(this->num_value_, scalar, this->values_.data(), 1);

	return *this;
}

template<size_t num_row, size_t num_column>
Matrix<num_row, num_column>& Matrix<num_row, num_column>::operator*=(const Matrix<num_column, num_column>& A) {
	*this = (*this) * A;
	return *this;
}

template<size_t num_row, size_t num_column>
Matrix<num_row, num_column> Matrix<num_row, num_column>::operator+(const Matrix& A) const {	
	Matrix result = *this;
	return result += A;
}

template<size_t num_row, size_t num_column>
Matrix<num_row, num_column> Matrix<num_row, num_column>::operator*(const double scalar) const {
	Matrix result = *this;

	if constexpr (this->num_value_ < ms::blas_dscal_criteria) {
		for (size_t i = 0; i < num_row * num_column; ++i)
			result.values_[i] *= scalar;
	}
	else 
		cblas_dscal(this->num_value_, scalar, result.values_.data(), 1);

	return result;
}

template<size_t num_row, size_t num_column>
Dynamic_Matrix Matrix<num_row, num_column>::operator*(const Dynamic_Matrix& B) const {
	dynamic_require(num_column == B.num_row_, "dimension should be matched for matrix multiplication");

	const auto layout = CBLAS_LAYOUT::CblasRowMajor;
	const auto transA = CBLAS_TRANSPOSE::CblasNoTrans;
	const auto transB = B.transpose_type_;
	const auto m = static_cast<MKL_INT>(num_row);
	const auto n = static_cast<MKL_INT>(B.num_column_);
	const auto k = static_cast<MKL_INT>(num_column);
	const auto alpha = 1.0;
	const auto lda = static_cast<MKL_INT>(num_column);
	const auto ldb = static_cast<MKL_INT>(B.leading_dimension());
	const auto beta = 0.0;
	const auto ldc = n;

	Dynamic_Matrix result(m, n);
	cblas_dgemm(layout, transA, transB, m, n, k, alpha, this->values_.data(), lda, B.values_.data(), ldb, beta, result.values_.data(), ldc);

	return result;
}

template <size_t num_row, size_t num_column>
template <size_t other_num_column>
Matrix<num_row, other_num_column> Matrix<num_row, num_column>::operator*(const Matrix<num_column, other_num_column>& A) const {
	const auto layout = CBLAS_LAYOUT::CblasRowMajor;
	const auto transA = CBLAS_TRANSPOSE::CblasNoTrans;
	const auto transB = CBLAS_TRANSPOSE::CblasNoTrans;
	const auto m = static_cast<MKL_INT>(num_row);
	const auto n = static_cast<MKL_INT>(other_num_column);
	const auto k = static_cast<MKL_INT>(num_column);
	const double alpha = 1;
	const MKL_INT lda = static_cast<MKL_INT>(num_column);
	const MKL_INT ldb = static_cast<MKL_INT>(other_num_column);
	const double beta = 0;
	const MKL_INT ldc = n;

	Matrix<num_row, other_num_column> result;
	cblas_dgemm(layout, transA, transB, m, n, k, alpha, this->values_.data(), lda, A.values_.data(), ldb, beta, result.values_.data(), ldc);

	return result;
}

template<size_t num_row, size_t num_column>
Euclidean_Vector<num_row> Matrix<num_row, num_column>::operator*(const Euclidean_Vector<num_column>& x) const {
	std::array<double, num_row> result = { 0 };
	if constexpr (this->num_value_ < ms::blas_mv_criteria) {
		for (size_t i = 0; i < num_row; ++i)
			for (size_t j = 0; j < num_column; ++j)
				result[i] += this->values_[i * num_column + j] * x.at(j);
	}
	return result;
}

template<size_t num_row, size_t num_column>
bool Matrix<num_row, num_column>::operator==(const Matrix& A) const {
	return this->values_ == A.values_;
}


template<size_t num_row, size_t num_column>
double Matrix<num_row, num_column>::at(const size_t row_index, const size_t column_index) const {
	dynamic_require(row_index < num_row && column_index < num_column, "index can not exceed given range");
	return this->values_[row_index * num_column + column_index];
}

template<size_t num_row, size_t num_column>
Euclidean_Vector<num_row> Matrix<num_row, num_column>::column(const size_t column_index) const {
	dynamic_require(column_index < num_column, "index can not exceed given range");

	std::array<double, num_row> column_value;	
	for (size_t i = 0; i < num_row; ++i)
		column_value[i] = this->at(i, column_index);

	return column_value;
}

template<size_t num_row, size_t num_column>
std::string Matrix<num_row, num_column>::to_string(void) const {
	std::ostringstream oss;
	oss << std::setprecision(16) << std::showpoint << std::left;
	for (size_t i = 0; i < num_row; ++i) {
		for (size_t j = 0; j < num_column; ++j)
			oss << std::setw(25) << this->at(i, j);
		oss << "\n";
	}
	return oss.str();
}

template<size_t num_row, size_t num_column>
void Matrix<num_row, num_column>::change_columns(const size_t start_column_index, const Dynamic_Matrix& A) {
	const auto [num_A_row, num_A_column] = A.size();
	dynamic_require(num_A_row == num_row, "number of row should be same");
	dynamic_require(start_column_index + num_A_column <= num_column, "index can not exceed given range");

	for (size_t i = 0; i < num_row; ++i)
		for (size_t j = 0; j < num_A_column; ++j)
			this->value_at(i, start_column_index + j) = A.at(i, j);
}

template<size_t num_row, size_t num_column>
template <size_t num_other_column>
void Matrix<num_row, num_column>::change_columns(const size_t start_column_index, const Matrix<num_row, num_other_column>& A) {
	dynamic_require(start_column_index + num_other_column <= num_column, "index can not exceed given range");

	for (size_t i = 0; i < num_row; ++i)
		for (size_t j = 0; j < num_other_column; ++j)
			this->value_at(i, start_column_index + j) = A.at(i, j);
}

//template<size_t num_row, size_t num_column>
//void Matrix<num_row, num_column>::change_column(const size_t column_index, const Euclidean_Vector<num_row>& x) {
//	dynamic_require(column_index < num_column, "index can not exceed given range");
//	
//	for (size_t i = 0; i < num_row; ++i)
//		this->at(i, column_index) = x[i];
//}
//

template<size_t num_row, size_t num_column>
double& Matrix<num_row, num_column>::value_at(const size_t row_index, const size_t column_index) {
	// we should range check before call private at!
	// dynamic_require(row_index < num_row && column_index < num_column, "index can not exceed given range"); 
	return this->values_[row_index * num_column + column_index];
}

template<size_t num_row, size_t num_column>
std::ostream& operator<<(std::ostream& os, const Matrix<num_row, num_column>& m) {
	return os << m.to_string();
}

template<size_t num_row, size_t num_column>
Matrix<num_row, num_column> operator*(const double scalar, const Matrix<num_row, num_column>& A) {
	return A * scalar;
}

template <size_t num_row>
Euclidean_Vector<num_row> Dynamic_Matrix::column(const size_t column_index) const {
	dynamic_require(this->num_row_ == num_row, "this dynamic matrix should have given row dimension");

	std::array<double, num_row> column_values = { 0 };

	for (size_t i = 0; i < num_row; ++i)
		column_values[i] = this->at(i, column_index);

	return column_values;
}

template <size_t dimension>
void Dynamic_Matrix::change_row(const size_t row_index, const Euclidean_Vector<dimension>& vec) {
	dynamic_require(row_index < this->num_row_,		"column idnex can not exceed number of column");
	dynamic_require(this->num_column_ == dimension, "vector dimension should be matched with number of column");

	const auto jump_index = row_index * this->num_column_;
	std::copy(vec.cbegin(), vec.cend(), this->values_.begin() + jump_index);
}

template <size_t dimension>
void Dynamic_Matrix::change_column(const size_t column_index, const Euclidean_Vector<dimension>& vec) {
	dynamic_require(column_index < this->num_column_,	"column idnex can not exceed number of column");
	dynamic_require(this->num_row_ == dimension,		"vector dimension should be matched with number of row");

	for (size_t i = 0; i < this->num_row_; ++i)
		this->value_at(i, column_index) = vec.at(i);
}

template <size_t num_row, size_t num_column>
void Dynamic_Matrix::change_rows(const size_t start_row_index, const Matrix<num_row, num_column>& A) {
	dynamic_require(this->num_column_ == num_column,				"dimension should be matched");
	dynamic_require(start_row_index + num_row <= this->num_row_,	"range can't not exceed given range");
	dynamic_require(!this->is_transposed(),							"it should be not transposed for this routine");

	const auto jump_index = start_row_index * this->num_column_;
	std::copy(A.values_.begin(), A.values_.end(), this->values_.begin() + jump_index);
}

template <size_t num_row, size_t num_column>
void Dynamic_Matrix::change_columns(const size_t start_column_index, const Matrix<num_row, num_column>& A) {
	dynamic_require(this->num_row_ == num_row,								"dimension should be matched");
	dynamic_require(start_column_index + num_column <= this->num_column_,	"range can't not exceed given range");
	dynamic_require(!this->is_transposed(),									"it should be not transposed for this routine");

	for (size_t i = 0; i < this->num_row_; ++i) 
		for (size_t j = 0; j < num_column; ++j) 
			this->value_at(i, start_column_index + j) = A.at(i, j);
}