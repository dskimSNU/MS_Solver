#pragma once
#include "Matrix.h"

#include <initializer_list>


using ushort = unsigned short;


template<typename Function, ushort range_dimension_>
class Vector_Function
{
private:
	static constexpr size_t domain_dimension_ = Function::domain_dimension();

private:
	std::array<Function, range_dimension_> functions_ = { 0 };

public:
	Vector_Function(void) = default;
	Vector_Function(const std::array<Function, range_dimension_>& functions) : functions_(functions) {};
	
	template <typename... Args>
	Vector_Function(const Args... args) {
		static_require(sizeof...(Args) <= range_dimension_, "Number of arguments can not exceed dimension");
		
		functions_ = { Function(args)... };		
	}

	Euclidean_Vector<range_dimension_> operator()(const Euclidean_Vector<domain_dimension_>& space_vector) const {

		std::array<double, range_dimension_> result = { 0 };
		for (size_t i = 0; i < range_dimension_; ++i)
			result[i] = functions_[i](space_vector);

		return result;
	}

	std::vector<Euclidean_Vector<range_dimension_>> operator()(const std::vector<Euclidean_Vector<domain_dimension_>>& space_vectors) const {
		const auto num_vector = space_vectors.size();

		std::vector<Euclidean_Vector<range_dimension_>> result;
		result.reserve(num_vector);

		for (size_t i = 0; i < num_vector; ++i)
			result.push_back((*this)(space_vectors[i]));		

		return result;
	}

	bool operator==(const Vector_Function& other) const {
		return this->functions_ == other.functions_;
	}

	const Function& operator[](const size_t index) const {
		dynamic_require(index < range_dimension_, "index can not exceed range dimension");
		return this->functions_[index];
	}

	const Function& at(const size_t index) const {
		dynamic_require(index < range_dimension_, "index can not exceed range dimension");
		return this->functions_[index];
	}

	template <ushort temp = range_dimension_, std::enable_if_t<temp == 2, bool> = true >
	Vector_Function<Function, range_dimension_ + 1> cross_product(const Vector_Function& other) const {
		std::array<Function, range_dimension_ + 1> result;
			result[2] = this->at(0) * other.at(1) - this->at(1) * other.at(0);

		return result;
	}

	template <ushort temp = range_dimension_, std::enable_if_t<temp == 3, bool> = true >
	Vector_Function<Function, range_dimension_> cross_product(const Vector_Function& other) const {
		static_require(range_dimension_ == 3, "cross product only valid for R^3 dimension");

		std::array<Function, range_dimension_> result;
		result[0] = this->at(1) * other.at(2) - this->at(2) * other.at(1);
		result[1] = this->at(2) * other.at(0) - this->at(0) * other.at(2);
		result[2] = this->at(0) * other.at(1) - this->at(1) * other.at(0);

		return result;
	}

	Function* data(void) {
		return this->functions_.data();
	}

	template <size_t variable_index>
	Vector_Function<Function, range_dimension_> differentiate(void) const {
		static_require(variable_index < domain_dimension_, "variable index can not exceed domain dimension");
				
		std::array<Function, range_dimension_> result;
		for (size_t i = 0; i < range_dimension_; ++i)
			result[i] = this->functions_[i].differentiate<variable_index>();

		return result;
	}

	auto L2_norm(void) const {
		Function result(0);

		for (const auto& function : this->functions_)
			result += (function ^ 2);

		return result.root(0.5);
	}

	std::string to_string(void) const {
		std::string result;

		for (const auto& function : this->functions_)
			result += function.to_string() + "\n";

		return result;
	}

	constexpr ushort range_dimension(void) const {
		return range_dimension_;
	}
};


// Vector_Function class template for Range dimension is not compile time constant
template <typename Function>
using Dynamic_Vector_Function_ = Vector_Function<Function, 0>;


template <typename Function>
class Vector_Function<Function, 0>
{
private:
	static constexpr size_t domain_dimension_ = Function::domain_dimension();

private:
	std::vector<Function> functions_;

public:
	Vector_Function(std::vector<Function>&& functions) : functions_(std::move(functions)) {};
	Vector_Function(const std::initializer_list<Function> list) : functions_(list) {};

	Dynamic_Euclidean_Vector_ operator()(const Euclidean_Vector<domain_dimension_>& space_vector) const {
		const auto range_dimension_ = this->range_dimension();
		
		std::vector<double> result(range_dimension_);
		for (size_t i = 0; i < range_dimension_; ++i)
			result[i] = functions_.at(i)(space_vector);

		return std::move(result);
	}

	const Function& operator[](const size_t index) const {
		dynamic_require(index < this->range_dimension(), "index can not exceed range dimension");
		return this->functions_[index];
	}

	bool operator==(const Vector_Function& other) const {
		return this->functions_ == other.functions_;
	}

	const Function& at(const size_t index) const {
		dynamic_require(index < this->range_dimension(), "index can not exceed range dimension");
		return this->functions_[index];
	}

	size_t range_dimension(void) const {
		return this->functions_.size();
	}

	std::string to_string(void) const {
		std::string result;

		for (const auto& function : this->functions_)
			result += function.to_string() + "\n";

		return result;
	}
};


template <typename Function, ushort range_num_row, ushort range_num_column>
class Matrix_Function
{
private:
	static constexpr ushort num_value_			= range_num_row * range_num_column;
	static constexpr ushort domain_dimension_	= Function::domain_dimension();

private:
	std::array<Function, num_value_> functions_;

public:
	Matrix<range_num_row, range_num_column> operator()(const Euclidean_Vector<domain_dimension_>& space_vector) {
		std::array<double, num_value_> values;

		for (ushort i = 0; i < range_num_row; ++i)
			for (ushort j = 0; j < range_num_column; ++j)
				values[i * range_num_column + j] = this->at(i, j)(space_vector);
		
		return values;
	}

	const Function& at(const ushort row_index, const ushort column_index) const {
		dynamic_require(row_index < range_num_row && column_index < range_num_column, "index can not exceed given range");
		return this->functions_[row_index * range_num_column + column_index];
	}

	void change_column(const ushort column_index, const Vector_Function<Function, range_num_row>& column_vector) {
		for (ushort i = 0; i < range_num_row; ++i) 
			this->at(i, column_index) = column_vector[i];		
	}

private:
	Function& at(const ushort row_index, const ushort column_index) {
		dynamic_require(row_index < range_num_row&& column_index < range_num_column, "index can not exceed given range");
		return this->functions_[row_index * range_num_column + column_index];
	}
};


template <typename Function>
std::ostream& operator<<(std::ostream& os, const Dynamic_Vector_Function_<Function>& vf) {
	return os << vf.to_string();
}