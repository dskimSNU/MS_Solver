#pragma once
#include "Numerical_Flux_Function.h"
#include "Element.h"


template <ushort num_equation>
class Supersonic_Inlet1_Neighbor_Solution
{
private:
	Supersonic_Inlet1_Neighbor_Solution(void) = delete;

private:
	inline static Euclidean_Vector<num_equation> inflow_;

public:
	static void initialize(const Euclidean_Vector<num_equation>& inflow) { inflow_ = inflow; };
	static Euclidean_Vector<num_equation> calculate(void) { return inflow_; };
};


template <ushort num_equation>
class Supersonic_Inlet2_Neighbor_Solution
{
private:
	Supersonic_Inlet2_Neighbor_Solution(void) = delete;

private:
	inline static Euclidean_Vector<num_equation> inflow_;

public:
	static void initialize(const Euclidean_Vector<num_equation>& inflow) { inflow_ = inflow; };
	static Euclidean_Vector<num_equation> calculate(void) { return inflow_; };
};

template <ushort num_equation>
class Slip_Wall_Neighbor_Solution
{
private:
	Slip_Wall_Neighbor_Solution(void) = delete;

public:
	static Euclidean_Vector<num_equation> calculate(const Euclidean_Vector<num_equation>& solution) {
		std::array<double, num_equation> neighbor_cell_solution_value = solution;		
		
		//temporal code for 2D
		neighbor_cell_solution_value[1] *= -1;
		neighbor_cell_solution_value[2] *= -1;
		return neighbor_cell_solution_value;
		//temporal code for 2D
	};
};

template <typename Numerical_Flux_Function>
class Boundary_Flux_Function
{
protected:
	static_require(ms::is_numeirical_flux_function<Numerical_Flux_Function>, "it shuld be numerical flux function");

	static constexpr ushort space_dimension_	= Numerical_Flux_Function::space_dimension();
	static constexpr ushort num_equation_		= Numerical_Flux_Function::num_equation();

	using Space_Vector_		= Euclidean_Vector<space_dimension_>;
	using Solution_			= Euclidean_Vector<num_equation_>;
	using Boundary_Flux_	= Euclidean_Vector<num_equation_>;

public:
	virtual Boundary_Flux_ calculate(const Solution_& oc_cvariable, const Space_Vector_& normal) const abstract; //virtual은 함수 템플릿에서 사용할 수 없습니다.
};


template <typename Numerical_Flux_Function>
class Supersonic_Inlet1 : public Boundary_Flux_Function<Numerical_Flux_Function>
{
private:
	using This_			= Supersonic_Inlet1<Numerical_Flux_Function>;
	using Space_Vector_ = This_::Space_Vector_;
	using Solution_		= This_::Solution_;

public:
	This_::Boundary_Flux_ calculate(const Solution_& solution, const Space_Vector_& normal) const override {
		return Numerical_Flux_Function::calculate(solution, Supersonic_Inlet1_Neighbor_Solution<This_::num_equation_>::calculate(), normal);
	}
};


template <typename Numerical_Flux_Function>
class Supersonic_Inlet2 : public Boundary_Flux_Function<Numerical_Flux_Function>
{
private:
	using This_			= Supersonic_Inlet2<Numerical_Flux_Function>;
	using Space_Vector_ = This_::Space_Vector_;
	using Solution_		= This_::Solution_;

public:
	This_::Boundary_Flux_ calculate(const Solution_& solution, const Space_Vector_& normal) const override {
		return Numerical_Flux_Function::calculate(solution, Supersonic_Inlet2_Neighbor_Solution<This_::num_equation_>::calculate(), normal);
	}
};


template <typename Numerical_Flux_Function>
class Supersonic_Outlet : public Boundary_Flux_Function<Numerical_Flux_Function>
{
private:
	using This_			= Supersonic_Outlet<Numerical_Flux_Function>;
	using Space_Vector_ = This_::Space_Vector_;
	using Solution_		= This_::Solution_;

public:
	This_::Boundary_Flux_ calculate(const Solution_& solution, const Space_Vector_& normal) const override {
		return Numerical_Flux_Function::calculate(solution, solution, normal);
	}
};


template <typename Numerical_Flux_Function>
class Slip_Wall : public Boundary_Flux_Function<Numerical_Flux_Function>
{
private:
	using This_ = Slip_Wall<Numerical_Flux_Function>;
	using Space_Vector_ = This_::Space_Vector_;
	using Solution_ = This_::Solution_;

public:
	This_::Boundary_Flux_ calculate(const Solution_& oc_cvariable, const Space_Vector_& normal) const override {
		static_require(This_::space_dimension_ <= 3, "dimension can not exceed 3");

		if constexpr (ms::is_Euler<Numerical_Flux_Function::Governing_Equation_>) {
			const auto pvariable = Euler<This_::space_dimension_>::conservative_to_primitive(oc_cvariable);

			if constexpr (This_::space_dimension_ == 2) {
				const auto p = pvariable[2];
				return { 0.0, p * normal[0], p * normal[1], 0.0 };
			}
			else {
				const auto p = pvariable[3];
				return { 0.0, p * normal[0], p * normal[1], p * normal[2], 0.0 };
			}
		}
		else {
			throw std::runtime_error("Governing equation should be Euler");			
			return {};
		}
	}
};



template <typename Numerical_Flux_Function>
class Boundary_Flux_Function_Factory
{
public:
	static std::unique_ptr<Boundary_Flux_Function<Numerical_Flux_Function>> make(const ElementType boundary_type);
};


template <typename Numerical_Flux_Function>
std::unique_ptr<Boundary_Flux_Function<Numerical_Flux_Function>> Boundary_Flux_Function_Factory<Numerical_Flux_Function>::make(const ElementType boundary_type) {
	switch (boundary_type)
	{
	case ElementType::supersonic_inlet1:
		return std::make_unique<Supersonic_Inlet1<Numerical_Flux_Function>>();
	case ElementType::supersonic_inlet2:
		return std::make_unique<Supersonic_Inlet2<Numerical_Flux_Function>>();
	case ElementType::supersonic_outlet:
		return std::make_unique<Supersonic_Outlet<Numerical_Flux_Function>>();
	case ElementType::slip_wall:
		return std::make_unique<Slip_Wall<Numerical_Flux_Function>>();
	default:
		throw std::runtime_error("wrong element type");
		break;
	}	
}