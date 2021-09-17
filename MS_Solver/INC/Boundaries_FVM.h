#pragma once
#include "Boundary_Flux_Function.h"
#include "Grid.h"
#include "Reconstruction_Method_FVM.h"


//FVM�̸� �������� ����ϴ� variable
template <typename Numerical_Flux_Function>
class Boundaries_FVM_Base
{
private:
    static constexpr size_t space_dimension_ = Numerical_Flux_Function::space_dimension();

    using Space_Vector_ = Euclidean_Vector<space_dimension_>;

protected:
    std::vector<Space_Vector_> normals_;
    std::vector<uint> oc_indexes_;
    std::vector<double> volumes_;
    std::vector<std::unique_ptr<Boundary_Flux_Function<Numerical_Flux_Function>>> boundary_flux_functions_;

public:
    Boundaries_FVM_Base(const Grid<space_dimension_>& grid);
};


//FVM�̰� Constant Reconstruction�̸� �������� ����ϴ� variable & Method
template <typename Numerical_Flux_Function>
class Boundaries_FVM_Constant : public Boundaries_FVM_Base<Numerical_Flux_Function>
{
private:
    static constexpr size_t space_dimension_ = Numerical_Flux_Function::space_dimension();
    static constexpr size_t num_equation_ = Numerical_Flux_Function::num_equation();

    using Solution_         = Euclidean_Vector<num_equation_>;
    using Boundary_Flux_    = Euclidean_Vector<num_equation_>;

public:
    Boundaries_FVM_Constant(const Grid<space_dimension_>& grid) : Boundaries_FVM_Base<Numerical_Flux_Function>(grid) {};

    void calculate_RHS(std::vector<Boundary_Flux_>& RHS, const std::vector<Solution_>& solutions) const;
};


//FVM�̰� Linear Reconstruction�̸� �������� ����ϴ� variable & Method
template <typename Reconstruction_Method, typename Numerical_Flux_Function>
class Boundaries_FVM_Linear : public Boundaries_FVM_Base<Numerical_Flux_Function>
{
private:
    static constexpr size_t space_dimension_ = Numerical_Flux_Function::space_dimension();
    static constexpr size_t num_equation_ = Numerical_Flux_Function::num_equation();

    using Space_Vector_ = Euclidean_Vector <space_dimension_>;
    using Solution_ = Euclidean_Vector<num_equation_>;
    using Boundary_Flux_ = Euclidean_Vector<num_equation_>;

private:
    std::vector<Space_Vector_> oc_to_boundary_vectors_;
    const Reconstruction_Method& reconstruction_method_;

public:
    Boundaries_FVM_Linear(const Grid<space_dimension_>& grid, const Reconstruction_Method& reconstruction_method);

    void calculate_RHS(std::vector<Boundary_Flux_>& RHS, const std::vector<Solution_>& solutions) const;
};


//template definition
template <typename Numerical_Flux_Function>
Boundaries_FVM_Base<Numerical_Flux_Function>::Boundaries_FVM_Base(const Grid<space_dimension_>& grid) {
    SET_TIME_POINT;

    this->oc_indexes_ = grid.boundary_owner_cell_indexes();
    this->volumes_ = grid.boundary_volumes();
    this->boundary_flux_functions_ = grid.boundary_flux_functions<Numerical_Flux_Function>();
    this->normals_ = grid.boundary_normals_at_centers(this->oc_indexes_);

    Log::content_ << std::left << std::setw(50) << "@ Boundaries FVM base precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
    Log::print();
}

template <typename Numerical_Flux_Function>
void Boundaries_FVM_Constant<Numerical_Flux_Function>::calculate_RHS(std::vector<Boundary_Flux_>& RHS, const std::vector<Solution_>& solutions) const {
    const auto num_boundary = this->normals_.size();

    for (size_t i = 0; i < num_boundary; ++i) {
        const auto& boundary_flux_function = this->boundary_flux_functions_.at(i);

        const auto oc_index = this->oc_indexes_[i];
        const auto& normal = this->normals_[i];

        const auto boundary_flux = boundary_flux_function->calculate(solutions[oc_index], normal);
        const auto delta_RHS = this->volumes_[i] * boundary_flux;

        RHS[oc_index] -= delta_RHS;
    }
}

template <typename Reconstruction_Method, typename Numerical_Flux_Function>
Boundaries_FVM_Linear<Reconstruction_Method, Numerical_Flux_Function>::Boundaries_FVM_Linear(const Grid<space_dimension_>& grid, const Reconstruction_Method& reconstruction_method)
    : Boundaries_FVM_Base<Numerical_Flux_Function>(grid), reconstruction_method_(reconstruction_method) {
    SET_TIME_POINT;

    this->oc_to_boundary_vectors_ = grid.owner_cell_center_to_boundary_center_vectors(this->oc_indexes_);

    Log::content_ << std::left << std::setw(50) << "@ Boundaries FVM linear precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
    Log::print();
};

template <typename Reconstruction_Method, typename Numerical_Flux_Function>
void Boundaries_FVM_Linear<Reconstruction_Method, Numerical_Flux_Function>::calculate_RHS(std::vector<Boundary_Flux_>& RHS, const std::vector<Solution_>& solutions) const {
    const auto& solution_gradients = this->reconstruction_method_.get_solution_gradients();

    const auto num_boundary = this->normals_.size();

    for (size_t i = 0; i < num_boundary; ++i) {
        const auto& boundary_flux_function = this->boundary_flux_functions_.at(i);
        const auto& normal = this->normals_.at(i);

        const auto oc_index = this->oc_indexes_.at(i);

        const auto& oc_solution = solutions[oc_index];
        const auto& oc_solution_gradient = solution_gradients[oc_index];
        const auto& oc_to_face_vector = this->oc_to_boundary_vectors_[i];

        const auto oc_side_solution = oc_solution + oc_solution_gradient * oc_to_face_vector;

        const auto boundary_flux = boundary_flux_function->calculate(oc_side_solution, normal);

        const auto delta_RHS = this->volumes_[i] * boundary_flux;

        RHS[oc_index] -= delta_RHS;
    }
}