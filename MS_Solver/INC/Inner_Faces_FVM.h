#pragma once
#include "Grid.h"
#include "Reconstruction_Method_FVM.h"


//FVM�̸� �������� ����ϴ� variable
template <ushort space_dimension>
class Inner_Faces_FVM_Base
{
private:
    using Space_Vector_ = Euclidean_Vector<space_dimension>;

protected:
    std::vector<Space_Vector_> normals_;
    std::vector<std::pair<uint, uint>> oc_nc_index_pairs_;
    std::vector<double> volumes_;

public:
    Inner_Faces_FVM_Base(const Grid<space_dimension>& grid);
};


//FVM�̰� Constant Reconstruction�̸� �������� ����ϴ� variable & Method
template<typename Numerical_Flux_Function>
class Inner_Faces_FVM_Constant : public Inner_Faces_FVM_Base<Numerical_Flux_Function::space_dimension()>
{
private:
    static constexpr ushort space_dimension_ = Numerical_Flux_Function::space_dimension();
    static constexpr ushort num_equation_ = Numerical_Flux_Function::num_equation();

    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_equation_>;
    using Residual_ = Euclidean_Vector< num_equation_>;

public:
    Inner_Faces_FVM_Constant(const Grid<space_dimension_>& grid) : Inner_Faces_FVM_Base<space_dimension_>(grid) {};

    void calculate_RHS(std::vector<Residual_>& RHS, const std::vector<Solution_>& solutions) const;
};


//FVM�̰� Linear Reconstruction�̸� �������� ����ϴ� variable & Method
template <typename Reconstruction_Method, typename Numerical_Flux_Function>
class Inner_Faces_FVM_Linear : public Inner_Faces_FVM_Base<Numerical_Flux_Function::space_dimension()>
{
private:
    static constexpr ushort space_dimension_    = Numerical_Flux_Function::space_dimension();
    static constexpr ushort num_equation_       = Numerical_Flux_Function::num_equation();

    using This_ = Inner_Faces_FVM_Linear<Reconstruction_Method, Numerical_Flux_Function>;
    using Space_Vector_ = Euclidean_Vector<This_::space_dimension_>;
    using Solution_ = Euclidean_Vector<This_::num_equation_>;
    using Residual_ = Euclidean_Vector<This_::num_equation_>;

private:
    std::vector<std::pair<Space_Vector_, Space_Vector_>> oc_nc_to_face_vector_pairs_;
    const Reconstruction_Method& reconstruction_method_;

public:
    Inner_Faces_FVM_Linear(const Grid<space_dimension_>& grid, const Reconstruction_Method& reconstruction_method);

public:
    void calculate_RHS(std::vector<Residual_>& RHS, const std::vector<Solution_>& solutions) const;
};


// template definition part
template <ushort space_dimension>
Inner_Faces_FVM_Base<space_dimension>::Inner_Faces_FVM_Base(const Grid<space_dimension>& grid) {
    SET_TIME_POINT;

    this->oc_nc_index_pairs_ = grid.inner_face_oc_nc_index_pairs();
    this->volumes_ = grid.inner_face_volumes();
    this->normals_ = grid.inner_face_normals_at_center(this->oc_nc_index_pairs_);

    Log::content_ << std::left << std::setw(50) << "@ Inner faces FVM base precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
    Log::print();
}

template<typename Numerical_Flux_Function>
void Inner_Faces_FVM_Constant<Numerical_Flux_Function>::calculate_RHS(std::vector<Residual_>& RHS, const std::vector<Solution_>& solutions) const {
    const auto num_inner_face = this->normals_.size();

    const auto numerical_fluxes = Numerical_Flux_Function::calculate(solutions, this->normals_, this->oc_nc_index_pairs_);

    for (size_t i = 0; i < num_inner_face; ++i) {
        const auto [oc_index, nc_index] = this->oc_nc_index_pairs_[i];
        const auto delta_RHS = this->volumes_[i] * numerical_fluxes[i];
        RHS[oc_index] -= delta_RHS;
        RHS[nc_index] += delta_RHS;
    }
}

template<typename Reconstruction_Method, typename Numerical_Flux_Function>
Inner_Faces_FVM_Linear<Reconstruction_Method, Numerical_Flux_Function>::Inner_Faces_FVM_Linear(const Grid<space_dimension_>& grid, const Reconstruction_Method& reconstruction_method)
    : Inner_Faces_FVM_Base<space_dimension_>(grid), reconstruction_method_(reconstruction_method) {
    SET_TIME_POINT;

    this->oc_nc_to_face_vector_pairs_ = grid.inner_face_oc_nc_to_face_pairs(this->oc_nc_index_pairs_);
 
    Log::content_ << std::left << std::setw(50) << "@ Inner faces FVM linear precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
    Log::print();
};


template<typename Reconstruction_Method, typename Numerical_Flux_Function>
void Inner_Faces_FVM_Linear<Reconstruction_Method, Numerical_Flux_Function>::calculate_RHS(std::vector<Residual_>& RHS, const std::vector<Solution_>& solutions) const {
    const auto& solution_gradients = this->reconstruction_method_.get_solution_gradients();

    const auto num_inner_face = this->normals_.size();

    for (size_t i = 0; i < num_inner_face; ++i) {
        const auto [oc_index, nc_index] = this->oc_nc_index_pairs_[i];
        const auto& oc_solution = solutions[oc_index];
        const auto& nc_solution = solutions[nc_index];

        const auto& oc_solution_gradient = solution_gradients[oc_index];
        const auto& nc_solution_gradient = solution_gradients[nc_index];

        const auto& [oc_to_face_vector, nc_to_face_vector] = this->oc_nc_to_face_vector_pairs_[i];

        const auto oc_side_solution = oc_solution + oc_solution_gradient * oc_to_face_vector;
        const auto nc_side_solution = nc_solution + nc_solution_gradient * nc_to_face_vector;
        const auto& inner_face_normal = this->normals_[i];

        const auto numerical_flux = Numerical_Flux_Function::calculate(oc_side_solution, nc_side_solution, inner_face_normal);
        const auto delta_RHS = this->volumes_[i] * numerical_flux;
        RHS[oc_index] -= delta_RHS;
        RHS[nc_index] += delta_RHS;
    }
}