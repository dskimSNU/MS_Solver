#pragma once
#include "Grid.h"
#include "Reconstruction_Method_HOM.h"
#include "Numerical_Flux_Function.h"
#include "Solution_Scaler.h"

//HOM이면 공통으로 사용하는 variable & method
template<typename Reconstruction_Method, typename Numerical_Flux_Function>
class Periodic_Boundaries_HOM
{
private:
    static constexpr ushort space_dimension_    = Reconstruction_Method::space_dimension();
    static constexpr ushort num_basis_          = Reconstruction_Method::num_basis();
    static constexpr ushort num_equation_       = Numerical_Flux_Function::num_equation();

    using This_                 = Periodic_Boundaries_HOM<Reconstruction_Method, Numerical_Flux_Function>;
    using Space_Vector_         = Euclidean_Vector<space_dimension_>;
    using Solution_Coefficient_ = Matrix<num_equation_, num_basis_>;
    using Residual_             = Matrix<num_equation_, num_basis_>;

protected:
    std::vector<std::pair<uint, uint>> oc_nc_index_pairs_;
    std::vector<std::pair<Dynamic_Matrix, Dynamic_Matrix>> oc_nc_side_basis_qnodes_pairs_;
    std::vector<std::pair<Dynamic_Matrix, Dynamic_Matrix>> oc_nc_side_qweights_basis_pairs_;
    std::vector<std::vector<Space_Vector_>> set_of_normals_;

public:
    Periodic_Boundaries_HOM(const Grid<space_dimension_>& grid, const Reconstruction_Method& reconstruction_method);

public:
    void calculate_RHS(std::vector<Residual_>& RHS, const std::vector<Solution_Coefficient_>& solution_coefficients) const;

public:
    void initialize_scaling_method(void) const;
};


// template definition part
template<typename Reconstruction_Method, typename Numerical_Flux_Function>
Periodic_Boundaries_HOM<Reconstruction_Method, Numerical_Flux_Function>::Periodic_Boundaries_HOM(const Grid<space_dimension_>& grid, const Reconstruction_Method& reconstruction_method){
    SET_TIME_POINT;

    constexpr auto integrand_degree = 2 * Reconstruction_Method::solution_order() + 1;

    this->oc_nc_index_pairs_ = grid.periodic_boundary_oc_nc_index_pairs();
    auto pbdry_quadrature_rule_pairs = grid.periodic_boundary_quadrature_rule_pairs(integrand_degree);

    const auto num_periodic_pair = oc_nc_index_pairs_.size();
    this->oc_nc_side_basis_qnodes_pairs_.reserve(num_periodic_pair);
    this->oc_nc_side_qweights_basis_pairs_.reserve(num_periodic_pair);

    std::vector<uint> oc_indexes(num_periodic_pair);
    std::vector<std::vector<Euclidean_Vector<space_dimension_>>> set_of_oc_side_qnodes(num_periodic_pair);

    for (uint i = 0; i < num_periodic_pair; ++i) {
        const auto& [oc_side_quadrature_rule, nc_side_quadrature_rule] = pbdry_quadrature_rule_pairs[i];
        
        const auto& oc_side_qnodes = oc_side_quadrature_rule.points;
        const auto& oc_side_qweights = oc_side_quadrature_rule.weights;

        const auto& nc_side_qnodes = nc_side_quadrature_rule.points;
        const auto& nc_side_qweights = nc_side_quadrature_rule.weights;

        const auto [oc_index, nc_index] = this->oc_nc_index_pairs_[i];
        auto oc_side_basis_qnode = reconstruction_method.basis_nodes(oc_index, oc_side_qnodes);
        auto nc_side_basis_qnode = reconstruction_method.basis_nodes(nc_index, nc_side_qnodes);

        const auto num_qnode = oc_side_qnodes.size();

        Dynamic_Matrix oc_side_basis_weight(num_qnode, This_::num_basis_);
        Dynamic_Matrix nc_side_basis_weight(num_qnode, This_::num_basis_);

        for (ushort q = 0; q < num_qnode; ++q) {
            oc_side_basis_weight.change_row(q, reconstruction_method.calculate_basis_node(oc_index, oc_side_qnodes[q]) * oc_side_qweights[q]);
            nc_side_basis_weight.change_row(q, reconstruction_method.calculate_basis_node(nc_index, nc_side_qnodes[q]) * nc_side_qweights[q]);
        }

        this->oc_nc_side_basis_qnodes_pairs_.push_back({ std::move(oc_side_basis_qnode), std::move(nc_side_basis_qnode) });
        this->oc_nc_side_qweights_basis_pairs_.push_back({ std::move(oc_side_basis_weight), std::move(nc_side_basis_weight) });

        oc_indexes[i] = oc_index;
        set_of_oc_side_qnodes[i] = std::move(oc_side_qnodes);
    }

    this->set_of_normals_ = grid.periodic_boundary_set_of_normals(oc_indexes, set_of_oc_side_qnodes);


    Log::content_ << std::left << std::setw(50) << "@ Periodic Boundaries HOM precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
    Log::print();
}

template<typename Reconstruction_Method, typename Numerical_Flux_Function>
void Periodic_Boundaries_HOM<Reconstruction_Method, Numerical_Flux_Function>::calculate_RHS(std::vector<Residual_>& RHS, const std::vector<Solution_Coefficient_>& solution_coefficients) const {
    const auto num_periodic_pairs = this->oc_nc_index_pairs_.size();

    for (uint i = 0; i < num_periodic_pairs; ++i) {
        const auto [oc_index, nc_index] = this->oc_nc_index_pairs_[i];

        const auto& oc_solution_coefficient = solution_coefficients[oc_index];
        const auto& nc_solution_coefficient = solution_coefficients[nc_index];

        const auto& [oc_side_basis_qnodes, nc_side_basis_qnodes] = this->oc_nc_side_basis_qnodes_pairs_[i];
        const auto oc_side_solution_qnodes = oc_solution_coefficient * oc_side_basis_qnodes;
        const auto nc_side_solution_qnodes = nc_solution_coefficient * nc_side_basis_qnodes;

        const auto [num_equation, num_qnode] = oc_side_solution_qnodes.size();
        const auto& normals = this->set_of_normals_[i];

        Dynamic_Matrix numerical_flux_quadrature(This_::num_equation_, num_qnode);
        for (ushort q = 0; q < num_qnode; ++q) {
            const auto oc_side_solution = oc_side_solution_qnodes.column<This_::num_equation_>(q);
            const auto nc_side_solution = nc_side_solution_qnodes.column<This_::num_equation_>(q);

            numerical_flux_quadrature.change_column(q, Numerical_Flux_Function::calculate(oc_side_solution, nc_side_solution, normals[q]));
        }               

        Residual_ owner_side_delta_rhs, neighbor_side_delta_rhs;
        const auto& [oc_side_basis_weight, nc_side_basis_weight] = this->oc_nc_side_qweights_basis_pairs_[i];
        ms::gemm(numerical_flux_quadrature, oc_side_basis_weight, owner_side_delta_rhs);
        ms::gemm(numerical_flux_quadrature, nc_side_basis_weight, neighbor_side_delta_rhs);

        RHS[oc_index] -= owner_side_delta_rhs;
        RHS[nc_index] += neighbor_side_delta_rhs;
    }
}

template<typename Reconstruction_Method, typename Numerical_Flux_Function>
void Periodic_Boundaries_HOM<Reconstruction_Method, Numerical_Flux_Function>::initialize_scaling_method(void) const {
    const auto num_pbdry_pair = this->oc_nc_index_pairs_.size();

    for (uint i = 0; i < num_pbdry_pair; ++i) {
        const auto [oc_index, nc_index] = this->oc_nc_index_pairs_[i];
        const auto& [oc_side_basis_qnodes, nc_side_basis_qnodes] = this->oc_nc_side_basis_qnodes_pairs_[i];

        Solution_Scaler<This_::space_dimension_>::record_face_basis_qnodes(oc_index, oc_side_basis_qnodes);
        Solution_Scaler<This_::space_dimension_>::record_face_basis_qnodes(nc_index, nc_side_basis_qnodes);
    }
}
