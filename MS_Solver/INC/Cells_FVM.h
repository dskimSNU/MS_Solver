#pragma once
#include "Grid.h"
#include "Governing_Equation.h"


//FVM이면 공통으로 사용하는 variable & method
template <typename Governing_Equation>
class Cells_FVM
{
private:
    static constexpr ushort space_dimension_ = Governing_Equation::space_dimension();
    static constexpr ushort num_equation_ = Governing_Equation::num_equation();

    using Space_Vector_ = Euclidean_Vector<space_dimension_>;

public:
    using Discretized_Solution_ = Euclidean_Vector<num_equation_>;

protected:
    std::vector<Space_Vector_> centers_;
    std::vector<double> volumes_;
    std::vector<std::array<double, space_dimension_>> projected_volumes_;
    std::vector<double> residual_scale_factors_;

public:
    Cells_FVM(const Grid<space_dimension_>& grid);

    double calculate_time_step(const std::vector<Euclidean_Vector<num_equation_>>& solutions, const double cfl) const;
    void calculate_RHS(std::vector<Euclidean_Vector<num_equation_>>& RHS, const std::vector<Euclidean_Vector<num_equation_>>& solutions) const;

    template <typename Initial_Condition>
    auto calculate_initial_solutions(void) const;

    void estimate_error(const std::vector<Euclidean_Vector<num_equation_>>& computed_solutions, const double time) const;
};


//template definition
template <typename Governing_Equation>
Cells_FVM<Governing_Equation>::Cells_FVM(const Grid<space_dimension_>& grid) {
    SET_TIME_POINT;

    this->centers_ = grid.cell_center_nodes();
    this->volumes_ = grid.cell_volumes();
    this->projected_volumes_ = grid.cell_projected_volumes();

    const auto num_cell = this->centers_.size();
    this->residual_scale_factors_.resize(num_cell);

    for (uint i = 0; i < num_cell; ++i) 
        this->residual_scale_factors_[i] = 1.0 / this->volumes_[i];

    Log::content_ << std::left << std::setw(50) << "@ Cells FVM precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
    Log::print();
};

template <typename Governing_Equation>
double Cells_FVM<Governing_Equation>::calculate_time_step(const std::vector<Euclidean_Vector<num_equation_>>& solutions, const double cfl) const {
    const auto projected_maximum_lambdas = Governing_Equation::calculate_coordinate_projected_maximum_lambdas(solutions);
    const auto num_cell = projected_maximum_lambdas.size();

    std::vector<double> local_time_step(num_cell);
    for (size_t i = 0; i < num_cell; ++i) {
        const auto [y_projected_volume, x_projected_volume] = this->projected_volumes_[i];
        const auto [x_projeced_maximum_lambda, y_projeced_maximum_lambda] = projected_maximum_lambdas[i];

        const auto x_radii = y_projected_volume * x_projeced_maximum_lambda;
        const auto y_radii = x_projected_volume * y_projeced_maximum_lambda;
        dynamic_require(std::isfinite(x_radii) && std::isfinite(y_radii), "time step should be finite number");

        local_time_step[i] = cfl * this->volumes_[i] / (x_radii + y_radii);
    }

    return *std::min_element(local_time_step.begin(), local_time_step.end());
}

template <typename Governing_Equation>
void Cells_FVM<Governing_Equation>::calculate_RHS(std::vector<Euclidean_Vector<num_equation_>>& RHS, const std::vector<Euclidean_Vector<num_equation_>>& solutions) const {
    const auto num_cell = RHS.size();

    for (size_t i = 0; i < num_cell; ++i)
        RHS[i] *= this->residual_scale_factors_[i];
}

template <typename Governing_Equation>
template <typename Initial_Condition>
auto Cells_FVM<Governing_Equation>::calculate_initial_solutions(void) const {
    return Initial_Condition::calculate_solutions(this->centers_);
}

template <typename Governing_Equation>
void Cells_FVM<Governing_Equation>::estimate_error(const std::vector<Euclidean_Vector<num_equation_>>& computed_solutions, const double time) const {

    Log::content_ << "================================================================================\n";
    Log::content_ << "\t\t\t\t Error Anlysis\n";
    Log::content_ << "================================================================================\n";

    double arithmetic_mean_L1_error = 0.0;

    const auto exact_solutions = Sine_Wave<space_dimension_>::calculate_exact_solutions(this->centers_, time);
    const auto num_cell = computed_solutions.size();

    for (size_t i = 0; i < num_cell; ++i) {
        const auto local_diff = (exact_solutions[i] - computed_solutions[i]).L2_norm();
        const auto local_L1_error = local_diff;
        arithmetic_mean_L1_error += local_L1_error;
    }

    arithmetic_mean_L1_error = arithmetic_mean_L1_error / num_cell;

    Log::content_ << std::left << std::setprecision(16);
    Log::content_ << std::setw(25) << "L1 error\n";
    Log::content_ << std::setw(25) << arithmetic_mean_L1_error << "\n";
    Log::print();
}

