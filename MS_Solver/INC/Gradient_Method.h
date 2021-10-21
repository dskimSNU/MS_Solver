#pragma once
#include "Grid.h"

template <ushort num_equation_, ushort space_dimension_>
class Least_Square_Base
{
private:    
    using Solution_             = Euclidean_Vector<num_equation_>;
    using Solution_gradient_    = Matrix<num_equation_, space_dimension_>;

protected:
    std::vector<std::vector<uint>> set_of_near_cell_indexes_;
    std::vector<std::vector<uint>> set_of_near_ghost_cell_indexes_;

    std::vector<Dynamic_Matrix> least_square_matrixes_;
    std::vector<Ghost_Cell<space_dimension_>> ghost_cells_;

public:
    std::vector<Solution_gradient_> calculate_solution_gradients(const std::vector<Solution_>& solutions) const;

private:
    std::vector<Dynamic_Matrix> calculate_solution_delta_matrixes(const std::vector<Solution_>& solutions) const;

public:
    static constexpr ushort space_dimension(void) { return space_dimension_; };
    static constexpr ushort num_equation(void) { return num_equation_; };
};


//template <ushort num_equation_, ushort space_dimension_>
//class Vertex_Least_Square : public Least_Square_Base<num_equation_, space_dimension_>
//{
//public:
//    Vertex_Least_Square(const Grid<space_dimension_>& grid);
//
//    static std::string name(void) { return "Vertex_Least_Square"; };
//};


template <ushort num_equation_, ushort space_dimension_>
class Face_Least_Square : public Least_Square_Base<num_equation_, space_dimension_>
{
public:
    Face_Least_Square(const Grid<space_dimension_>& grid);

    static std::string name(void) { return "Face_Least_Square"; };
};



//template definition part
template <ushort num_equation_, ushort space_dimension_>
std::vector<Matrix<num_equation_, space_dimension_>> Least_Square_Base<num_equation_, space_dimension_>::calculate_solution_gradients(const std::vector<Solution_>& solutions)  const {
    const auto num_solution = solutions.size();
    std::vector<Solution_gradient_> solution_gradients(num_solution);

    const auto solution_delta_matrixes = this->calculate_solution_delta_matrixes(solutions);
    for (size_t i = 0; i < num_solution; ++i)
        ms::gemm(solution_delta_matrixes[i], this->least_square_matrixes_[i], solution_gradients[i]);

    return solution_gradients;
}

template <ushort num_equation_, ushort space_dimension_>
std::vector<Dynamic_Matrix> Least_Square_Base<num_equation_, space_dimension_>::calculate_solution_delta_matrixes(const std::vector<Solution_>& solutions) const {
    const auto num_cell = solutions.size();
    std::vector<Dynamic_Matrix> solution_delta_matrixes;
    solution_delta_matrixes.reserve(num_cell);

    for (size_t i = 0; i < num_cell; ++i) {
        const auto& near_cell_indexes = this->set_of_near_cell_indexes_[i];
        const auto& near_ghost_cell_indexes = this->set_of_near_ghost_cell_indexes_[i];

        const auto num_near_cell = near_cell_indexes.size();
        const auto num_near_ghost_cell = near_ghost_cell_indexes.size();
        const auto num_total_near_cell = num_near_cell + num_near_ghost_cell;

        Dynamic_Matrix solution_delta_matrix(num_equation_, num_total_near_cell);

        for (size_t j = 0; j < num_near_cell; ++j) {
            const auto solution_delta = solutions[near_cell_indexes[j]] - solutions[i];
            solution_delta_matrix.change_column(j, solution_delta);
        }

        for (size_t j = 0; j < num_near_ghost_cell; ++j) {
            const auto& ghost_cell = this->ghost_cells_[near_ghost_cell_indexes[j]];
            const auto ghost_cell_solution = ghost_cell.solution(solutions[ghost_cell.solution_related_cell_index]);

            const auto solution_delta = ghost_cell_solution - solutions[i];
            solution_delta_matrix.change_column(num_near_cell + j, solution_delta);
        }

        solution_delta_matrixes.push_back(std::move(solution_delta_matrix));
    }

    return solution_delta_matrixes;
}

//template <ushort num_equation_, ushort space_dimension_>
//Vertex_Least_Square<num_equation_, space_dimension_>::Vertex_Least_Square(const Grid<space_dimension_>& grid) {
//    SET_TIME_POINT;
//
//    const auto cell_center_nodes = grid.cell_center_nodes();
//    auto set_of_vertex_share_cell_indexes = grid.set_of_vertex_share_cell_indexes_consider_pbdry();
//
//    const auto num_cell = cell_center_nodes.size();
//    this->set_of_near_cell_indexes_.reserve(num_cell);
//    this->least_square_matrixes_.reserve(num_cell);
//
//    for (uint i = 0; i < num_cell; ++i) {
//        auto& vertex_share_cell_indexes = set_of_vertex_share_cell_indexes[i];
//        const auto num_vertex_share_cell = vertex_share_cell_indexes.size();
//
//        Dynamic_Matrix center_to_center_matrix(space_dimension_, num_vertex_share_cell);
//
//        const auto this_center = cell_center_nodes[i];
//        for (ushort j = 0; j < num_vertex_share_cell; ++j) {
//            const auto neighbor_center = cell_center_nodes[vertex_share_cell_indexes[j]];
//            const auto center_to_center_vector = neighbor_center - this_center;
//
//            center_to_center_matrix.change_column(j, center_to_center_vector);
//        }
//
//        const auto& Rc = center_to_center_matrix;
//        auto RcT = Rc.transpose();
//        auto least_square_matrix = RcT * (Rc * RcT).be_inverse();
//
//        this->set_of_near_cell_indexes_.push_back(std::move(vertex_share_cell_indexes));
//        this->least_square_matrixes_.push_back(std::move(least_square_matrix));
//    }
//
//    Log::content_ << std::left << std::setw(50) << "@ Vertex Least Sqaure precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
//    Log::print();
//
//    const auto& cell_elements = grid.get_grid_elements().cell_elements;
//
//    const auto num_cell = cell_elements.size();
//    this->near_cell_indexes_set_.reserve(num_cell);
//    this->least_square_matrixes_.reserve(num_cell);
//
//    auto set_of_vertex_share_cell_indexes = grid.set_of_vertex_share_cell_indexes_consider_pbdry();
//
//    for (size_t i = 0; i < num_cell; ++i) {
//        auto& vertex_share_cell_indexes = set_of_vertex_share_cell_indexes[i];
//        const auto num_vertex_share_cell = vertex_share_cell_indexes.size();
//
//        const auto& element = cell_elements[i];
//        const auto& geometry = element.geometry_;
//
//        Dynamic_Matrix center_to_center_matrix(space_dimension_, num_vertex_share_cell);
//
//        const auto this_center = geometry.center_node();
//
//        for (size_t j = 0; j < num_vertex_share_cell; ++j) {
//            const auto& neighbor_geometry = cell_elements[vertex_share_cell_indexes[j]].geometry_;
//            const auto neighbor_center = neighbor_geometry.center_node();
//            const auto center_to_center = neighbor_center - this_center;
//
//            center_to_center_matrix.change_column(j, center_to_center);
//        }
//
//        const auto& Rc = center_to_center_matrix;
//        auto RcT = Rc.transpose();
//        auto least_square_matrix = RcT * (Rc * RcT).be_inverse();
//
//        this->near_cell_indexes_set_.push_back(std::move(vertex_share_cell_indexes));
//        this->least_square_matrixes_.push_back(std::move(least_square_matrix));
//    }
//
//    Log::content_ << std::left << std::setw(50) << "@ Vertex Least Sqaure precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
//    Log::print();
//}



template <ushort num_equation_, ushort space_dimension_>
Face_Least_Square<num_equation_, space_dimension_>::Face_Least_Square(const Grid<space_dimension_>& grid) {
    SET_TIME_POINT;
    this->ghost_cells_ = grid.make_ghost_cells();    
    this->set_of_near_cell_indexes_ = grid.set_of_face_share_cell_indexes_ignore_pbdry();
    this->set_of_near_ghost_cell_indexes_ = grid.set_of_face_share_ghost_cell_indexes(this->ghost_cells_);

    const auto cell_center_nodes = grid.cell_center_nodes();
    
    const auto num_cell = cell_center_nodes.size();
    this->least_square_matrixes_.reserve(num_cell);

    for (uint i = 0; i < num_cell; ++i) {
        const auto& near_cell_indexes = this->set_of_near_cell_indexes_[i];
        const auto& near_ghost_cell_indexes = this->set_of_near_ghost_cell_indexes_[i];
        
        const auto num_near_cell = near_cell_indexes.size();
        const auto num_near_ghost_cell = near_ghost_cell_indexes.size();
        const auto num_total_near_cell = num_near_cell + num_near_ghost_cell;

        Dynamic_Matrix center_to_center_matrix(num_total_near_cell, space_dimension_);

        const auto this_center = cell_center_nodes[i];

        for (ushort j = 0; j < num_near_cell; ++j) {
            const auto near_cell_center = cell_center_nodes[near_cell_indexes[j]];
            const auto center_to_center_vector = near_cell_center - this_center;

            center_to_center_matrix.change_row(j, center_to_center_vector);
        }

        for (ushort j = 0; j < num_near_ghost_cell; ++j) {
            const auto near_ghost_cell_center = this->ghost_cells_[near_ghost_cell_indexes[j]].center_node;
            const auto center_to_center_vector = near_ghost_cell_center - this_center;

            center_to_center_matrix.change_row(num_near_cell + j, center_to_center_vector);
        }

        const auto& Rc = center_to_center_matrix;
        auto RcT = Rc.transpose();
        auto least_square_matrix = Rc * (RcT * Rc).be_inverse();

        this->least_square_matrixes_.push_back(std::move(least_square_matrix));
    }

    Log::content_ << std::left << std::setw(50) << "@ Face Least Sqaure precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
    Log::print();
}