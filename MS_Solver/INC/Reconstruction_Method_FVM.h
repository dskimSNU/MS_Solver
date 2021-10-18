#pragma once
#include "Spatial_Discrete_Method.h"
#include "Gradient_Method.h"
#include "PostAI.h"
#include "Tecplot.h"


class RM {};


class Constant_Reconstruction : public RM 
{
public:
    template <size_t space_dimension>
    Constant_Reconstruction(const Grid<space_dimension>& grid) {};
    static std::string name(void) { return "Constant_Reconstruction"; };
};


template <typename Gradient_Method>
class Linear_Reconstruction : public RM
{
private:
    static constexpr size_t num_equation_       = Gradient_Method::num_equation();
    static constexpr size_t space_dimension_    = Gradient_Method::space_dimension();

    using Solution_             = Euclidean_Vector<num_equation_>;
    using Solution_Gradient_    = Matrix<num_equation_, space_dimension_>;

public:
    static std::string name(void) { return "Linear_Reconstruction_" + Gradient_Method::name(); };

private:
    Gradient_Method gradient_method_;
    std::vector<Solution_Gradient_> solution_gradients_;

public:
    Linear_Reconstruction(const Grid<space_dimension_>& grid) : gradient_method_(grid) {};
    
public:
    void reconstruct(const std::vector<Solution_>& solutions);        
    const std::vector<Solution_Gradient_>& get_solution_gradients(void) const { return solution_gradients_; };
};


class MLP_u1_Limiting_Strategy
{
private:
    MLP_u1_Limiting_Strategy(void) = delete;

public:
    static double limiter_function(const double P1_mode_solution, const double P0_solution, const double allowable_min, const double allowable_max) {
        if (P1_mode_solution == 0)
            return 1.0;

        if (P1_mode_solution < 0)
            return (std::min)((allowable_min - P0_solution) / P1_mode_solution, 1.0);
        else
            return (std::min)((allowable_max - P0_solution) / P1_mode_solution, 1.0);
    };
};


template <typename Gradient_Method>
class MLP_u1 : public RM
{
private:
    static constexpr size_t num_equation_ = Gradient_Method::num_equation();
    static constexpr size_t space_dimension_ = Gradient_Method::space_dimension();

    using Solution_ = Euclidean_Vector<num_equation_>;
    using Solution_Gradient_ = Matrix<num_equation_, space_dimension_>;

public:
    static std::string name(void) { return "MLP_u1_" + Gradient_Method::name(); };

private:
    Gradient_Method gradient_method_;
    const std::unordered_map<uint, std::set<uint>>& vnode_index_to_share_cell_index_set_;
    std::vector<std::vector<uint>> set_of_vnode_indexes_;
    std::vector<Dynamic_Matrix> center_to_vertex_matrixes_;
    std::vector<Solution_Gradient_> solution_gradients_;

public:
    MLP_u1(const Grid<space_dimension_>& grid);

public:
    void reconstruct(const std::vector<Solution_>& solutions);
    void characteristic_reconstruct(const std::vector<Solution_>& solutions); // ds
    const std::vector<Solution_Gradient_>& get_solution_gradients(void) const { return this->solution_gradients_; };

protected:
    auto calculate_vertex_node_index_to_min_max_solution(const std::vector<Solution_>& solutions) const;
};


struct ANN_Model
{
    std::vector<Dynamic_Matrix> weights;
    std::vector<Dynamic_Euclidean_Vector> biases;
};


class ReLU {
public:
    double operator()(const double value) {
        return (value < 0) ? 0 : value;
    };
};

class Sigmoid {
public:
    double operator()(const double value) {
        return 1.0 / (1.0 + std::exp(-value));
    };
};


class HardTanh {
public:
    double operator()(const double value) {
        return (value > 1) ? 1 : (value < 0) ? 0 : value;
    };
};


template <typename Gradient_Method>
class ANN_limiter : public RM
{
private:
    static constexpr ushort num_equation_ = Gradient_Method::num_equation();
    static constexpr ushort space_dimension_ = Gradient_Method::space_dimension();

    using Solution_             = Euclidean_Vector<num_equation_>;
    using Solution_Gradient_    = Matrix<num_equation_, space_dimension_>;

    inline static std::string model_name_;

private:    
    Gradient_Method gradient_method_;    
    std::vector<Solution_Gradient_> solution_gradients_;
    std::vector<double> chracteristic_length_;
    std::vector<std::vector<uint>> set_of_indexes_;

public:
    ANN_limiter(const Grid<space_dimension_>& grid);

public:
    void reconstruct(const std::vector<Solution_>& solutions);    
    const std::vector<Solution_Gradient_>& get_solution_gradients(void) const { return solution_gradients_; };

private:
    double limit(Dynamic_Euclidean_Vector& feature) const;
    ANN_Model read_model(void) const;

public:
    static void set_model(const std::string_view model_name) { model_name_ = model_name; };
    static std::string name(void) { return "ANN_Reconstruction_" + model_name_; };
};

namespace ms {
	template <typename T>
	inline constexpr bool is_reconsturction_method = std::is_base_of_v<RM, T>;

    template <typename T>
    inline constexpr bool is_constant_reconustruction = std::is_same_v<Constant_Reconstruction, T>;


    template <typename Spatial_Discrete_Method, typename Reconstruction_Method, typename = void>
    inline constexpr bool is_default_reconstruction;
        
    template <typename Spatial_Discrete_Method, typename Reconstruction_Method>
    inline constexpr bool is_default_reconstruction<typename Spatial_Discrete_Method, typename Reconstruction_Method, std::enable_if_t<std::is_same_v<FVM, Spatial_Discrete_Method>>>
        = ms::is_constant_reconustruction<Reconstruction_Method>;
}


//template definition part
template <typename Gradient_Method>
void Linear_Reconstruction<Gradient_Method>::reconstruct(const std::vector<Solution_>& solutions) {
    this->solution_gradients_ = this->gradient_method_.calculate_solution_gradients(solutions);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////MLP-u1 limiter/////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gradient_Method>
MLP_u1<Gradient_Method>::MLP_u1(const Grid<space_dimension_>& grid) : gradient_method_(grid), vnode_index_to_share_cell_index_set_(grid.get_vnode_index_to_share_cell_index_set_consider_pbdry()) {
    SET_TIME_POINT;

    this->set_of_vnode_indexes_ = grid.cell_set_of_vnode_indexes();
    this->center_to_vertex_matrixes_ = grid.cell_center_to_vertex_matrixes();

    const auto num_cell = set_of_vnode_indexes_.size();
    this->solution_gradients_.resize(num_cell);

    Log::content_ << std::left << std::setw(50) << "@ MLP u1 precalculation" << " ----------- " << GET_TIME_DURATION << "s\n\n";
    Log::print();
}


template <typename Gradient_Method>
void MLP_u1<Gradient_Method>::reconstruct(const std::vector<Solution_>& solutions) {
    const auto solution_gradients = this->gradient_method_.calculate_solution_gradients(solutions);
    const auto vnode_index_to_min_max_solution = this->calculate_vertex_node_index_to_min_max_solution(solutions);

    Post_AI_Data::conditionally_record_solution_datas(solutions, solution_gradients);//postAI

    for (uint i = 0; i < solutions.size(); ++i) {
        const auto& gradient = solution_gradients[i];
        const auto P1_mode_solution_vnodes = gradient * this->center_to_vertex_matrixes_[i];

        std::array<double, num_equation_> limiting_values;
        limiting_values.fill(1);

        const auto& vnode_indexes = this->set_of_vnode_indexes_[i];
        const auto num_vertex = vnode_indexes.size();

        for (ushort j = 0; j < num_vertex; ++j) {
            const auto& [min_solution, max_solution] = vnode_index_to_min_max_solution.at(vnode_indexes[j]);

            for (ushort e = 0; e < num_equation_; ++e) {
                const auto limiting_value = MLP_u1_Limiting_Strategy::limiter_function(P1_mode_solution_vnodes.at(e, j), solutions[i].at(e), min_solution.at(e), max_solution.at(e));
                limiting_values[e] = (std::min)(limiting_values[e], limiting_value);
            }
        }
        Post_AI_Data::conditionally_record_limiting_value(i, limiting_values); //postAI

        const Matrix limiting_value_matrix = limiting_values;
        this->solution_gradients_[i] = limiting_value_matrix * gradient;
    }
    //Tecplot::conditionally_post_solution(solutions); //post
    Post_AI_Data::conditionally_post(); //postAI
};

template <typename Gradient_Method>
void MLP_u1<Gradient_Method>::characteristic_reconstruct(const std::vector<Solution_>& solutions) {
    const auto solution_gradients = this->gradient_method_.calculate_solution_gradients(solutions);

    const auto num_cells = solutions.size();
    const auto num_equations = solutions[0].size();
    const double gamma = 1.4;

    std::vector<Matrix<num_equations, num_equations>> set_of_K_matrices(num_cells_); //right eigenvectors
    for (int i = 0; i < num_cells_; i++) {
        const auto rho = solutions[i][0];
        const auto u = solutions[i][1] / solutions[i][0];
        const auto v = solutions[i][2] / solutions[i][0];
        const auto e = solutions[i][3] / solutions[i][0];
        const auto p = e * (gamma - 1.0) * rho;
        const auto H = 0.5 * (u * u + v * v) + e + p / rho;

        std::vector<double> K1 = { 1.0, u - a, v, H - a * u };
        std::vector<double> K2 = { 1.0, u, v, 0.5 * (u * u + v * v) };
        std::vector<double> K3 = { 0.0, 0.0, 1.0, v };
        std::vector<double> K4 = { 1.0, u + a, v, H + u * a };

    }

    const auto vnode_index_to_min_max_solution = this->calculate_vertex_node_index_to_min_max_solution(solutions);

    for (uint i = 0; i < solutions.size(); ++i) {
        const auto& gradient = solution_gradients[i];
        const auto P1_mode_solution_vnodes = gradient * this->center_to_vertex_matrixes_[i];

        std::array<double, num_equation_> limiting_values;
        limiting_values.fill(1);

        const auto& vnode_indexes = this->set_of_vnode_indexes_[i];
        const auto num_vertex = vnode_indexes.size();

        for (ushort j = 0; j < num_vertex; ++j) {
            const auto& [min_solution, max_solution] = vnode_index_to_min_max_solution.at(vnode_indexes[j]);

            for (ushort e = 0; e < num_equation_; ++e) {
                const auto limiting_value = MLP_u1_Limiting_Strategy::limiter_function(P1_mode_solution_vnodes.at(e, j), solutions[i].at(e), min_solution.at(e), max_solution.at(e));
                limiting_values[e] = (std::min)(limiting_values[e], limiting_value);
            }
        }

        const Matrix limiting_value_matrix = limiting_values;
        this->solution_gradients_[i] = limiting_value_matrix * gradient;
    }

};


template <typename Gradient_Method>
auto MLP_u1<Gradient_Method>::calculate_vertex_node_index_to_min_max_solution(const std::vector<Solution_>& solutions) const {
    const auto num_vnode = this->vnode_index_to_share_cell_index_set_.size();

    std::unordered_map<uint, std::pair<Solution_, Solution_>> vnode_index_to_min_max_solution;
    vnode_index_to_min_max_solution.reserve(num_vnode);

    for (const auto& [vnode_index, share_cell_indexes] : this->vnode_index_to_share_cell_index_set_) {
        const auto vnode_share_cell_solutions = ms::extract_by_index(solutions, share_cell_indexes);
        const auto gathered_min_sol = ms::min_value_gathering_vector(vnode_share_cell_solutions);
        const auto gathered_max_sol = ms::max_value_gathering_vector(vnode_share_cell_solutions);

        vnode_index_to_min_max_solution.emplace(vnode_index, std::make_pair(gathered_min_sol, gathered_max_sol));
    }

    return vnode_index_to_min_max_solution;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////ANN-based limiter////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gradient_Method>
ANN_limiter<Gradient_Method>::ANN_limiter(const Grid<space_dimension_>& grid) :gradient_method_(grid) {
    this->set_of_indexes_ = grid.ANN_indexes();

    const auto cell_volumes = grid.cell_volumes();
    const auto num_cell = cell_volumes.size();
    this->chracteristic_length_.resize(num_cell);

    for (uint i = 0; i < num_cell; ++i)
        this->chracteristic_length_[i] = std::pow(cell_volumes[i], 1.0 / space_dimension_);

}

template <typename Gradient_Method>
void ANN_limiter<Gradient_Method>::reconstruct(const std::vector<Solution_>& solutions) {
    this->solution_gradients_ = this->gradient_method_.calculate_solution_gradients(solutions);
    const auto initial_solution_gradients = this->solution_gradients_;

    const auto num_cell = solutions.size();    

    std::vector<double> post_limiter_value(num_cell);//post

    for (size_t i = 0; i < num_cell; ++i) {
        if (this->set_of_indexes_[i].empty()) {
            this->solution_gradients_[i] *= 0.0;
            continue;
        }

        const auto ordered_indexes = this->set_of_indexes_[i];
        const auto num_ordered_index = ordered_indexes.size();

        const auto num_input_values = 3 * num_ordered_index; //temporal code

        std::array<double, num_equation_> limiting_values = { 1 };

        for (size_t j = 0; j < num_equation_; ++j) {
            std::vector<double> input_values(num_input_values);

            for (size_t k = 0; k < num_ordered_index; ++k) {
                const auto index = ordered_indexes[k];

                const auto& solution = solutions[index];
                const auto& solution_gradient = initial_solution_gradients[index];

                const auto solution_start_index = 0;
                const auto solution_gradient_x_start_index = num_ordered_index;
                const auto solution_gradient_y_start_index = 2 * num_ordered_index;

                input_values[solution_start_index + k] = solution.at(j);
                input_values[solution_gradient_x_start_index + k] = solution_gradient.at(j, 0) * this->chracteristic_length_[i];
                input_values[solution_gradient_y_start_index + k] = solution_gradient.at(j, 1) * this->chracteristic_length_[i];
            }

            Dynamic_Euclidean_Vector input = std::move(input_values);
            limiting_values[j] = this->limit(input);
        }


        post_limiter_value[i] = limiting_values[0];//post

        const Matrix limiting_matrix = limiting_values;
        this->solution_gradients_[i] = limiting_matrix * this->solution_gradients_[i];
    }
}

template <typename Gradient_Method>
double ANN_limiter<Gradient_Method>::limit(Dynamic_Euclidean_Vector& feature) const {
    static const auto model = this->read_model();
    static const auto num_layer = model.weights.size();
    static const ReLU activation_function;
    static const Sigmoid output_function;
    //static const HardTanh output_function;

    //hidden layer
    for (ushort i = 0; i < num_layer - 1; ++i) {
        auto new_feature = model.biases[i];

        ms::gemvpv(model.weights[i], feature, new_feature);
        new_feature.apply(activation_function);
        feature = std::move(new_feature);
    }

    //output layer
    auto new_feature = model.biases[num_layer - 1];

    ms::gemvpv(model.weights[num_layer - 1], feature, new_feature);
    new_feature.apply(output_function);
    feature = std::move(new_feature);

    return feature[0];
}

template <typename Gradient_Method>
ANN_Model ANN_limiter<Gradient_Method>::read_model(void) const {    
    std::ifstream file("RSC/" + model_name_ + ".bin", std::ios::binary);
    dynamic_require(file.is_open(), "Model file should be open");

    int num_layer;
    file.read((char*)&num_layer, sizeof(int));

    ANN_Model model;
    model.weights.reserve(num_layer);
    model.biases.reserve(num_layer);

    for (int i = 0; i < num_layer; i++) {
        int num_row;
        int num_column;

        file.read((char*)&num_row, sizeof(int));
        file.read((char*)&num_column, sizeof(int));

        std::vector<double> weight(num_row * num_column);
        std::vector<double> bias(num_row);

        file.read((char*)&weight[0], sizeof(double) * num_row * num_column);
        file.read((char*)&bias[0], sizeof(double) * num_row);

        model.weights.push_back({ static_cast<size_t>(num_row), static_cast<size_t>(num_column), std::move(weight) });
        model.biases.push_back({ std::move(bias) });
    }

    return model;
}