#pragma once
#include "Matrix.h"
#include "Debugger.h"

class GE {}; // Governing Equation


template <ushort space_dimension_>
class SCL : public GE    //Scalar Conservation Law 
{
private:
    SCL(void) = delete;

protected:
    static constexpr ushort num_equation_ = 1;

public:
    static constexpr ushort space_dimension(void) { return space_dimension_; };
    static constexpr ushort num_equation(void) { return num_equation_; };
};


template <ushort space_dimension>
class Linear_Advection : public SCL<space_dimension>
{
private:
    Linear_Advection(void) = delete;

private:
    using This_             = Linear_Advection<space_dimension>;
    using Space_Vector_     = Euclidean_Vector<space_dimension>;
    using Solution_         = Euclidean_Vector<This_::num_equation_>;
    using Physical_Flux_    = Matrix<This_::num_equation_, space_dimension>;

private:
    inline static Euclidean_Vector<space_dimension> advection_speeds_;

public:
    static void initialize(const Euclidean_Vector<space_dimension>& advection_speed) { advection_speeds_ = advection_speed; };

public:
    static Euclidean_Vector<space_dimension> advection_speed(void) { return advection_speeds_; };
    static auto physical_flux(const This_::Solution_& solution);
    static auto physical_fluxes(const std::vector<This_::Solution_>& solutions);
    static std::vector<std::array<double, space_dimension>> calculate_coordinate_projected_maximum_lambdas(const std::vector<Solution_>& solutions);
    static double inner_face_maximum_lambda(const Solution_& solution_o, const Solution_& solution_n, const Space_Vector_& nomal_vector);

public:
    static std::string name(void) { return "Linear_Advection_" + std::to_string(space_dimension) + "D"; }; 
};


template <ushort space_dimension>
class Burgers : public SCL<space_dimension>
{
private:
    Burgers(void) = delete;

private:
    using This_             = Burgers<space_dimension>;
    using Space_Vector_     = Euclidean_Vector<space_dimension>;
    using Solution_         = Euclidean_Vector<This_::num_equation_>;
    using Physical_Flux_    = Matrix<This_::num_equation_, space_dimension>;

public:
    static auto physical_flux(const Solution_& solution);
    static auto physical_fluxes(const std::vector<Solution_>& solutions);
    static std::vector<std::array<double, space_dimension>> calculate_coordinate_projected_maximum_lambdas(const std::vector<Solution_>& solutions);
    static double inner_face_maximum_lambda(const Solution_& solution_o, const Solution_& solution_n, const Space_Vector_& nomal_vector);

public:
    static std::string name(void) { return "Burgers_" + std::to_string(space_dimension) + "D"; };
};


template <ushort space_dimension_>
class Euler : public GE
{
private:
    Euler(void) = delete;

private:
    static constexpr ushort num_equation_ = 2 + space_dimension_;

    using This_                 = Euler<space_dimension_>;
    using Space_Vector_         = Euclidean_Vector<space_dimension_>;
    using Solution_             = Euclidean_Vector<num_equation_>;
    using Physical_Flux_        = Matrix<num_equation_, space_dimension_>;

public:
    static auto physical_flux(const Solution_& conservative_variable);
    static auto physical_flux(const Solution_& conservative_variable, const Solution_& primitivie_variable);
    static auto physical_fluxes(const std::vector<Solution_>& conservative_variables, const std::vector<Solution_>& primitive_variables);
    static std::vector<std::array<double, space_dimension_>> calculate_coordinate_projected_maximum_lambdas(const std::vector<Solution_>& conservative_variables);
    static double inner_face_maximum_lambda(const Solution_& oc_primitive_variable, const Solution_& nc_primitive_variable, const Space_Vector_& nomal_vector);
    
public:
    static Solution_ conservative_to_primitive(const Solution_& conservative_variable);
    static constexpr ushort space_dimension(void) { return space_dimension_; };
    static constexpr ushort num_equation(void) { return num_equation_; };
    static std::string name(void) { return "Euler" + std::to_string(space_dimension_) + "D"; };
};


namespace ms {
    template <typename T>
    inline constexpr bool is_governing_equation = std::is_base_of_v<GE, T>;
    template <typename T>
    inline constexpr bool is_SCL = std::is_base_of_v<SCL<T::space_dimension()>, T>;
    template <typename T>
    inline constexpr bool is_linear_advection = std::is_same_v<Linear_Advection<T::space_dimension()>, T>;
    template <typename T>
    inline constexpr bool is_Euler = std::is_same_v<Euler<T::space_dimension()>, T>;
}


//Template Definition Part
template <ushort space_dimension>
auto Linear_Advection<space_dimension>::physical_flux(const Solution_& solution) {
    std::array<double, space_dimension> flux_values;
    for (ushort i = 0; i < space_dimension; ++i)
        flux_values[i] = This_::advection_speeds_[i] * solution;
    
    return Matrix<1, space_dimension>(flux_values);
}

template <ushort space_dimension>
auto Linear_Advection<space_dimension>::physical_fluxes(const std::vector<Solution_>& solutions) {
    const auto num_solution = solutions.size();    
    std::vector<This_::Physical_Flux_> physical_fluxes(num_solution);

    for (size_t i = 0; i < num_solution; ++i)
        physical_fluxes[i] = This_::physical_flux(solutions[i]);

    return physical_fluxes;
}

template <ushort space_dimension>
std::vector<std::array<double, space_dimension>> Linear_Advection<space_dimension>::calculate_coordinate_projected_maximum_lambdas(const std::vector<Solution_>& solutions) {
    const auto num_solution = solutions.size();
    
    std::array<double, space_dimension> projected_maximum_lambda;
    for (ushort i = 0; i < space_dimension; ++i)
        projected_maximum_lambda[i] = std::abs(This_::advection_speeds_[i]);
    
    return std::vector<std::array<double, space_dimension>>(num_solution, projected_maximum_lambda);
}

template <ushort space_dimension>
double Linear_Advection<space_dimension>::inner_face_maximum_lambda(const Solution_& solution_o, const Solution_& solution_n, const Space_Vector_& nomal_vector) {
    return std::abs(This_::advection_speeds_.inner_product(nomal_vector));
}

template <ushort space_dimension>
auto Burgers<space_dimension>::physical_flux(const Solution_& solution) {
    //const auto sol = solution.at(0); //scalar
    const double flux_value = 0.5 * std::pow(solution,2.0);

    std::array<double, space_dimension> flux_values;
    flux_values.fill(flux_value);

    return Physical_Flux_(flux_values);
}

template <ushort space_dimension>
auto Burgers<space_dimension>::physical_fluxes(const std::vector<Solution_>& solutions) {
    const auto num_solution = solutions.size();
    std::vector<This_::Physical_Flux_> physical_fluxes(num_solution);

    for (size_t i = 0; i < num_solution; ++i)
        physical_fluxes[i] = This_::physical_flux(solutions[i]);

    return physical_fluxes;
}

template <ushort space_dimension>
std::vector<std::array<double, space_dimension>> Burgers<space_dimension>::calculate_coordinate_projected_maximum_lambdas(const std::vector<Solution_>& solutions) {
    const auto num_solution = solutions.size();

    std::vector<std::array<double, space_dimension>> projected_maximum_lambdas(num_solution);
    for (size_t i = 0; i < num_solution; ++i) {
        const auto maximum_lambdas = std::abs(solutions[i]);
        projected_maximum_lambdas[i].fill(maximum_lambdas);
    }

    return projected_maximum_lambdas;
}

template <ushort space_dimension>
double Burgers<space_dimension>::inner_face_maximum_lambda(const Solution_& solution_o, const Solution_& solution_n, const Space_Vector_& normal_vector) {
    double normal_component_sum = 0.0;
    for (ushort i = 0; i < space_dimension; ++i)
        normal_component_sum += normal_vector[i];
            
    return (std::max)(std::abs(solution_o * normal_component_sum), std::abs(solution_n * normal_component_sum));
}


template <ushort space_dimension_>
auto Euler<space_dimension_>::physical_flux(const Solution_& cvariable) {
    const auto pvariable = conservative_to_primitive(cvariable);
    return physical_flux(cvariable, pvariable);
}

template <ushort space_dimension_>
auto Euler<space_dimension_>::physical_flux(const Solution_& conservative_variable, const Solution_& primitivie_variable) {
    if constexpr (space_dimension_ == 2) {
        const auto rho = conservative_variable.at(0);
        const auto rhou = conservative_variable.at(1);
        const auto rhov = conservative_variable.at(2);
        const auto rhoE = conservative_variable.at(3);
        const auto u = primitivie_variable.at(0);
        const auto v = primitivie_variable.at(1);
        const auto p = primitivie_variable.at(2);
        const auto a = primitivie_variable.at(3);


        const auto rhouv = rhou * v;

        dynamic_require(rho >= 0 && p >= 0, "density and pressure shold be positive");

        return Matrix<num_equation_,space_dimension_>({
            rhou,				rhov,
            rhou * u + p,		rhouv,
            rhouv,				rhov * v + p,
            (rhoE + p) * u,		(rhoE + p) * v
        });
    }
    else {
        const auto rho = conservative_variable[0];
        const auto rhou = conservative_variable[1];
        const auto rhov = conservative_variable[2];
        const auto rhow = conservative_variable[3];
        const auto rhoE = conservative_variable[4];
        const auto u = primitivie_variable[0];
        const auto v = primitivie_variable[1];
        const auto w = primitivie_variable[2];
        const auto p = primitivie_variable[3];
        const auto a = primitivie_variable[4];

        const auto rhouv = rhou * v;
        const auto rhouw = rhou * w;
        const auto rhovw = rhov * w;

        dynamic_require(rho >= 0 && p >= 0, "density and pressure shold be positive");

        return Matrix<num_equation_, space_dimension_>({
            rhou,				rhov,               rhow,
            rhou * u + p,		rhouv,              rhouw,
            rhouv,				rhov * v + p,       rhovw,
            rhouw,              rhovw,              rhow * w + p,
            (rhoE + p) * u,		(rhoE + p) * v,      (rhoE + p) * w
        });
    }
}

template <ushort space_dimension_>
auto Euler<space_dimension_>::physical_fluxes(const std::vector<Solution_>& conservative_variables, const std::vector<Solution_>& primitive_variables) {
    const size_t num_solution = conservative_variables.size();

    std::vector<Physical_Flux_> physical_fluxes;
    physical_fluxes.reserve(num_solution);

    for (size_t i = 0; i < num_solution; ++i)
        physical_fluxes.push_back(physical_flux(conservative_variables[i], primitive_variables[i]));

    return physical_fluxes;
}

template <ushort space_dimension_>
std::vector<std::array<double, space_dimension_>> Euler<space_dimension_>::calculate_coordinate_projected_maximum_lambdas(const std::vector<Solution_>& conservative_variables) {
    auto num_solution = conservative_variables.size();

    std::vector<Solution_> primitive_variables;
    primitive_variables.reserve(num_solution);

    for (size_t i = 0; i < num_solution; ++i)
        primitive_variables.push_back(Euler<space_dimension_>::conservative_to_primitive(conservative_variables[i]));

    std::vector<std::array<double, space_dimension_>> coordinate_projected_maximum_lambdas(num_solution);

    for (size_t i = 0; i < num_solution; ++i) {
        if constexpr (space_dimension_ == 2) {
            const auto u = primitive_variables[i].at(0);
            const auto v = primitive_variables[i].at(1);
            const auto a = primitive_variables[i].at(3);

            const auto x_projected_maximum_lambda = std::abs(u) + a;
            const auto y_projected_maximum_lambda = std::abs(v) + a;

            coordinate_projected_maximum_lambdas[i] = { x_projected_maximum_lambda, y_projected_maximum_lambda };
        }
        else {
            const auto u = primitive_variables[i].at(0);
            const auto v = primitive_variables[i].at(1);
            const auto w = primitive_variables[i].at(2);
            const auto a = primitive_variables[i].at(4);

            const auto x_projected_maximum_lambda = std::abs(u) + a;
            const auto y_projected_maximum_lambda = std::abs(v) + a;
            const auto z_projected_maximum_lambda = std::abs(w) + a;

            coordinate_projected_maximum_lambdas[i] = { x_projected_maximum_lambda, y_projected_maximum_lambda, z_projected_maximum_lambda };
        }
    }

    return coordinate_projected_maximum_lambdas;
}


template <ushort space_dimension_>
double Euler<space_dimension_>::inner_face_maximum_lambda(const Solution_& oc_primitive_variable, const Solution_& nc_primitive_variable, const Space_Vector_& normal_vector) {
    double oc_side_face_maximum_lambda = 0;
    double nc_side_face_maximum_lambda = 0;
    
    if (space_dimension_ == 2) {
        const auto oc_u = oc_primitive_variable[0];
        const auto oc_v = oc_primitive_variable[1];
        const auto oc_a = oc_primitive_variable[3];
        oc_side_face_maximum_lambda = std::abs(oc_u * normal_vector[0] + oc_v * normal_vector[1]) + oc_a;

        const auto nc_u = nc_primitive_variable[0];
        const auto nc_v = nc_primitive_variable[1];
        const auto nc_a = nc_primitive_variable[3];
        nc_side_face_maximum_lambda = std::abs(nc_u * normal_vector[0] + nc_v * normal_vector[1]) + nc_a;
    }
    else {
        const auto oc_u = oc_primitive_variable[0];
        const auto oc_v = oc_primitive_variable[1];
        const auto oc_w = oc_primitive_variable[2];
        const auto oc_a = oc_primitive_variable[4];
        oc_side_face_maximum_lambda = std::abs(oc_u * normal_vector[0] + oc_v * normal_vector[1] + oc_w * normal_vector[2]) + oc_a;

        const auto nc_u = nc_primitive_variable[0];
        const auto nc_v = nc_primitive_variable[1];
        const auto nc_w = nc_primitive_variable[2];
        const auto nc_a = nc_primitive_variable[4];
        nc_side_face_maximum_lambda = std::abs(nc_u * normal_vector[0] + nc_v * normal_vector[1] + nc_w * normal_vector[2]) + nc_a;
    }

    return (std::max)(oc_side_face_maximum_lambda, nc_side_face_maximum_lambda);
}


template <ushort space_dimension_>
Euler<space_dimension_>::Solution_ Euler<space_dimension_>::conservative_to_primitive(const Solution_& conservative_variable) {
    constexpr auto gamma = 1.4;

    if constexpr (space_dimension_ == 2) {
        const auto rho = conservative_variable[0];
        const auto rhou = conservative_variable[1];
        const auto rhov = conservative_variable[2];
        const auto rhoE = conservative_variable[3];

        const auto one_over_rho = 1.0 / rho;

        const auto u = rhou * one_over_rho;
        const auto v = rhov * one_over_rho;
        const auto p = (rhoE - 0.5 * (rhou * u + rhov * v)) * (gamma - 1);
        const auto a = std::sqrt(gamma * p * one_over_rho);

        return { u,v,p,a };
    }
    else {
        const auto rho = conservative_variable[0];
        const auto rhou = conservative_variable[1];
        const auto rhov = conservative_variable[2];
        const auto rhow = conservative_variable[3];
        const auto rhoE = conservative_variable[4];

        const auto one_over_rho = 1.0 / rho;

        const auto u = rhou * one_over_rho;
        const auto v = rhov * one_over_rho;
        const auto w = rhow * one_over_rho;
        const auto p = (rhoE - 0.5 * (rhou * u + rhov * v + rhow * w)) * (gamma - 1);
        const auto a = std::sqrt(gamma * p * one_over_rho);

        return { u,v,w,p,a };
    }
}