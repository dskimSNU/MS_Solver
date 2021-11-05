#pragma once
#include "Governing_Equation.h"

#include <numbers>


class IC {}; //Initial Condition

//For 2D SCL
template <ushort space_dimension_>
class Constant1 : public IC 
{
private:
    Constant1(void) = delete;

private:
    static constexpr ushort num_eqation_ = 1;

    using This_         = Constant1<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_     = Euclidean_Vector<num_eqation_>;

public:
    static Euclidean_Vector<1> calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Euclidean_Vector<1>> calculate_solutions(const std::vector<Space_Vector_>& space_vectors);

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template<ushort space_dimension_>
class Sine_Wave : public IC 
{
// sin(kx)sin(ky)sin(kz)
private:
    Sine_Wave(void) = delete;

private:
    static constexpr ushort num_eqation_ = 1;

    using This_             = Sine_Wave<space_dimension_>;
    using Space_Vector_     = Euclidean_Vector<space_dimension_>;

private:
    inline static std::array<double, space_dimension_> wave_numbers_;

public:
    static void initialize(const std::array<double, space_dimension_>& wave_lengths);

public:
    static Euclidean_Vector<1> calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Euclidean_Vector<1>> calculate_solutions(const std::vector<Space_Vector_>& space_vectors);
    static std::vector<Euclidean_Vector<1>> calculate_exact_solutions(const std::vector<Space_Vector_>& space_vectors, const double end_time);

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Square_Wave : public IC 
{
private:
    Square_Wave(void) = delete;

private:
    static constexpr ushort num_eqation_ = 1;

    using This_         = Square_Wave<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_     = Euclidean_Vector<num_eqation_>;

public:
    static Euclidean_Vector<1> calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Euclidean_Vector<1>> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Circle_Wave : public IC 
{
private:
    Circle_Wave(void) = delete;

private:
    static constexpr ushort num_eqation_ = 1;

    using This_         = Circle_Wave<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_     = Euclidean_Vector<num_eqation_>;

public:
    static Euclidean_Vector<1> calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Euclidean_Vector<1>> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Gaussian_Wave : public IC 
{
private:
    Gaussian_Wave(void) = delete;

private:
    static constexpr ushort num_eqation_ = 1;

    using This_         = Gaussian_Wave<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_     = Euclidean_Vector<num_eqation_>;

public:
    static Euclidean_Vector<1> calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Euclidean_Vector<1>> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Big_Square_Wave : public IC
{
private:
    Big_Square_Wave(void) = delete;

private:
    static constexpr ushort num_eqation_ = 1;

    using This_ = Big_Square_Wave<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Euclidean_Vector<1> calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Euclidean_Vector<1>> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template<ushort space_dimension_>
class Big_Sine_Wave : public IC
{
    // sin(kx)
private:
    Big_Sine_Wave(void) = delete;

private:
    static constexpr ushort num_eqation_ = 1;

    using This_ = Sine_Wave<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;

private:
    inline static std::array<double, space_dimension_> wave_numbers_;

public:
    static void initialize(const std::array<double, space_dimension_>& wave_lengths);

public:
    static Euclidean_Vector<1> calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Euclidean_Vector<1>> calculate_solutions(const std::vector<Space_Vector_>& space_vectors);
    static std::vector<Euclidean_Vector<1>> calculate_exact_solutions(const std::vector<Space_Vector_>& space_vectors, const double end_time);

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};


//For 2D Euler
//Toro shock-tube 5 cases
template <ushort space_dimension_>
class Modified_SOD : public IC 
{
private:
    Modified_SOD(void) = delete;

private:
    static constexpr ushort num_eqation_ = 2 + space_dimension_;

    using This_         = Modified_SOD<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_     = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Supersonic_Expansion : public IC
{
private:
    Supersonic_Expansion(void) = delete;

private:
    static constexpr size_t num_eqation_ = 2 + space_dimension_;

    using This_ = Supersonic_Expansion<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Blast_Wave_Problem : public IC
{
private:
    Blast_Wave_Problem(void) = delete;

private:
    static constexpr size_t num_eqation_ = 2 + space_dimension_;

    using This_ = Blast_Wave_Problem<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Double_Strong_Shock_Problem : public IC
{
private:
    Double_Strong_Shock_Problem(void) = delete;

private:
    static constexpr size_t num_eqation_ = 2 + space_dimension_;

    using This_ = Double_Strong_Shock_Problem<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Slowly_Moving_Contact_Problem : public IC
{
private:
    Slowly_Moving_Contact_Problem(void) = delete;

private:
    static constexpr size_t num_eqation_ = 2 + space_dimension_;

    using This_ = Slowly_Moving_Contact_Problem<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class SOD : public IC
{
private:
    SOD(void) = delete;

private:
    static constexpr ushort num_eqation_ = 2 + space_dimension_;

    using This_ = SOD;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Harten_Lax_Problem : public IC
{
private:
    Harten_Lax_Problem(void) = delete;

private:
    static constexpr ushort num_eqation_ = 2 + space_dimension_;

    using This_ = Harten_Lax_Problem<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Blast_Wave_Interaction
{
private:
    Blast_Wave_Interaction(void) = delete;

private:
    static constexpr ushort num_eqation_ = 2 + space_dimension_;

    using This_ = Blast_Wave_Interaction<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Shu_Osher : public IC 
{
private:
    Shu_Osher(void) = delete;

private:
    static constexpr size_t num_eqation_ = 2 + space_dimension_;

    using This_         = Shu_Osher<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_     = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Explosion_Problem : public IC
{
private:
    Explosion_Problem(void) = delete;

private:
    static constexpr size_t num_eqation_ = 2 + space_dimension_;

    using This_ = Explosion_Problem<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

template <ushort space_dimension_>
class Riemann_Problem : public IC
{
private:
    Riemann_Problem(void) = delete;

private:
    static constexpr size_t num_eqation_ = 2 + space_dimension_;

    using This_ = Riemann_Problem<space_dimension_>;
    using Space_Vector_ = Euclidean_Vector<space_dimension_>;
    using Solution_ = Euclidean_Vector<num_eqation_>;

public:
    static Solution_ calculate_solution(const Space_Vector_& space_vector);
    static std::vector<Solution_> calculate_solutions(const std::vector<Space_Vector_>& cell_centers);//for FVM

public:
    static std::string name(void);
    static constexpr ushort space_dimension(void);
};

namespace ms {
    template <typename T>
    inline constexpr bool is_initial_condition = std::is_base_of_v<IC, T>;

    template <typename T>
    inline constexpr bool is_sine_wave = std::is_same_v<T, Sine_Wave<T::space_dimension()>>;
}


//Template Definition Part
//Constant1
template <ushort space_dimension_>
Euclidean_Vector<1> Constant1<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    return { 1 };
};
template <ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Constant1<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& space_vectors) {
    std::vector<Euclidean_Vector<1>> result(space_vectors.size(), { 1 });
    return result;
};
template <ushort space_dimension_>
std::string Constant1<space_dimension_>::name(void) {
    return "Constant1";
}
template <ushort space_dimension_>
constexpr ushort Constant1<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Sine wave
template <ushort space_dimension_>
void Sine_Wave<space_dimension_>::initialize(const std::array<double, space_dimension_>& wave_lengths) {
    for (ushort i = 0; i < space_dimension_; ++i)
        This_::wave_numbers_[i] = 2 * std::numbers::pi / wave_lengths[i];
        //This_::wave_numbers_[i] = 2 * std::numbers::pi / wave_lengths[0];
}
template <ushort space_dimension_>
Euclidean_Vector<1> Sine_Wave<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    double solution = 1.0;

    for (ushort i = 0; i < space_dimension_; ++i) 
        solution *= std::sin(This_::wave_numbers_[i] * space_vector[i]);

    //solution *= std::sin(This_::wave_numbers_[0] * space_vector[0]);

    return { solution };
}
template<ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Sine_Wave<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& space_vectors) {
    const auto num_cell = space_vectors.size();

    std::vector<Euclidean_Vector<1>> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(space_vectors[i]);

    return solutions;
}
template<ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Sine_Wave<space_dimension_>::calculate_exact_solutions(const std::vector<Space_Vector_>& space_vectors, const double end_time) {
    const auto advection_speed = Linear_Advection<space_dimension_>::advection_speed();

    const auto num_cell = space_vectors.size();
    std::vector<Euclidean_Vector<1>> exact_solutions(num_cell);

    for (size_t i = 0; i < num_cell; ++i) {
        const auto& cell_center = space_vectors[i];
        
        double exact_solution = 1.0;

        for (ushort j = 0; j < space_dimension_; ++j)
            exact_solution *= std::sin(This_::wave_numbers_[j] * (cell_center[j] - advection_speed[j] * end_time));

        exact_solutions[i] = exact_solution;
    }

    return exact_solutions;
}
template <ushort space_dimension_>
std::string Sine_Wave<space_dimension_>::name(void) {
    return "Sine_Wave";
};
template <ushort space_dimension_>
constexpr ushort Sine_Wave<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Square wave
template <ushort space_dimension_>
Euclidean_Vector<1> Square_Wave<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    for (ushort i = 0; i < space_dimension_; ++i) {
        if (space_vector[i] < 0.25 || 0.75 < space_vector[i])
            return { 0 };
    }
    return { 1 };    
}
template <ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Square_Wave<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();
    std::vector<Solution_> solutions(num_cell);

    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Square_Wave<space_dimension_>::name(void) {
    return "Square_Wave";
};
template <ushort space_dimension_>
constexpr ushort Square_Wave<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Circle wave
template <ushort space_dimension_>
Euclidean_Vector<1> Circle_Wave<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    double temp = 0.0;
    for (ushort i = 0; i < space_dimension_; ++i) 
        temp += (space_vector[i] - 0.5) * (space_vector[i] - 0.5);

    if (temp <= 0.25 * 0.25)
        return { 1 };
    else
        return { 0 };
}
template <ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Circle_Wave<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();
    std::vector<Solution_> solutions(num_cell);

    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Circle_Wave<space_dimension_>::name(void) {
    return "Circle_Wave";
};
template <ushort space_dimension_>
constexpr ushort Circle_Wave<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Gaussian wave
template <ushort space_dimension_>
Euclidean_Vector<1> Gaussian_Wave<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr double beta = 20.0;

    double temp = 0.0;
    for (ushort i = 0; i < space_dimension_; ++i)
        temp += (space_vector[i] - 0.5) * (space_vector[i] - 0.5);

    return { std::exp(-beta * temp) };
}
template <ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Gaussian_Wave<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();
    std::vector<Solution_> solutions(num_cell);

    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Gaussian_Wave<space_dimension_>::name(void) {
    return "Gaussian_Wave";
};
template <ushort space_dimension_>
constexpr ushort Gaussian_Wave<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Big Square wave
template <ushort space_dimension_>
Euclidean_Vector<1> Big_Square_Wave<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    if (space_vector[0] < 0.25 || 0.75 < space_vector[0])
        return { 0 };
    
    return { 1 };    
}
template <ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Big_Square_Wave<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();
    std::vector<Solution_> solutions(num_cell);

    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Big_Square_Wave<space_dimension_>::name(void) {
    return "Big_Square_Wave";
};
template <ushort space_dimension_>
constexpr ushort Big_Square_Wave<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Big sine wave
template <ushort space_dimension_>
void Big_Sine_Wave<space_dimension_>::initialize(const std::array<double, space_dimension_>& wave_lengths) {
    for (ushort i = 0; i < space_dimension_; ++i)
        This_::wave_numbers_[i] = 2 * std::numbers::pi / wave_lengths[0];
}
template <ushort space_dimension_>
Euclidean_Vector<1> Big_Sine_Wave<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    double solution = 1.0;

    solution *= std::sin(This_::wave_numbers_[0] * space_vector[0]);

    return { solution };
}
template<ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Big_Sine_Wave<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& space_vectors) {
    const auto num_cell = space_vectors.size();

    std::vector<Euclidean_Vector<1>> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(space_vectors[i]);

    return solutions;
}
template<ushort space_dimension_>
std::vector<Euclidean_Vector<1>> Big_Sine_Wave<space_dimension_>::calculate_exact_solutions(const std::vector<Space_Vector_>& space_vectors, const double end_time) {
    //확인 필요
    const auto advection_speed = Linear_Advection<space_dimension_>::advection_speed();

    const auto num_cell = space_vectors.size();
    std::vector<Euclidean_Vector<1>> exact_solutions(num_cell);

    for (size_t i = 0; i < num_cell; ++i) {
        const auto& cell_center = space_vectors[i];

        double exact_solution = 1.0;

        for (ushort j = 0; j < space_dimension_; ++j)
            exact_solution *= std::sin(This_::wave_numbers_[j] * (cell_center[j] - advection_speed[j] * end_time));

        exact_solutions[i] = exact_solution;
    }

    return exact_solutions;
}
template <ushort space_dimension_>
std::string Big_Sine_Wave<space_dimension_>::name(void) {
    return "Big_Sine_Wave";
};
template <ushort space_dimension_>
constexpr ushort Big_Sine_Wave<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//SOD
template <ushort space_dimension_>
SOD<space_dimension_>::Solution_ SOD<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1 / (gamma - 1);
    constexpr auto discontinuity_location = 0.5;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 1.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 0.125;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 0.1;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ ==3) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 1.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 0.125;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 0.1;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename SOD<space_dimension_>::Solution_> SOD<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string SOD<space_dimension_>::name(void) {
    return "SOD";
};
template <ushort space_dimension_>
constexpr ushort SOD<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Harten-Lax
template <ushort space_dimension_>
Harten_Lax_Problem<space_dimension_>::Solution_ Harten_Lax_Problem<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1.0 / (gamma - 1.0);
    constexpr auto discontinuity_location = 0.5;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 0.445;
            constexpr auto u = 0.698;
            constexpr auto v = 0.0;
            constexpr auto p = 3.528;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 0.5;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 0.571;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 0.445;
            constexpr auto u = 0.698;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 3.528;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 0.5;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 0.571;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Harten_Lax_Problem<space_dimension_>::Solution_> Harten_Lax_Problem<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Harten_Lax_Problem<space_dimension_>::name(void) {
    return "Harten_Lax_Problem";
};
template <ushort space_dimension_>
constexpr ushort Harten_Lax_Problem<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Blast-wave interaction
template <ushort space_dimension_>
Blast_Wave_Interaction<space_dimension_>::Solution_ Blast_Wave_Interaction<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1.0 / (gamma - 1.0);
    constexpr auto discontinuity_location_1 = 0.1;
    constexpr auto discontinuity_location_2 = 0.9;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location_1) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 1000.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else if ((x_coordinate >= discontinuity_location_1) && (x_coordinate <= discontinuity_location_2)) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 0.01;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 100.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (x_coordinate <= discontinuity_location_1) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 1000.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else if ((x_coordinate >= discontinuity_location_1) && (x_coordinate <= discontinuity_location_2)) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 0.01;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 100.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Blast_Wave_Interaction<space_dimension_>::Solution_> Blast_Wave_Interaction<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Blast_Wave_Interaction<space_dimension_>::name(void) {
    return "Blast_Wave_Interaction";
};
template <ushort space_dimension_>
constexpr ushort Blast_Wave_Interaction<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Shu-Osher
template <ushort space_dimension_>
Shu_Osher<space_dimension_>::Solution_ Shu_Osher<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1 / (gamma - 1);
    constexpr auto pi = std::numbers::pi;
    constexpr auto discontinuity_location = 0.125;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 3.857143;
            constexpr auto u = 2.629369;    
            constexpr auto v = 0.0;
            constexpr auto p = 10.333333;   

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            const auto rho = 1 + 0.2 * std::sin(16 * pi * x_coordinate);
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 1.0;

            const auto rhou = rho * u;
            const auto rhov = rho * v;
            const auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 3.857143;
            constexpr auto u = 2.629369;    
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 10.333333;   

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            const auto rho = 1 + 0.2 * std::sin(16 * pi * x_coordinate);
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 1.0;

            const auto rhou = rho * u;
            const auto rhov = rho * v;
            const auto rhow = rho * w;
            const auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Shu_Osher<space_dimension_>::Solution_> Shu_Osher<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Shu_Osher<space_dimension_>::name(void) {
    return "Shu_Osher";
};
template <ushort space_dimension_>
constexpr ushort Shu_Osher<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Explosion
template <ushort space_dimension_>
Explosion_Problem<space_dimension_>::Solution_ Explosion_Problem<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1 / (gamma - 1);
    constexpr auto discontinuity_radius = 0.4;

    const auto x_coordinate = space_vector.at(0);
    const auto y_coordinate = space_vector.at(1);
    const auto radius = std::sqrt(x_coordinate * x_coordinate + y_coordinate * y_coordinate);

    if constexpr (space_dimension_ == 2) {
        if (radius < discontinuity_radius) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;   
            constexpr auto v = 0.0;
            constexpr auto p = 1.0;   

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 0.125;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 0.1;

            const auto rhou = rho * u;
            const auto rhov = rho * v;
            const auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (radius < discontinuity_radius) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;    
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 1.0;   

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 0.125;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 0.1;

            const auto rhou = rho * u;
            const auto rhov = rho * v;
            const auto rhow = rho * w;
            const auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Explosion_Problem<space_dimension_>::Solution_> Explosion_Problem<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Explosion_Problem<space_dimension_>::name(void) {
    return "Explosion_Problem";
};
template <ushort space_dimension_>
constexpr ushort Explosion_Problem<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//Explosion
template <ushort space_dimension_>
Riemann_Problem<space_dimension_>::Solution_ Riemann_Problem<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1 / (gamma - 1);

    const auto x_coordinate = space_vector.at(0);
    const auto y_coordinate = space_vector.at(1);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate >= 0.8 && x_coordinate <= 1.0 && y_coordinate >= 0.8 && y_coordinate <= 1.0) {
            constexpr auto rho = 1.5;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 1.5;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }

        else if (x_coordinate >= 0.0 && x_coordinate <= 0.8 && y_coordinate >= 0.8 && y_coordinate <= 1.0) {
            constexpr auto rho = 0.5323;
            constexpr auto u = 1.206;
            constexpr auto v = 0.0;
            constexpr auto p = 0.3;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }

        else if (x_coordinate >= 0.0 && x_coordinate <= 0.8 && y_coordinate >= 0.0 && y_coordinate <= 0.8) {
            constexpr auto rho = 0.1380;
            constexpr auto u = 1.206;
            constexpr auto v = 1.206;
            constexpr auto p = 0.029;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }

        else {
            constexpr auto rho = 0.5323;
            constexpr auto u = 1.206;
            constexpr auto v = 0.0;
            constexpr auto p = 0.3;

            const auto rhou = rho * u;
            const auto rhov = rho * v;
            const auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }

    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Riemann_Problem<space_dimension_>::Solution_> Riemann_Problem<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Riemann_Problem<space_dimension_>::name(void) {
    return "Riemann_Problem";
};
template <ushort space_dimension_>
constexpr ushort Riemann_Problem<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//T1. Modified SOD
template <ushort space_dimension_>
Modified_SOD<space_dimension_>::Solution_ Modified_SOD<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1.0 / (gamma - 1.0);
    constexpr auto discontinuity_location = 0.3;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.75;
            constexpr auto v = 0.0;
            constexpr auto p = 1.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 0.125;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 0.1;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.75;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 1.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 0.125;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 0.1;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Modified_SOD<space_dimension_>::Solution_> Modified_SOD<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Modified_SOD<space_dimension_>::name(void) {
    return "T1_Modified_SOD";
};
template <ushort space_dimension_>
constexpr ushort Modified_SOD<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//T2. Supersonic expansion
template <ushort space_dimension_>
Supersonic_Expansion<space_dimension_>::Solution_ Supersonic_Expansion<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1.0 / (gamma - 1.0);
    constexpr auto discontinuity_location = 0.5;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = -2.0;
            constexpr auto v = 0.0;
            constexpr auto rhoE = 3.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 1.0;
            constexpr auto u = 2.0;
            constexpr auto v = 0.0;
            constexpr auto rhoE = 3.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = -2.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto rhoE = 3.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 1.0;
            constexpr auto u = 2.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto rhoE = 3.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Supersonic_Expansion<space_dimension_>::Solution_> Supersonic_Expansion<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Supersonic_Expansion<space_dimension_>::name(void) {
    return "T2_Supersonic_Expansion";
};
template <ushort space_dimension_>
constexpr ushort Supersonic_Expansion<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//T3. Blast wave problem
template <ushort space_dimension_>
Blast_Wave_Problem<space_dimension_>::Solution_ Blast_Wave_Problem<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1.0 / (gamma - 1.0);
    constexpr auto discontinuity_location = 0.5;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 1000.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto p = 0.01;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 1000.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 1.0;
            constexpr auto u = 0.0;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 0.01;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Blast_Wave_Problem<space_dimension_>::Solution_> Blast_Wave_Problem<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Blast_Wave_Problem<space_dimension_>::name(void) {
    return "T3_Blast_Wave_Problem";
};
template <ushort space_dimension_>
constexpr ushort Blast_Wave_Problem<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//T4. Double strong shock problem
template <ushort space_dimension_>
Double_Strong_Shock_Problem<space_dimension_>::Solution_ Double_Strong_Shock_Problem<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1.0 / (gamma - 1.0);
    constexpr auto discontinuity_location = 0.4;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 5.99924;
            constexpr auto u = 19.5975;
            constexpr auto v = 0.0;
            constexpr auto p = 460.894;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 5.99242;
            constexpr auto u = -6.19633;
            constexpr auto v = 0.0;
            constexpr auto p = 46.0950;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 5.99924;
            constexpr auto u = 19.5975;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 460.894;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 5.99242;
            constexpr auto u = -6.19633;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 46.0950;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Double_Strong_Shock_Problem<space_dimension_>::Solution_> Double_Strong_Shock_Problem<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Double_Strong_Shock_Problem<space_dimension_>::name(void) {
    return "T4_Double_Strong_Shock_Problem";
};
template <ushort space_dimension_>
constexpr ushort Double_Strong_Shock_Problem<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}

//T5.Slowly-moving contact problem
template <ushort space_dimension_>
Slowly_Moving_Contact_Problem<space_dimension_>::Solution_ Slowly_Moving_Contact_Problem<space_dimension_>::calculate_solution(const Space_Vector_& space_vector) {
    constexpr auto gamma = 1.4;
    constexpr auto c = 1.0 / (gamma - 1.0);
    constexpr auto discontinuity_location = 0.8;

    const auto x_coordinate = space_vector.at(0);

    if constexpr (space_dimension_ == 2) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = -19.59745;
            constexpr auto v = 0.0;
            constexpr auto p = 1000.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
        else {
            constexpr auto rho = 1.0;
            constexpr auto u = -19.59745;
            constexpr auto v = 0.0;
            constexpr auto p = 0.01;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v);

            return { rho, rhou, rhov, rhoE };
        }
    }
    else if constexpr (space_dimension_ == 3) {
        if (x_coordinate <= discontinuity_location) {
            constexpr auto rho = 1.0;
            constexpr auto u = -19.59745;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 1000.0;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
        else {
            constexpr auto rho = 1.0;
            constexpr auto u = -19.59745;
            constexpr auto v = 0.0;
            constexpr auto w = 0.0;
            constexpr auto p = 0.01;

            constexpr auto rhou = rho * u;
            constexpr auto rhov = rho * v;
            constexpr auto rhow = rho * w;
            constexpr auto rhoE = p * c + 0.5 * (rhou * u + rhov * v + rhow * w);

            return { rho, rhou, rhov, rhow, rhoE };
        }
    }
    else {
        throw std::runtime_error("not supported space dimension");
        return {};
    }
}
template <ushort space_dimension_>
std::vector<typename Slowly_Moving_Contact_Problem<space_dimension_>::Solution_> Slowly_Moving_Contact_Problem<space_dimension_>::calculate_solutions(const std::vector<Space_Vector_>& cell_centers) {
    const auto num_cell = cell_centers.size();

    std::vector<Solution_> solutions(num_cell);
    for (size_t i = 0; i < num_cell; ++i)
        solutions[i] = This_::calculate_solution(cell_centers[i]);

    return solutions;
}
template <ushort space_dimension_>
std::string Slowly_Moving_Contact_Problem<space_dimension_>::name(void) {
    return "T5_Slowly_Moving_Contact_Problem";
};
template <ushort space_dimension_>
constexpr ushort Slowly_Moving_Contact_Problem<space_dimension_>::space_dimension(void) {
    return space_dimension_;
}