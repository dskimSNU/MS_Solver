#pragma once
#include "Debugger.h"
#include "Semi_Discrete_Equation.h"
#include "Time_Integral_Method.h"
#include "Time_Step_Method.h"
#include "Solve_Controller.h"
#include "Tecplot.h"

template <typename Time_Integral_Method>
class Discrete_Equation
{
private:
    static_require(ms::is_time_integral_method<Time_Integral_Method>, "It should be time integral method");

public: 
    template<typename Time_Step_Method, typename Semi_Discrete_Equation, typename Solution>
    static void solve(Semi_Discrete_Equation& semi_discrete_equation, std::vector<Solution>& solutions) {
        double current_time = 0.0;
        Tecplot::syncronize_time(current_time);

        Log::content_ << "================================================================================\n";
        Log::content_ << "\t\t\t\t Solving\n";
        Log::content_ << "================================================================================\n";
                
        //Tecplot::post_solution(solutions, "initial");//post
        Tecplot::post_condition_ = true;//post
        //Post_AI_Data::post_condition_ = true;//postAI
        semi_discrete_equation.reconstruct(solutions);

        SET_TIME_POINT;
        while (true) {
            if (Solve_Controller::is_time_to_end(current_time)) {
                Tecplot::post_solution(solutions, "final");//post
                break;
            }                      

            if (Solve_Controller::is_time_to_post(current_time)) {
                //Tecplot::post_solution(solutions);//post
                Tecplot::post_condition_ = true;//post
            }
           
            SET_TIME_POINT;
            auto time_step = semi_discrete_equation.calculate_time_step<Time_Step_Method>(solutions);
            Solve_Controller::controll_time_step(current_time, time_step);

            Time_Integral_Method::update_solutions(semi_discrete_equation, solutions, time_step);
            current_time += time_step;

            Log::content_ << "computation cost: " << std::to_string(GET_TIME_DURATION) << "s \n";
            Log::print();
        }

        Log::content_ << "\n================================================================================\n";
        Log::content_ << "\t\t\t Total ellapsed time: " << GET_TIME_DURATION << "s\n";
        Log::content_ << "================================================================================\n\n";
        Log::print();
    }
};