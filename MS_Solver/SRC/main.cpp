#include "../INC/Initial_Condition.h"
#include "../INC/Discrete_Equation.h"
#include "../INC/Setting.h"

namespace ms {
	inline void apply_user_defined_setting(void);
}

using Grid_Element_Builder_		= Grid_Element_Builder<GRID_FILE_TYPE, __DIMENSION__>;
using Grid_						= Grid<__DIMENSION__>;
using Semi_Discrete_Equation_	= Semi_Discrete_Equation<GOVERNING_EQUATION, SPATIAL_DISCRETE_METHOD, RECONSTRUCTION_METHOD, NUMERICAL_FLUX_FUNCTION, SCAILING_METHOD_FLAG>;
using Discrete_Equation_		= Discrete_Equation<TIME_INTEGRAL_METHOD>;

int main(void) {
	ms::apply_user_defined_setting();

	const char delimiter = ',';
	std::string grid_file_names_str = GRID_FILE_NAMES;
	auto grid_file_names = ms::parse(grid_file_names_str, delimiter);

	for (auto& grid_file_name : grid_file_names) {
		ms::be_removed(grid_file_name, " ");

		Tecplot::initialize<GOVERNING_EQUATION>(__POST_ORDER__, POST_FILE_FORMAT); //post
		Solve_Controller::initialize(SOLVE_END_CONDITION, SOLVE_POST_CONDITION);

		const auto date_str = Log::date_string();

		Log::content_ << "current date : " << date_str << "\n\n";
		Log::content_ << "================================================================================\n";
		Log::content_ << "\t\t\t\t SETTING \n";
		Log::content_ << "================================================================================\n";
		Log::content_ << std::left << std::setw(35) << "Grid " << grid_file_name << "\n";
		Log::content_ << std::left << std::setw(35) << "Governing Equation" << GOVERNING_EQUATION::name() << "\n";
		Log::content_ << std::left << std::setw(35) << "Initial Condtion" << INITIAL_CONDITION::name() << "\n";
		Log::content_ << std::left << std::setw(35) << "Spatial Discrete Method" << SPATIAL_DISCRETE_METHOD::name() << "\n";

		if constexpr (SCAILING_METHOD_FLAG)
			Log::content_ << std::left << std::setw(35) << "Reconstruction Method" << RECONSTRUCTION_METHOD::name() << " with scailing method\n";
		else
			Log::content_ << std::left << std::setw(35) << "Reconstruction Method" << RECONSTRUCTION_METHOD::name() << "\n";

		Log::content_ << std::left << std::setw(35) << "Numeraical Flux Function" << NUMERICAL_FLUX_FUNCTION::name() << "\n";
		Log::content_ << std::left << std::setw(35) << "Time Integral Method" << TIME_INTEGRAL_METHOD::name() << "\n";
		Log::content_ << std::left << std::setw(35) << "Time Step Method" << TIME_STEP_METHOD::name() << "\n";
		Log::content_ << std::left << std::setw(35) << "Solve End Condtion" << Solve_Controller::end_condition_name() << "\n";
		Log::content_ << "================================================================================\n";
		Log::content_ << "================================================================================\n\n";
		Log::print();

		Log::set_path(__DEFAULT_PATH__ + grid_file_name + "_" + date_str + "/");
		Tecplot::set_path(__DEFAULT_PATH__ + grid_file_name + "_" + date_str + "/"); //post
		//Post_AI_Data::set_path(__DEFAULT_PATH__ + grid_file_name + "_" + date_str + "/Post_AI/"); //postAI

		auto grid_element = Grid_Element_Builder_::build_from_grid_file(grid_file_name);
		Grid_ grid(std::move(grid_element));

		Tecplot::post_grid(grid); //post
		//Post_AI_Data::intialize(grid); //postAI

		Semi_Discrete_Equation_ semi_discrete_equation(grid);
		auto solutions = semi_discrete_equation.calculate_initial_solutions<INITIAL_CONDITION>();

		try { Discrete_Equation_::solve<TIME_STEP_METHOD>(semi_discrete_equation, solutions); }
		catch (const std::exception& exception) {
			Log::content_ << "\n================================================================================\n";
			Log::content_ << "\t\t\t Abnormal Termination\n";
			Log::content_ << "================================================================================\n";
			Log::content_ << "Essential requirement is not satisfied => " << exception.what() << "\n";
			Log::print();
			Log::write();
			Tecplot::post_solution(solutions, "abnormal_termination");
			std::exit(523);
		}

		//semi_discrete_equation.estimate_error<INITIAL_CONDITION>(solutions, __SOLVE_END_CONDITION_CONSTANT__);

		Log::write();
	}
}


namespace ms {
	inline void apply_user_defined_setting(void) {
#if		__RECONSTRUCTION_METHOD__	== __ANN_RECONSTRUCTION__
		ANN_limiter<GRADIENT_METHOD>::set_model(TO_STRING(__ANN_MODEL__));
#endif
		if constexpr (__DIMENSION__ == 2) {
			Linear_Advection<2>::initialize({ X_ADVECTION_SPEED, Y_ADVECTION_SPEED });
			Sine_Wave<2>::initialize({ X_WAVE_LENGTH, Y_WAVE_LENGTH });

		#if __GOVERNING_EQUATION__ == __EULER__			
					Supersonic_Inlet1_Neighbor_Solution<GOVERNING_EQUATION::num_equation()>::initialize({ INFLOW_RHO1,INFLOW_RHOU1,INFLOW_RHOV1,INFLOW_RHOE1 });
					Supersonic_Inlet2_Neighbor_Solution<GOVERNING_EQUATION::num_equation()>::initialize({ INFLOW_RHO2,INFLOW_RHOU2,INFLOW_RHOV2,INFLOW_RHOE2 });
		#endif
		}
		else if constexpr (__DIMENSION__ == 3) {
			Linear_Advection<3>::initialize({ X_ADVECTION_SPEED, Y_ADVECTION_SPEED, Z_ADVECTION_SPEED });
			Sine_Wave<3>::initialize({ X_WAVE_LENGTH, Y_WAVE_LENGTH, Z_WAVE_LENGTH });
		}
		else
			throw std::runtime_error("not supported dimension");
	}
}