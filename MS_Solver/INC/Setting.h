#pragma once
//Setting
#define DIMENSION						2
#define GRID_FILE_TYPE					Gmsh
#define GRID_FILE_NAME					"Quad50"
#define GOVERNING_EQUATION_NAME			Linear_Advection
#define INITIAL_CONDITION_NAME			Square_Wave
#define SPATIAL_DISCRETE_METHOD			FVM
#define RECONSTRUCTION_ORDER			1 
#define RECONSTRUCTION_TYPE				MLP_u1
#define GRADIENT_METHOD					Vertex_Least_Square
#define NUMERICAL_FLUX_NAME				LLF
#define TIME_INTEGRAL_METHOD			SSPRK33
#define TIME_STEP_METHOD_NAME			CFL
#define TIME_STEP_CONSTANT				0.9 
#define END_CONDITION_NAME				Time
#define	END_CONDITION_CONSTANT			2.0
#define POST_ORDER						0
#define POST_CONDITION_NAME				Time
#define POST_CONDITION_CONSTANT			0.2
#define PATH							"E:/Code/Result/MS_Solver/" + GOVERNING_EQUATION::name() + "/" + INITIAL_CONDITION::name() + "/" + SPATIAL_DISCRETE_METHOD::name() + "_" + RECONSTRUCTION_METHOD::name()  + "/" + GRID_FILE_NAME + "/"


//Setting availiable list

//DIMENSION							2
//GRID_FILE_TYPE					Gmsh
//GRID_FILE_NAME					"-"
//GOVERNING_EQUATION_NAME			Linear_Advection, Burgers, Euler
//INITIAL_CONDITION_NAME			Sine_Wave, Square_Wave, Modified_SOD
//SPATIAL_DISCRETE_METHOD			FVM
//RECONSTRUCTION_ORDER				0,1 (For FVM)
//RECONSTRUCTION_TYPE				Linear_Reconstruction, MLP_u1, AI			# will be ignored when reconstruction order is 0
//GRADIENT_METHOD					Vertex_Least_Square, Face_Least_Square		# will be ignored when reconstruction order is 0
//NUMERICAL_FLUX_NAME				LLF
//TIME_INTGRAL_METHOD				SSPRK33
//TIME_STEP_METHOD_NAME				CFL, ConstDt
//TIME_STEP_CONSTNAT				-
//END_CONDITION_NAME				Time, Iter
//END_CONDITION_CONSTANT			-
//POST_CONDITION_NAME				Time
//POST_CONDITION_CONSTANT			-


// Mode 
	//#define POST_AI_DATA_MODE
	//#define ERROR_CALCULATION_MODE

// User Define Setting
	//Initial condtion
		//Sine Wave 2D
		#define X_WAVE_LENGTH 1
		#define Y_WAVE_LENGTH 1













//FORMAT SETTER
#define FORMAT1(x,y) x ## _ ## y ## D
#define SET_FORMAT1(x,y) FORMAT1(x,y) 
#define FORMAT2(x,y) x ## _ ## y 
#define SET_FORMAT2(x,y) FORMAT2(x,y)


//USING MACRO
//FORMAT1(X,Y) := X_YD
#define GOVERNING_EQUATION		SET_FORMAT1(GOVERNING_EQUATION_NAME	, DIMENSION)
#define INITIAL_CONDITION		SET_FORMAT1(INITIAL_CONDITION_NAME	, DIMENSION)

//FORAMT2(X,Y) := X_Y
#define SOLVE_END_CONDITION		SET_FORMAT2(End_By					, END_CONDITION_NAME)<END_CONDITION_CONSTANT>		
#define SOLVE_POST_CONDITION	SET_FORMAT2(Post_By					, POST_CONDITION_NAME)<POST_CONDITION_CONSTANT>		

#define NUMERICAL_FLUX			NUMERICAL_FLUX_NAME<GOVERNING_EQUATION>
#define TIME_STEP_METHOD		TIME_STEP_METHOD_NAME<TIME_STEP_CONSTANT>

#if	(RECONSTRUCTION_ORDER == 0)
	#define RECONSTRUCTION_METHOD	Constant_Reconstruction
#else
	#define RECONSTRUCTION_METHOD	RECONSTRUCTION_TYPE<GRADIENT_METHOD<GOVERNING_EQUATION::num_equation(), DIMENSION>>
#endif

