#include "../INC/Tecplot.h"

void Tecplot::record_cell_indexes(void) {
	const auto num_cell = This_::num_post_points_.size();
	std::vector<uint> cell_index(num_cell);
	for (uint i = 0; i < num_cell; ++i)
		cell_index[i] = i;

	This_::record_cell_variables("cell_index", cell_index);
}
void Tecplot::conditionally_record_cell_indexes(void) {
	if (This_::post_condition_)
		This_::record_cell_indexes();
}

void Tecplot::write_binary_header(const Post_File_Type file_type, const std::string_view post_file_path) {
	static int strand_id = 0;

	Binary_Writer post_file(post_file_path);

	//I. HEADER SECTION

	//i
	post_file << "#!TDV112";	//version	

	//ii
	post_file << 1;				//byte order, default

	//iii. Title and variable names
	if (file_type == Post_File_Type::grid) {
		post_file << 1;												//file type, grid = 1 
		post_file << This_::convert_to_binary_data("Grid");	//title

		const char delimiter = ',';
		const auto parsed_grid_variable_strs = ms::parse(This_::grid_variables_str_, delimiter);

		post_file << static_cast<int>(parsed_grid_variable_strs.size());	//num variable
		for (const auto& grid_variable_str : parsed_grid_variable_strs)
			post_file << This_::convert_to_binary_data(grid_variable_str);	//variable names
	}
	else {
		post_file << 2;																					//file type, solution = 2 
		post_file << This_::convert_to_binary_data("Solution_at_" + std::to_string(*This_::time_ptr_));	//title

		const auto parsed_solution_variable_strs = ms::parse(This_::solution_variables_str_, ',');

		const auto num_solution_data = static_cast<int>(parsed_solution_variable_strs.size());
		const auto num_additional_data = static_cast<int>(This_::additioinal_data_name_to_values_.size());

		post_file << num_solution_data + num_additional_data;						//num variable
		for (const auto& solution_variable_str : parsed_solution_variable_strs)
			post_file << This_::convert_to_binary_data(solution_variable_str);		//variable names
		for (const auto& [name, data] : This_::additioinal_data_name_to_values_)
			post_file << This_::convert_to_binary_data(name);						//variable names
	}

	//iiii. zones
	post_file << 299.0f;	//zone marker

	if (file_type == Post_File_Type::grid)
		post_file << convert_to_binary_data("Grid");												//zone name
	else {
		post_file << convert_to_binary_data("Solution_at_" + std::to_string(*This_::time_ptr_));	//zone name
		strand_id++;
	}

	post_file << -1;									//parent zone, default
	post_file << strand_id;								//strand id

	if (file_type == Post_File_Type::grid)
		post_file << 0.0;								//solution time
	else
		post_file << *This_::time_ptr_;					//solution time	
		//post_file << strand_id * 1.0E-3;				//debug

	post_file << -1;									//not used
	post_file << static_cast<int>(This_::zone_type_);	//zone type
	//post_file << 0;									//data packing block = 0 .. ?
	post_file << 0 << 0 << 0;							//specify var location, one to one face neighbor, user define face neighbor connection, default
	post_file << static_cast<int>(This_::num_node_);	//num points
	post_file << static_cast<int>(This_::num_element_);	//num elements
	post_file << 0 << 0 << 0 << 0;						//i,j,k cell dim, auxilarily name index pair, default
	post_file << 357.0f;								//EOH_marker
}
void Tecplot::write_binary_grid_post_file(const std::vector<std::vector<double>>& post_coordinate_blocks, const std::vector<std::vector<int>>& connectivities) {
	const auto grid_file_path = This_::path_ + "grid.plt";
	This_::write_binary_header(Post_File_Type::grid, grid_file_path);

	//II. DATA SECTION		
	const auto num_variable = post_coordinate_blocks.size();

	Binary_Writer grid_binary_file(grid_file_path, std::ios::app);

	//zone
	grid_binary_file << 299.0f;			//zone marker

	for (ushort i = 0; i < num_variable; ++i)
		grid_binary_file << 2;			//variable data format, double = 2

	grid_binary_file << 0 << 0 << -1;	//has passive variable, has variable sharing, zone number to share connectivity, default

	for (ushort i = 0; i < num_variable; ++i) {
		const auto min_value = *std::min_element(post_coordinate_blocks[i].begin(), post_coordinate_blocks[i].end());
		const auto max_value = *std::max_element(post_coordinate_blocks[i].begin(), post_coordinate_blocks[i].end());

		grid_binary_file << min_value << max_value;	//min,max value of each variable
	}

	for (ushort i = 0; i < num_variable; ++i)
		grid_binary_file << post_coordinate_blocks[i];			//values of each variable

	for (const auto& connectivity : connectivities)
		grid_binary_file << connectivity;			//connectivity
}
void Tecplot::write_binary_solution_post_file(const std::vector<std::vector<double>>& post_solution_binary_datas, const std::string& comment) {
	static size_t count = 1;

	std::string solution_file_path;
	if (comment.empty())
		solution_file_path = This_::path_ + "solution_" + std::to_string(count++) + ".plt";
	else
		solution_file_path = This_::path_ + "solution_" + std::to_string(count++) + "_" + comment + ".plt";

	This_::write_binary_header(Post_File_Type::solution, solution_file_path);

	//II. DATA SECTION
	Binary_Writer solution_binary_file(solution_file_path, std::ios::app);

	const auto num_data = post_solution_binary_datas.size();

	//zone
	solution_binary_file << 299.0f;			//zone marker

	for (ushort i = 0; i < num_data; ++i)
		solution_binary_file << 2;			//variable data format, double = 2

	solution_binary_file << 0 << 0 << -1;	//has passive variable, has variable sharing, zone number to share connectivity, default

	for (ushort i = 0; i < num_data; ++i) {
		const auto min_value = *std::min_element(post_solution_binary_datas[i].begin(), post_solution_binary_datas[i].end());
		const auto max_value = *std::max_element(post_solution_binary_datas[i].begin(), post_solution_binary_datas[i].end());

		solution_binary_file << min_value << max_value;	//min,max value of each variable
	}

	for (ushort i = 0; i < num_data; ++i)
		solution_binary_file << post_solution_binary_datas[i];	//values of each variable

	//finalize
	This_::additioinal_data_name_to_values_.clear();
	This_::post_condition_ = false;

	if (comment == "final") {
		count = 1;
		This_::reset();
	}
}

void Tecplot::write_ASCII_header(const Post_File_Type file_type, const std::string_view post_file_path) {
	static size_t strand_id = 0;

	Text header;
	header.reserve(10);
	if (file_type == Post_File_Type::grid) {
		header << "Title = Grid";
		header << "FileType = Grid";
		header << "Variables = " + This_::grid_variables_str_;
		header << "Zone T = Grid";
	}
	else {
		std::string solution_variable_str = "Variables = " + This_::solution_variables_str_;
		if (!This_::additioinal_data_name_to_values_.empty()) {
			for (const auto& [info_name, info_values] : This_::additioinal_data_name_to_values_)
				solution_variable_str += ", " + info_name;
		}

		header << "Title = Solution_at_" + ms::double_to_string(*time_ptr_);
		header << "FileType = Solution";
		header << solution_variable_str;
		header << "Zone T = Solution_at_" + ms::double_to_string(*time_ptr_);

		strand_id++;
	}

	header << "ZoneType =  " + ms::to_string(This_::zone_type_);
	header << "Nodes = " + std::to_string(This_::num_node_);
	header << "Elements = " + std::to_string(This_::num_element_);
	header << "DataPacking = Block";
	header << "StrandID = " + std::to_string(strand_id);

	if (file_type == Post_File_Type::grid)
		header << "SolutionTime = 0.0 \n\n";
	else
		header << "SolutionTime = " + ms::double_to_string(*time_ptr_) + "\n\n";

	header.write(post_file_path);
}
void Tecplot::write_ASCII_grid_post_file(const std::vector<std::vector<double>>& post_coordinate_blocks, const std::vector<std::vector<int>>& connectivities) {
	ushort str_per_line = 0;

	const auto space_dimension = post_coordinate_blocks.size();
	const auto num_line = space_dimension + connectivities.size();
	Text grid_post_data_text;
	grid_post_data_text.reserve(num_line);

	std::string coordinate_str;
	for (ushort i = 0; i < space_dimension; ++i) {		
		for (const auto coordinate : post_coordinate_blocks[i]) {
			str_per_line++;
			coordinate_str += ms::double_to_str_sp(coordinate) + " ";
			if (str_per_line == 10) {
				str_per_line = 0;
				coordinate_str += "\n";
			}
		};
		grid_post_data_text << std::move(coordinate_str);
	}

	std::string connectivity_str;
	for (const auto& connectivity : connectivities) {
		for (const auto index : connectivity)
			connectivity_str += std::to_string(index + 1) + " "; // ASCII format connectivity start with 1
		grid_post_data_text << std::move(connectivity_str);
	}

	const auto grid_file_path = This_::path_ + "grid.plt";

	This_::write_ASCII_header(Post_File_Type::grid, grid_file_path);
	grid_post_data_text.add_write(grid_file_path);
}
void Tecplot::write_ASCII_solution_post_file(const std::vector<std::vector<double>>& set_of_post_solution_datas, const std::string& comment) {
	//convert data to ASCII text
	size_t str_per_line = 1;

	const auto num_data = set_of_post_solution_datas.size();
	Text ASCII_text(num_data);

	for (ushort i = 0; i < num_data; ++i) {
		const auto datas = set_of_post_solution_datas[i];
		for (size_t j = 0; j < This_::num_node_; ++j, ++str_per_line) {
			ASCII_text[i] += ms::double_to_string(datas[j]) + " ";
			if (str_per_line == 10) {
				ASCII_text[i] += "\n";
				str_per_line = 0;
			}
		}
		ASCII_text[i] += "\n\n";
	}

	//set file name
	static size_t count = 1;

	std::string solution_file_path;
	if (comment.empty())
		solution_file_path = This_::path_ + "solution_" + std::to_string(count++) + ".plt";
	else
		solution_file_path = This_::path_ + "solution_" + std::to_string(count++) + "_" + comment + ".plt";

	//write ASCII file
	This_::write_ASCII_header(Post_File_Type::solution, solution_file_path);
	ASCII_text.add_write(solution_file_path);


	//finalize
	This_::additioinal_data_name_to_values_.clear();
	This_::post_condition_ = false;
	if (comment == "final") {
		count = 1;
		This_::reset();
	}
}


void Tecplot::reset(void) {
	This_::post_condition_ = false;
	This_::path_.clear();
	This_::post_order_ = 0;
	This_::grid_variables_str_.clear();
	This_::solution_variables_str_.clear();
	This_::num_element_ = 0;
	This_::num_node_ = 0;
	This_::time_ptr_ = nullptr;
	This_::num_post_points_.clear();
	This_::set_of_basis_post_points_.clear();
}
std::vector<int> Tecplot::convert_to_binary_data(const std::string& str) {
	std::vector<int> tecplot_binary_format;
	tecplot_binary_format.insert(tecplot_binary_format.end(), str.begin(), str.end());
	tecplot_binary_format.push_back(0); // null
	return tecplot_binary_format;
}

namespace ms {
	std::string to_string(const Zone_Type zone_type) {
		switch (zone_type) {
			case Zone_Type::FETriangle: return "FETriangle";
			case Zone_Type::FETetrahedron: return "FETetrahedron";
			default:
				throw std::runtime_error("not supproted zone type");
				return NULL;
		}
	}
}