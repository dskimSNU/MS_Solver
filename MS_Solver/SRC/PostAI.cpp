#include "../INC/PostAI.h"


void Post_AI_Data::conditionally_post(void) {
	if (This_::post_condition_)
		This_::post();
}

void Post_AI_Data::post(void) {

	static size_t num_post = 1;
	static size_t num_post_data = 1;

	auto file_name = "AI_Solver_Data_" + std::to_string(num_post) + ".txt";
	auto file_path = path_ + file_name;

	constexpr size_t num_solution_str = 3;
	for (const auto target_cell_index : target_cell_indexes_) {
		auto& data_text = ai_data_text_set_.at(target_cell_index);

		data_text.front() = "#########################" + std::to_string(num_post_data++);

		data_text.add_write(file_path);
		data_text.erase(data_text.end() - num_solution_str, data_text.end());

		if (10000 < num_post_data) {
			num_post_data = 1;

			file_name = "AI_Solver_Data_" + std::to_string(++num_post) + ".txt";
			file_path = path_ + file_name;
			comment_.add_write(file_path);
		}
	}
	target_cell_indexes_.clear();
}

void Post_AI_Data::post_scatter_data(const std::vector<double>& limiting_values) {

	const size_t num_cell = limiting_values.size();

	std::string limting_value_sentence;

	for (size_t i = 0; i < num_cell; i++)
		limting_value_sentence += ms::double_to_string(limiting_values[i]) + "\n";
	limting_value_sentence += "\n";

	Text limiting_value_text = { limting_value_sentence };

	const auto file_name = "limiting values.txt";
	const auto file_path = path_ + "scatter_data/" + file_name;
	limiting_value_text.add_write(file_path);
}

