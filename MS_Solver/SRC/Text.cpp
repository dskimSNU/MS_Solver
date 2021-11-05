#include "../INC/Text.h"

Text::Text(std::ifstream& file, const size_t num_read_line) {
	this->read(file, num_read_line);
}

Text& Text::operator<<(const std::string& str) {
	this->push_back(str);
	return *this;
}

Text& Text::operator<<(std::string&& str) {
	this->push_back(std::move(str));
	return *this;
}

void Text::read(const std::string& file_path) {	
	std::ifstream file_stream(file_path);
	dynamic_require(file_stream.is_open(), "Fail to open file" + file_path);

	std::string str;
	while (std::getline(file_stream, str))
		this->push_back(std::move(str));

	file_stream.close();
}

void Text::read(std::ifstream& file_stream, const size_t num_read_line) {
	dynamic_require(file_stream.is_open(), "Fail to open file");

	size_t index = 0;
	std::string str;
	while (std::getline(file_stream, str)) {
		this->push_back(std::move(str));
		if (++index == num_read_line)
			break;
	}
}

Text& Text::remove_empty_line(void) {
	this->erase(std::remove(this->begin(), this->end(), ""), this->end());
	return *this;
}

void Text::add_write(const std::string_view file_path) const {
	ms::make_path(file_path);
	std::ofstream output_file(file_path, std::ios::app);
	dynamic_require(output_file.is_open(), "output file stream should be opend before write");

	const auto num_sentence = this->size();
	for (auto i = this->begin(); i != this->end() - 1; ++i)
		output_file << *i << "\n";
	output_file << this->back();

	output_file.close();
}

void Text::write(const std::string_view file_path) const {
	ms::make_path(file_path);
	std::ofstream output_file(file_path);
	dynamic_require(output_file.is_open(), "output file stream should be opend before write"); 

	const auto num_sentence = this->size();
	for (auto i = this->begin(); i != this->end() - 1; ++i)
		output_file << *i << "\n";
	output_file << this->back();

	output_file.close();
}

void Text::merge(Text&& other) {
	this->insert(this->end(), std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
}

std::ostream& operator<<(std::ostream& ostream, const Text& text) {
	for (const auto& line : text)
		ostream << line << "\n";
	return ostream;
}


Binary_Writer::Binary_Writer(const std::string_view file_path) {
	ms::make_path(file_path);
	binary_file_stream_.open(file_path.data(), std::ios::binary);

	dynamic_require(this->binary_file_stream_.is_open(), "file should be opened");
}

Binary_Writer::Binary_Writer(const std::string_view file_path, std::ios_base::openmode mode) {
	ms::make_path(file_path);
	binary_file_stream_.open(file_path.data(), std::ios::binary | mode);

	dynamic_require(this->binary_file_stream_.is_open(), "file should be opened");
}

template <>
Binary_Writer& Binary_Writer::operator<<(const char* value) {
	this->binary_file_stream_.write(reinterpret_cast<const char*>(value), sizeof(value));
	return *this;
}

template <>
Binary_Writer& Binary_Writer::operator<<(const std::string& str) {
	this->binary_file_stream_ << str;
	return *this;
}

namespace ms {
	void be_replaced(std::string& str, const char target, const char replacement) {
		while (true) {
			const auto pos = str.find(target);

			if (pos == std::string::npos)
				break;

			str[pos] = replacement;
		}
	}

	void be_replaced(std::string& str, const std::string_view target, const std::string_view replacement) {
		if (target.empty())
			return;

		while (true) {
			const auto pos = str.find(target.data());

			if (pos == std::string::npos)
				break;

			str.replace(pos, target.size(), replacement.data());
		}
	}

	void be_removed(std::string& str, const std::string_view target) {
		ms::be_replaced(str, target, "");
	}

	std::vector<std::string> parse(const std::string& str, const char delimiter) {
		if (str.empty())
			return std::vector<std::string>();

		std::vector<std::string> parsed_string_set;
		for (auto iter1 = str.begin();;) {
			auto iter2 = std::find(iter1, str.end(), delimiter);

			if (iter1 != iter2)
				parsed_string_set.emplace_back(iter1, iter2);

			if (iter2 == str.end())
				return parsed_string_set;

			iter1 = iter2 + 1;
		}
	};

	std::vector<std::string> parse(const std::string& str, const std::vector<char>& delimiters) {
		if (delimiters.empty())
			return { str };

		const auto num_delimiter = delimiters.size();

		const auto reference_delimiter = delimiters[0];
		
		auto temp_str = str;
		for (size_t i = 1; i < num_delimiter; ++i)
			ms::be_replaced(temp_str, delimiters[i], reference_delimiter);

		return ms::parse(temp_str, reference_delimiter);
	}

	std::string replace(const std::string& str, const std::string_view target, const std::string_view replacement) {
		auto result = str;
		ms::be_replaced(result, target, replacement);
		return result;
	}

	std::string remove(const std::string& str, const std::string_view target) {
		auto result = str;
		ms::be_removed(result, target);
		return result;
	}

	size_t find_icase(const std::string& str, const std::string& target) {
		auto u_str = ms::upper_case(str);
		auto u_target = ms::upper_case(target);

		return u_str.find(u_target);
	}

	size_t rfind_nth(const std::string& object_str, const std::string& target_str, const size_t n) {
		if (n < 1)
			return std::string::npos;

		auto pos = std::string::npos;
		for (size_t i = 0; i < n; ++i) {
			if (pos == 0)
				return std::string::npos;

			pos = object_str.rfind(target_str, pos - 1); //rfind next pos
		}
		
		return pos;
	}

	std::string upper_case(const std::string& str) {
		auto result = str;
		std::transform(result.begin(), result.end(), result.begin(), toupper);
		//std::transform(result.begin(), result.end(), result.begin(), std::toupper); //filesystem이랑 있으면 충돌
		return result;
	}

	bool contains_icase(const std::string& str, const char* target) {
		return ms::find_icase(str, target) != std::string::npos;
	}

	std::string double_to_str_sp(const double value) {
		std::ostringstream os;
		os << std::setprecision(16) << std::showpoint << value;
		return os.str();
	}

	std::vector<std::string> file_paths_in_path(const std::string& path) {
		std::vector<std::string> file_name_text;

		std::filesystem::directory_iterator iter(path);
		while (iter != std::filesystem::end(iter)) {
			const auto& entry = *iter;

			if (entry.is_directory()) {
				iter++;
				continue;
			}

			file_name_text.push_back(entry.path().string());
			iter++;
		}

		return file_name_text;
	}

	void make_path(std::string_view file_path) {
		const auto file_name_size = file_path.size() - file_path.find_last_of("/") - 1;
		file_path.remove_suffix(file_name_size);

		if (file_path.empty())
			return;

		std::filesystem::path p(file_path);
		if (std::filesystem::exists(p))
			return;
		else
			std::filesystem::create_directories(p);
	}

	void rename(const std::string& path, const std::string& old_name, const std::string& new_name) {
		std::filesystem::rename(path + old_name, path + new_name);
	}
}


