#pragma once
#include "Boundary_Flux_Function.h"
#include "Grid_Element_Builder.h"

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <optional>


template <ushort space_dimension>
struct Ghost_Cell
{
	Euclidean_Vector<space_dimension> center_node;
	ElementType related_type;
	uint solution_related_cell_index;
	uint face_share_cell_index;

	template <ushort num_equation>
	Euclidean_Vector<num_equation> solution(const Euclidean_Vector<num_equation>& related_cell_solution) const;
};


template <ushort space_dimension>
class Grid
{
private:
	Grid_Elements<space_dimension> elements;
	std::unordered_map<uint, std::set<uint>> vnode_index_to_share_cell_index_set_ignore_pbdry_;
	std::unordered_map<uint, std::set<uint>> vnode_index_to_share_cell_index_set_consider_pbdry_;

public:
	Grid(Grid_Elements<space_dimension>&& grid_elements);

public:
	const Grid_Elements<space_dimension>& get_elements(void) const { return this->elements; };
	const std::unordered_map<uint, std::set<uint>>& get_vnode_index_to_share_cell_index_set_ignore_pbdry(void) const { return this->vnode_index_to_share_cell_index_set_ignore_pbdry_; };
	const std::unordered_map<uint, std::set<uint>>& get_vnode_index_to_share_cell_index_set_consider_pbdry(void) const { return this->vnode_index_to_share_cell_index_set_consider_pbdry_; };

public:
	std::vector<uint> boundary_owner_cell_indexes(void) const;
	std::vector<double> boundary_volumes(void) const;
	std::vector<ElementType> boundary_types(void) const;
	std::vector<Euclidean_Vector<space_dimension>> boundary_normals_at_centers(const std::vector<uint>& oc_indexes) const;
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> boundary_set_of_normals(const std::vector<uint>& oc_indexes, const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_nodes) const;
	std::vector<Euclidean_Vector<space_dimension>> boundary_center_nodes(void) const;
	std::vector<Euclidean_Vector<space_dimension>> boundary_owner_cell_to_faces(const std::vector<uint>& oc_indexes) const;
	std::vector<Quadrature_Rule<space_dimension>> boundary_quadrature_rules(const ushort polynomial_degree) const;
	
	template <typename Numerical_Flux_Function>
	auto boundary_flux_functions(void) const;

public:
	std::vector<double> cell_volumes(void) const;
	std::vector<std::array<double, space_dimension>> cell_projected_volumes(void) const;
	std::vector<Quadrature_Rule<space_dimension>> cell_quadrature_rules(const ushort polynomial_degree) const;
	std::vector<Euclidean_Vector<space_dimension>> cell_center_nodes(void) const;
	std::vector<std::vector<uint>> cell_set_of_vnode_indexes(void) const;
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> cell_set_of_vnodes(void) const;
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> cell_set_of_post_nodes(const ushort post_order) const;
	std::vector<std::vector<double>> cell_post_coordinate_blocks(const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_post_nodes) const;
	std::vector<std::vector<int>> cell_set_of_connectivities(const ushort post_order, const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_post_nodes) const;	
	std::vector<Dynamic_Matrix> cell_center_to_vertex_matrixes(void) const;
	std::vector<bool> cell_simplex_flags(void) const;
	std::vector<std::vector<Geometry<space_dimension>>> cell_set_of_sub_simplex_geometries(const std::vector<bool>& simplex_flags) const;

	template <ushort polynomial_degree>
	auto cell_basis_vector_functions(void) const;

public:
	std::vector<std::pair<uint, uint>> inner_face_oc_nc_index_pairs(void) const;
	std::vector<double> inner_face_volumes(void) const;
	std::vector<Euclidean_Vector<space_dimension>> inner_face_normals_at_center(const std::vector<std::pair<uint, uint>>& oc_nc_index_pairs) const;
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> inner_face_set_of_normals(const std::vector<uint>& oc_indexes, const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_nodes) const;
	std::vector<std::pair<Euclidean_Vector<space_dimension>, Euclidean_Vector<space_dimension>>> inner_face_oc_nc_to_face_pairs(const std::vector<std::pair<uint, uint>>& oc_nc_index_pairs) const;
	std::vector<Quadrature_Rule<space_dimension>> inner_face_quadrature_rules(const ushort polynomial_degree) const;

public:
	std::vector<std::pair<uint, uint>> periodic_boundary_oc_nc_index_pairs(void) const;
	std::vector<double> periodic_boundary_volumes(void) const;
	std::vector<Euclidean_Vector<space_dimension>> periodic_boundary_normals_at_center(const std::vector<std::pair<uint, uint>>& oc_nc_index_pairs) const;
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> periodic_boundary_set_of_normals(const std::vector<uint>& oc_indexes, const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_oc_side_nodes) const;
	std::vector<std::pair<Euclidean_Vector<space_dimension>, Euclidean_Vector<space_dimension>>> periodic_boundary_oc_nc_to_oc_nc_side_face_pairs(const std::vector<std::pair<uint, uint>>& oc_nc_index_pairs) const;
	std::vector<std::pair<Quadrature_Rule<space_dimension>, Quadrature_Rule<space_dimension>>> periodic_boundary_quadrature_rule_pairs(const ushort polynomial_degree) const;

public:
	std::unordered_map<uint, std::set<uint>> pbdry_vnode_index_to_matched_pbdry_vnode_index_set(void) const;
	std::vector<Ghost_Cell<space_dimension>> make_ghost_cells(void) const;
	std::vector<std::vector<uint>> ANN_indexes(void) const; //temporary	
	std::vector<std::vector<uint>> set_of_face_share_cell_indexes_ignore_pbdry(void) const;
	std::vector<std::vector<size_t>> set_of_face_share_cell_indexes_consider_pbdry(void) const;
	std::vector<std::vector<size_t>> set_of_vertex_share_cell_indexes_consider_pbdry(void) const;
	std::vector<std::vector<uint>> set_of_face_share_ghost_cell_indexes(const std::vector<Ghost_Cell<space_dimension>>& ghost_cells) const;

private:
	std::vector<uint> find_cell_indexes_have_these_vnodes_ignore_pbdry(const std::vector<uint>& vnode_indexes) const;
	std::vector<uint> find_cell_indexes_have_these_vnodes_consider_pbdry(const std::vector<uint>& vnode_indexes) const;
	std::optional<uint> find_face_share_cell_index_ignore_pbdry(const uint my_index, const std::vector<uint>& my_face_vnode_indexes) const;
	std::optional<uint> find_face_share_cell_index_consider_pbdry(const uint my_index, const std::vector<uint>& my_face_vnode_indexes) const;
	std::vector<std::vector<uint>> set_of_periodic_boundary_vnode_indexes(void) const;
	std::vector<std::vector<uint>> set_of_boundary_vnode_indexes(void) const;
};

namespace ms {
	template <typename T>
	void sort(std::vector<T>& vec) {
		std::sort(vec.begin(), vec.end());
	}

	template <typename T>
	std::vector<T>::const_iterator find(const std::vector<T>& vec, const T& val) {
		return std::find(vec.cbegin(), vec.cend(), val);
	}

	template <typename Container1, typename Container2, std::enable_if_t<std::is_same_v<typename Container1::value_type, typename Container2::value_type>, bool> = true>
	std::vector<typename Container1::value_type> set_intersection(const Container1& container1, const Container2& container2) {
		std::vector<typename Container1::value_type> intersection;
		std::set_intersection(container1.begin(), container1.end(), container2.begin(), container2.end(), std::back_inserter(intersection));

		return intersection;
	}

	template <typename Container1, typename Container2, std::enable_if_t<std::is_same_v<typename Container1::value_type, typename Container2::value_type>, bool> = true>
	std::vector<typename Container1::value_type> set_difference(const Container1& container1, const Container2& container2) {
		std::vector<typename Container1::value_type> difference;
		std::set_difference(container1.begin(), container1.end(), container2.begin(), container2.end(), std::back_inserter(difference));

		return difference;
	}
}

//Template Definition
template <ushort space_dimension>
template <ushort num_equation>
Euclidean_Vector<num_equation> Ghost_Cell<space_dimension>::solution(const Euclidean_Vector<num_equation>& related_cell_solution) const {
	switch (this->related_type)
	{
	case ElementType::periodic:
	case ElementType::supersonic_outlet: return related_cell_solution;
	case ElementType::supersonic_inlet1: return Supersonic_Inlet1_Neighbor_Solution<num_equation>::calculate();
	case ElementType::supersonic_inlet2: return Supersonic_Inlet2_Neighbor_Solution<num_equation>::calculate();
	default:
		throw std::runtime_error("not supproted element type");
		return {};
	}
}

template <ushort space_dimension>
Grid<space_dimension>::Grid(Grid_Elements<space_dimension>&& grid_elements) : elements(std::move(grid_elements)) {
	SET_TIME_POINT;
	
	//calculate vnode_index_to_share_cell_index_set_ignore_pbdry_
	const auto num_cell = this->elements.cell_elements.size();

	for (uint i = 0; i < num_cell; ++i) {
		const auto vnode_indexes = this->elements.cell_elements[i].vertex_node_indexes();
		for (const auto vnode_index : vnode_indexes) {
			if (!this->vnode_index_to_share_cell_index_set_ignore_pbdry_.contains(vnode_index))
				this->vnode_index_to_share_cell_index_set_ignore_pbdry_.emplace(vnode_index, std::set<uint>());

			this->vnode_index_to_share_cell_index_set_ignore_pbdry_.at(vnode_index).insert(i);
		}
	}

	const auto pbdry_vnode_index_to_matched_vnode_index_set = this->pbdry_vnode_index_to_matched_pbdry_vnode_index_set();

	//calculate vnode_index_to_share_cell_index_set_consider_pbdry_
	this->vnode_index_to_share_cell_index_set_consider_pbdry_ = this->vnode_index_to_share_cell_index_set_ignore_pbdry_;

	for (const auto& [vnode_index, matched_vnode_index_set] : pbdry_vnode_index_to_matched_vnode_index_set) {
		for (const auto matched_vnode_index : matched_vnode_index_set) {
			auto& i_set = this->vnode_index_to_share_cell_index_set_consider_pbdry_.at(vnode_index);
			const auto& j_set = this->vnode_index_to_share_cell_index_set_consider_pbdry_.at(matched_vnode_index);
			const auto difference = ms::set_difference(j_set, i_set);

			i_set.insert(difference.begin(), difference.end());
		}
	}

	Log::content_ << std::left << std::setw(50) << "@ Make grid " << " ----------- " << GET_TIME_DURATION << "s\n\n";
	Log::print();
}

template <ushort space_dimension>
std::vector<uint> Grid<space_dimension>::boundary_owner_cell_indexes(void) const {			
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_boundary = boundary_elements.size();
	std::vector<uint> oc_indexes(num_boundary);

	for (uint i = 0; i < num_boundary; ++i) {
		const auto& boundary_element = boundary_elements[i];

		const auto vnode_indexes = boundary_element.vertex_node_indexes();
		const auto indexes = this->find_cell_indexes_have_these_vnodes_ignore_pbdry(vnode_indexes);
		dynamic_require(indexes.size() == 1, "boundary should have unique owner cell");

		oc_indexes[i] = indexes.front();
	}

	return oc_indexes;
}

template <ushort space_dimension>
std::vector<double> Grid<space_dimension>::boundary_volumes(void) const {
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_boundary = boundary_elements.size();
	std::vector<double> volumes(num_boundary);

	for (uint i = 0; i < num_boundary; ++i) {
		const auto& boundary_element = boundary_elements[i];
		volumes[i] = boundary_element.geometry_.volume();
	}

	return volumes;
}

template <ushort space_dimension>
std::vector<ElementType> Grid<space_dimension>::boundary_types(void) const {
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_boundary = boundary_elements.size();
	std::vector<ElementType> types(num_boundary);

	for (uint i = 0; i < num_boundary; ++i) {
		const auto& boundary_element = boundary_elements[i];
		types[i] = boundary_element.type();
	}

	return types;
}

template <ushort space_dimension>
std::vector<Euclidean_Vector<space_dimension>> Grid<space_dimension>::boundary_normals_at_centers(const std::vector<uint>& oc_indexes) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_boundary = boundary_elements.size();
	std::vector<Euclidean_Vector<space_dimension>> normals(num_boundary);

	for (uint i = 0; i < num_boundary; ++i) {
		const auto& oc_element = cell_elements[oc_indexes[i]];
		const auto& boundary_element = boundary_elements[i];
		const auto  center = boundary_element.geometry_.center_node();

		normals[i] = boundary_element.normalized_normal_vector(oc_element, center);
	}

	return normals;
}

template <ushort space_dimension>
std::vector<std::vector<Euclidean_Vector<space_dimension>>> Grid<space_dimension>::boundary_set_of_normals(const std::vector<uint>& oc_indexes, const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_nodes) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_boundary = boundary_elements.size();
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> boundary_set_of_normals(num_boundary);

	for (uint i = 0; i < num_boundary; ++i) {
		const auto& oc_element = cell_elements[oc_indexes[i]];
		const auto& boundary_element = boundary_elements[i];

		boundary_set_of_normals[i] = boundary_element.normalized_normal_vectors(oc_element, set_of_nodes[i]);
	}

	return boundary_set_of_normals;
}


template <ushort space_dimension>
std::vector<Euclidean_Vector<space_dimension>> Grid<space_dimension>::boundary_center_nodes(void) const {
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_boundary = boundary_elements.size();
	std::vector<Euclidean_Vector<space_dimension>> center_nodes(num_boundary);

	for (uint i = 0; i < num_boundary; ++i) {
		const auto& boundary_element = boundary_elements[i];
		center_nodes[i] = boundary_element.geometry_.center_node();
	}

	return center_nodes;
}

template <ushort space_dimension>
std::vector<Euclidean_Vector<space_dimension>> Grid<space_dimension>::boundary_owner_cell_to_faces(const std::vector<uint>& oc_indexes) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_boundary = boundary_elements.size();
	std::vector<Euclidean_Vector<space_dimension>> center_to_center_vectors(num_boundary);

	for (size_t i = 0; i < num_boundary; ++i) {
		const auto& oc_geometry = cell_elements[oc_indexes[i]].geometry_;
		const auto& boundary_geometry = boundary_elements[i].geometry_;

		const auto oc_center = oc_geometry.center_node();
		const auto boundary_center = boundary_geometry.center_node();

		center_to_center_vectors[i] = boundary_center - oc_center;		
	}

	return center_to_center_vectors;
}

template <ushort space_dimension>
std::vector<Quadrature_Rule<space_dimension>> Grid<space_dimension>::boundary_quadrature_rules(const ushort polynomial_degree) const {
	const auto& boundary_elements = this->elements.boundary_elements;
	
	const auto num_boundary = boundary_elements.size();
	std::vector<Quadrature_Rule<space_dimension>> quadrature_rules(num_boundary);

	for (size_t i = 0; i < num_boundary; ++i) {
		const auto& boundary_geometry = boundary_elements[i].geometry_;
		quadrature_rules[i] = boundary_geometry.get_quadrature_rule(polynomial_degree);
	}

	return quadrature_rules;
}



template <ushort space_dimension>
template <typename Numerical_Flux_Function>
auto Grid<space_dimension>::boundary_flux_functions(void) const {
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_boundary = boundary_elements.size();
	std::vector<std::unique_ptr<Boundary_Flux_Function<Numerical_Flux_Function>>> boundary_flux_functions;
	boundary_flux_functions.reserve(num_boundary);

	for (uint i = 0; i < num_boundary; ++i) {
		const auto& boundary_element = boundary_elements[i];
		boundary_flux_functions.push_back(Boundary_Flux_Function_Factory<Numerical_Flux_Function>::make(boundary_element.type()));
	}

	return boundary_flux_functions;
}


template <ushort space_dimension>
std::vector<double> Grid<space_dimension>::cell_volumes(void) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	std::vector<double> volumes(num_cell);

	for (uint i = 0; i < num_cell; ++i) 
		volumes[i] = cell_elements[i].geometry_.volume();

	return volumes;
}

template <ushort space_dimension>
std::vector<std::array<double, space_dimension>> Grid<space_dimension>::cell_projected_volumes(void) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	std::vector<std::array<double, space_dimension>> projected_volumes(num_cell);

	for (uint i = 0; i < num_cell; ++i)
		projected_volumes[i] = cell_elements[i].geometry_.projected_volume();

	return projected_volumes;
}

template <ushort space_dimension>
std::vector<Quadrature_Rule<space_dimension>> Grid<space_dimension>::cell_quadrature_rules(const ushort polynomial_degree) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	std::vector<Quadrature_Rule<space_dimension>> quadrature_rules(num_cell);

	for (uint i = 0; i < num_cell; ++i) 
		quadrature_rules[i] = cell_elements[i].geometry_.get_quadrature_rule(polynomial_degree);
	
	return quadrature_rules;
}


template <ushort space_dimension>
std::vector<Euclidean_Vector<space_dimension>> Grid<space_dimension>::cell_center_nodes(void) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	std::vector<Euclidean_Vector<space_dimension>> center_nodes(num_cell);

	for (uint i = 0; i < num_cell; ++i)
		center_nodes[i] = cell_elements[i].geometry_.center_node();

	return center_nodes;
}

template <ushort space_dimension>
std::vector<std::vector<uint>> Grid<space_dimension>::cell_set_of_vnode_indexes(void) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	std::vector<std::vector<uint>> set_of_vnode_indexes(num_cell);

	for (uint i = 0; i < num_cell; ++i)
		set_of_vnode_indexes[i] = cell_elements[i].vertex_node_indexes();

	return set_of_vnode_indexes;
}

template <ushort space_dimension>
std::vector<std::vector<Euclidean_Vector<space_dimension>>> Grid<space_dimension>::cell_set_of_vnodes(void) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> set_of_vnodes(num_cell);

	for (uint i = 0; i < num_cell; ++i)
		set_of_vnodes[i] = cell_elements[i].geometry_.vertex_nodes();

	return set_of_vnodes;
}

template <ushort space_dimension>
std::vector<std::vector<Euclidean_Vector<space_dimension>>> Grid<space_dimension>::cell_set_of_post_nodes(const ushort post_order) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> set_of_post_nodes(num_cell);

	for (uint i = 0; i < num_cell; ++i)
		set_of_post_nodes[i] = cell_elements[i].geometry_.post_nodes(post_order);

	return set_of_post_nodes;
}

template <ushort space_dimension>
std::vector<std::vector<double>> Grid<space_dimension>::cell_post_coordinate_blocks(const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_post_nodes) const {

	std::vector<std::vector<double>> coordinates(space_dimension);

	for (const auto& post_nodes : set_of_post_nodes) {
		for (const auto& node : post_nodes) {
			for (ushort j = 0; j < space_dimension; ++j)
				coordinates[j].push_back(node[j]);
		}
	}

	return coordinates;
}

template <ushort space_dimension>
std::vector<std::vector<int>> Grid<space_dimension>::cell_set_of_connectivities(const ushort post_order, const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_post_nodes) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	std::vector<std::vector<int>> set_of_connectivities;

	size_t connectivity_start_index = 0;

	for (uint i = 0; i < num_cell; ++i) {
		const auto& post_nodes = set_of_post_nodes[i];
		const auto post_connectivities = cell_elements[i].geometry_.reference_geometry_.post_connectivities(post_order, connectivity_start_index);

		for (const auto& connectivity : post_connectivities) {
			const auto num_point = connectivity.size();
			std::vector<int> temp(num_point);

			for (uint j = 0; j < num_point; ++j)
				temp[j] = static_cast<int>(connectivity[j]);

			set_of_connectivities.push_back(std::move(temp));
		}

		const auto num_post_node = post_nodes.size();
		connectivity_start_index += num_post_node;
	}

	return set_of_connectivities;
}

template <ushort space_dimension>
std::vector<Dynamic_Matrix> Grid<space_dimension>::cell_center_to_vertex_matrixes(void) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto num_cell = cell_elements.size();

	std::vector<Dynamic_Matrix> center_to_vertex_matrixes;
	center_to_vertex_matrixes.reserve(num_cell);

	for (size_t i = 0; i < num_cell; ++i) {
		const auto& geometry = cell_elements[i].geometry_;

		const auto center_node = geometry.center_node();
		const auto vnodes = geometry.vertex_nodes();
		const auto num_vertex = vnodes.size();

		Dynamic_Matrix center_to_vertex_matrix(space_dimension, num_vertex);
		for (size_t i = 0; i < num_vertex; ++i) {
			const auto center_to_vertex = vnodes[i] - center_node;
			center_to_vertex_matrix.change_column(i, center_to_vertex);
		}

		center_to_vertex_matrixes.push_back(std::move(center_to_vertex_matrix));
	}

	return center_to_vertex_matrixes;
}


template <ushort space_dimension>
std::vector<bool> Grid<space_dimension>::cell_simplex_flags(void) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto num_cell = cell_elements.size();

	std::vector<bool> simplex_flags(num_cell, false);

	for (size_t i = 0; i < num_cell; ++i) 
		simplex_flags[i] = cell_elements[i].geometry_.reference_geometry_.is_simplex();
	
	return simplex_flags;
}


template <ushort space_dimension>
std::vector<std::vector<Geometry<space_dimension>>> Grid<space_dimension>::cell_set_of_sub_simplex_geometries(const std::vector<bool>& simplex_flags) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto num_cell = cell_elements.size();

	std::vector<std::vector<Geometry<space_dimension>>> set_of_sub_simplex_geometries(num_cell);

	for (size_t i = 0; i < num_cell; ++i) {
		if (!simplex_flags[i])
			set_of_sub_simplex_geometries[i] = cell_elements[i].geometry_.sub_simplex_geometries();
	}

	return set_of_sub_simplex_geometries;
}


template <ushort space_dimension>
template <ushort polynomial_degree>
auto  Grid<space_dimension>::cell_basis_vector_functions(void) const {
	const auto& cell_elements = this->elements.cell_elements;

	const auto num_cell = cell_elements.size();
	constexpr auto num_basis_ = ms::combination_with_repetition(1 + space_dimension, polynomial_degree);
	std::vector<Vector_Function<Polynomial<space_dimension>, num_basis_>> basis_vector_functions(num_cell);

	for (uint i = 0; i < num_cell; ++i) {
		const auto& cell_geometry = cell_elements[i].geometry_;
		basis_vector_functions[i] = cell_geometry.orthonormal_basis_vector_function<polynomial_degree>();
	}

	return basis_vector_functions;
}


template <ushort space_dimension>
std::vector<std::pair<uint, uint>> Grid<space_dimension>::inner_face_oc_nc_index_pairs(void) const {
	const auto& inner_face_elements = this->elements.inner_face_elements;

	const auto num_inner_face = inner_face_elements.size();
	std::vector<std::pair<uint, uint>> inner_face_oc_nc_index_pairs(num_inner_face);

	for (uint i = 0; i < num_inner_face; ++i) {
		const auto& inner_face_element = inner_face_elements[i];

		const auto cell_indexes = this->find_cell_indexes_have_these_vnodes_ignore_pbdry(inner_face_element.vertex_node_indexes());
		dynamic_require(cell_indexes.size() == 2, "inner face should have an unique owner neighbor cell pair");

		//set first index as oc index
		const auto oc_index = cell_indexes[0];
		const auto nc_index = cell_indexes[1];
		inner_face_oc_nc_index_pairs[i] = { oc_index,nc_index };
	}

	return inner_face_oc_nc_index_pairs;
}

template <ushort space_dimension>
std::vector<double> Grid<space_dimension>::inner_face_volumes(void) const {
	const auto& inner_face_elements = this->elements.inner_face_elements;

	const auto num_inner_face = inner_face_elements.size();
	std::vector<double> volumes(num_inner_face);

	for (uint i = 0; i < num_inner_face; ++i) {
		const auto& inner_face_element = inner_face_elements[i];
		volumes[i] = inner_face_element.geometry_.volume();
	}

	return volumes;
}

template <ushort space_dimension>
std::vector<Euclidean_Vector<space_dimension>> Grid<space_dimension>::inner_face_normals_at_center(const std::vector<std::pair<uint, uint>>& oc_nc_index_pairs) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto& inner_face_elements = this->elements.inner_face_elements;

	const auto num_inner_face = inner_face_elements.size();
	std::vector<Euclidean_Vector<space_dimension>> normals(num_inner_face);

	for (uint i = 0; i < num_inner_face; ++i) {
		const auto [oc_index, nc_index] = oc_nc_index_pairs[i];

		const auto& oc_element = cell_elements[oc_index];
		const auto& inner_face_element = inner_face_elements[i];
		const auto  center = inner_face_element.geometry_.center_node();

		normals[i] = inner_face_element.normalized_normal_vector(oc_element, center);
	}

	return normals;
}

template <ushort space_dimension>
std::vector<std::vector<Euclidean_Vector<space_dimension>>> Grid<space_dimension>::inner_face_set_of_normals(const std::vector<uint>& oc_indexes, const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_nodes) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto& inner_face_elements = this->elements.inner_face_elements;

	const auto num_inner_face = inner_face_elements.size();
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> inner_face_set_of_normals(num_inner_face);

	for (uint i = 0; i < num_inner_face; ++i) {
		const auto& oc_element = cell_elements[oc_indexes[i]];
		const auto& inner_face_element = inner_face_elements[i];

		inner_face_set_of_normals[i] = inner_face_element.normalized_normal_vectors(oc_element, set_of_nodes[i]);
	}

	return inner_face_set_of_normals;
}


template <ushort space_dimension>
std::vector<std::pair<Euclidean_Vector<space_dimension>, Euclidean_Vector<space_dimension>>> Grid<space_dimension>::inner_face_oc_nc_to_face_pairs(const std::vector<std::pair<uint, uint>>& oc_nc_index_pairs) const {
	const auto& inner_face_elements = this->elements.inner_face_elements;

	const auto num_inner_face = inner_face_elements.size();
	std::vector<std::pair<Euclidean_Vector<space_dimension>, Euclidean_Vector<space_dimension>>> oc_nc_to_face_pairs(num_inner_face);

	const auto cell_center_nodes = this->cell_center_nodes();

	for (size_t i = 0; i < num_inner_face; ++i) {
		const auto [oc_index, nc_index] = oc_nc_index_pairs[i];

		const auto oc_center = cell_center_nodes[oc_index];
		const auto nc_center = cell_center_nodes[nc_index];
		const auto inner_face_center = inner_face_elements[i].geometry_.center_node();

		const auto oc_to_face_vector = inner_face_center - oc_center;
		const auto nc_to_face_vector = inner_face_center - nc_center;

		oc_nc_to_face_pairs[i] = std::make_pair(oc_to_face_vector, nc_to_face_vector);
	}

	return oc_nc_to_face_pairs;
}

template <ushort space_dimension>
std::vector<Quadrature_Rule<space_dimension>> Grid<space_dimension>::inner_face_quadrature_rules(const ushort polynomial_degree) const {
	const auto& inner_face_elements = this->elements.inner_face_elements;

	const auto num_inner_face = inner_face_elements.size();
	std::vector<Quadrature_Rule<space_dimension>> quadrature_rules(num_inner_face);

	for (size_t i = 0; i < num_inner_face; ++i) {
		const auto& inner_face_geometry = inner_face_elements[i].geometry_;
		quadrature_rules[i] = inner_face_geometry.get_quadrature_rule(polynomial_degree);
	}

	return quadrature_rules;
}



template <ushort space_dimension>
std::vector<std::pair<uint, uint>> Grid<space_dimension>::periodic_boundary_oc_nc_index_pairs(void) const {
	const auto& pbdry_element_pairs = this->elements.periodic_boundary_element_pairs;

	const auto num_pbdry_pair = pbdry_element_pairs.size();
	std::vector<std::pair<uint, uint>> pbdry_oc_nc_index_pairs(num_pbdry_pair);

	for (size_t i = 0; i < num_pbdry_pair; ++i) {
		const auto& [oc_side_element, nc_side_element] = pbdry_element_pairs[i];

		const auto oc_indexes = this->find_cell_indexes_have_these_vnodes_ignore_pbdry(oc_side_element.vertex_node_indexes());
		const auto nc_indexes = this->find_cell_indexes_have_these_vnodes_ignore_pbdry(nc_side_element.vertex_node_indexes());
		dynamic_require(oc_indexes.size() == 1, "periodic boundary should have unique owner cell");
		dynamic_require(nc_indexes.size() == 1, "periodic boundary should have unique neighbor cell");

		const auto oc_index = oc_indexes.front();
		const auto nc_index = nc_indexes.front();
		pbdry_oc_nc_index_pairs[i] = { oc_index,nc_index };
	}

	return pbdry_oc_nc_index_pairs;
}

template <ushort space_dimension>
std::vector<double> Grid<space_dimension>::periodic_boundary_volumes(void) const {
	const auto& pbdry_element_pairs = this->elements.periodic_boundary_element_pairs;

	const auto num_pbdry_pair = pbdry_element_pairs.size();
	std::vector<double> volumes(num_pbdry_pair);

	for (uint i = 0; i < num_pbdry_pair; ++i) {
		const auto& [oc_side_pbdry_element, nc_side_pbdry_element] = pbdry_element_pairs[i];
		volumes[i] = oc_side_pbdry_element.geometry_.volume();
	}

	return volumes;
}

template <ushort space_dimension>
std::vector<Euclidean_Vector<space_dimension>> Grid<space_dimension>::periodic_boundary_normals_at_center(const std::vector<std::pair<uint, uint>>& oc_nc_index_pairs) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto& pbdry_element_pairs = this->elements.periodic_boundary_element_pairs;

	const auto num_pbdry_pair = pbdry_element_pairs.size();
	std::vector<Euclidean_Vector<space_dimension>> normals(num_pbdry_pair);

	for (uint i = 0; i < num_pbdry_pair; ++i) {
		const auto [oc_index, nc_index] = oc_nc_index_pairs[i];

		const auto& oc_element = cell_elements[oc_index];
		const auto& [oc_side_element, nc_side_element] = pbdry_element_pairs[i];
		const auto  center = oc_side_element.geometry_.center_node();

		normals[i] = oc_side_element.normalized_normal_vector(oc_element, center);
	}

	return normals;
}

template <ushort space_dimension>
std::vector<std::vector<Euclidean_Vector<space_dimension>>> Grid<space_dimension>::periodic_boundary_set_of_normals(const std::vector<uint>& oc_indexes, const std::vector<std::vector<Euclidean_Vector<space_dimension>>>& set_of_oc_side_nodes) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto& pbdry_element_pairs = this->elements.periodic_boundary_element_pairs;

	const auto num_pbdry_pair = pbdry_element_pairs.size();
	std::vector<std::vector<Euclidean_Vector<space_dimension>>> pbdry_set_of_normals(num_pbdry_pair);

	for (uint i = 0; i < num_pbdry_pair; ++i) {
		const auto& oc_element = cell_elements[oc_indexes[i]];
		const auto& [oc_side_element, nc_side_element] = pbdry_element_pairs[i];

		pbdry_set_of_normals[i] = oc_side_element.normalized_normal_vectors(oc_element, set_of_oc_side_nodes[i]);
	}

	return pbdry_set_of_normals;
}

template <ushort space_dimension>
std::vector<std::pair<Euclidean_Vector<space_dimension>, Euclidean_Vector<space_dimension>>> Grid<space_dimension>::periodic_boundary_oc_nc_to_oc_nc_side_face_pairs(const std::vector<std::pair<uint, uint>>& oc_nc_index_pairs) const {
	const auto& pbdry_element_pairs = this->elements.periodic_boundary_element_pairs;

	const auto num_pbdry_pair = pbdry_element_pairs.size();
	std::vector<std::pair<Euclidean_Vector<space_dimension>, Euclidean_Vector<space_dimension>>> oc_nc_to_face_pairs(num_pbdry_pair);

	const auto cell_center_nodes = this->cell_center_nodes();

	for (size_t i = 0; i < num_pbdry_pair; ++i) {
		const auto [oc_index, nc_index] = oc_nc_index_pairs[i];
		const auto [oc_side_element, nc_side_element] = pbdry_element_pairs[i];

		const auto oc_center = cell_center_nodes[oc_index];
		const auto nc_center = cell_center_nodes[nc_index];

		const auto oc_side_center = oc_side_element.geometry_.center_node();
		const auto nc_side_center = nc_side_element.geometry_.center_node();

		const auto oc_to_face_vector = oc_side_center - oc_center;
		const auto nc_to_face_vector = nc_side_center - nc_center;

		oc_nc_to_face_pairs[i] = std::make_pair(oc_to_face_vector, nc_to_face_vector);
	}

	return oc_nc_to_face_pairs;
}

template <ushort space_dimension>
std::vector<std::pair<Quadrature_Rule<space_dimension>, Quadrature_Rule<space_dimension>>> Grid<space_dimension>::periodic_boundary_quadrature_rule_pairs(const ushort polynomial_degree) const {
	const auto& pbdry_element_pairs = this->elements.periodic_boundary_element_pairs;

	const auto num_pbdry_pair = pbdry_element_pairs.size();
	std::vector<std::pair<Quadrature_Rule<space_dimension>, Quadrature_Rule<space_dimension>>> quadrature_rule_pairs(num_pbdry_pair);

	for (size_t i = 0; i < num_pbdry_pair; ++i) {
		const auto& [oc_side_element, nc_side_element] = pbdry_element_pairs[i];
		const auto& oc_side_geometry = oc_side_element.geometry_;
		const auto& nc_side_geometry = nc_side_element.geometry_;

		quadrature_rule_pairs[i] = std::make_pair(oc_side_geometry.get_quadrature_rule(polynomial_degree), nc_side_geometry.get_quadrature_rule(polynomial_degree));
	}

	return quadrature_rule_pairs;
}

template <ushort space_dimension>
std::unordered_map<uint, std::set<uint>> Grid<space_dimension>::pbdry_vnode_index_to_matched_pbdry_vnode_index_set(void) const {
	std::unordered_map<uint, std::set<uint>> pbdry_vnode_index_to_matched_vnode_index_set;

	for (const auto& [oc_side_element, nc_side_element] : this->elements.periodic_boundary_element_pairs) {
		const auto oc_side_vnode_indexes = oc_side_element.vertex_node_indexes();
		const auto nc_side_vnode_indexes = nc_side_element.vertex_node_indexes();

		const auto num_vnode = oc_side_vnode_indexes.size();
		for (ushort i = 0; i < num_vnode; ++i) {
			const auto i_vnode_index = oc_side_vnode_indexes[i];
			const auto j_vnode_index = nc_side_vnode_indexes[i];

			if (!pbdry_vnode_index_to_matched_vnode_index_set.contains(i_vnode_index))
				pbdry_vnode_index_to_matched_vnode_index_set.emplace(i_vnode_index, std::set<uint>());

			if (!pbdry_vnode_index_to_matched_vnode_index_set.contains(j_vnode_index))
				pbdry_vnode_index_to_matched_vnode_index_set.emplace(j_vnode_index, std::set<uint>());

			pbdry_vnode_index_to_matched_vnode_index_set.at(i_vnode_index).insert(j_vnode_index);
			pbdry_vnode_index_to_matched_vnode_index_set.at(j_vnode_index).insert(i_vnode_index);
		}
	}

	//consider pbdry conner
	for (ushort i = 0; i < space_dimension - 1; ++i) {
		for (auto& [vnode_index, matched_vnode_index_set] : pbdry_vnode_index_to_matched_vnode_index_set) {
			if (matched_vnode_index_set.size() == 1)
				continue;

			for (const auto matched_vnode_index : matched_vnode_index_set) {
				const auto& other_matched_vnode_index_set = pbdry_vnode_index_to_matched_vnode_index_set.at(matched_vnode_index);

				auto& i_set = matched_vnode_index_set;
				const auto& j_set = other_matched_vnode_index_set;
				const auto difference = ms::set_difference(j_set, i_set);

				if (difference.empty())
					continue;

				i_set.insert(difference.begin(), difference.end());
				i_set.erase(vnode_index);
			}
		}
	}

	return pbdry_vnode_index_to_matched_vnode_index_set;
}


template <ushort space_dimension>
std::vector<Ghost_Cell<space_dimension>> Grid<space_dimension>::make_ghost_cells(void) const {		
	const auto bdry_owner_cell_indexes =  this->boundary_owner_cell_indexes();
	const auto num_boundary = bdry_owner_cell_indexes.size();

	const auto pbdry_oc_nc_index_pairs = this->periodic_boundary_oc_nc_index_pairs();
	const auto num_pbdry_pair = pbdry_oc_nc_index_pairs.size();

	std::vector<Ghost_Cell<space_dimension>> ghost_cells;
	ghost_cells.reserve(num_boundary + 2 * num_pbdry_pair);

	const auto cell_center_nodes = this->cell_center_nodes();

	const auto oc_to_boundary = this->boundary_owner_cell_to_faces(bdry_owner_cell_indexes);

	for (uint i = 0; i < num_boundary; ++i) {
		const auto& bdry_element = this->elements.boundary_elements[i];
		const auto oc_index = bdry_owner_cell_indexes[i];

		const auto ghost_cell_center = cell_center_nodes[oc_index] + 2 * oc_to_boundary[i];
		const auto ghost_cell_related_type = bdry_element.type();
		const auto ghost_cell_solution_related_cell_index = oc_index;
		const auto ghost_cell_face_share_cell_index = oc_index;

		ghost_cells.push_back({ ghost_cell_center, ghost_cell_related_type, ghost_cell_solution_related_cell_index,ghost_cell_face_share_cell_index });
	}

	for (uint i = 0; i < num_pbdry_pair; ++i) {
		const auto& [oc_side_element, nc_side_element] = this->elements.periodic_boundary_element_pairs[i];
		const auto [oc_index, nc_index] = pbdry_oc_nc_index_pairs[i];

		const auto oc_center_node = cell_center_nodes[oc_index];
		const auto nc_center_node = cell_center_nodes[nc_index];
		const auto ref_oc_side_vnode = oc_side_element.geometry_.get_nodes().front();
		const auto ref_nc_side_vnode = nc_side_element.geometry_.get_nodes().front();

		const auto oc_center_to_vnode = ref_oc_side_vnode - oc_center_node;
		const auto oc_vnode_to_center = -1 * oc_center_to_vnode;

		const auto nc_center_to_vnode = ref_nc_side_vnode - nc_center_node;
		const auto nc_vnode_to_center = -1 * nc_center_to_vnode;
				
		const auto oc_side_ghost_cell_center = cell_center_nodes[oc_index] + oc_center_to_vnode + nc_vnode_to_center;
		const auto oc_side_ghost_cell_related_type = ElementType::periodic;
		const auto oc_side_ghost_cell_solution_related_cell_index = nc_index;
		const auto oc_side_ghost_cell_face_share_cell_index = oc_index;
		ghost_cells.push_back({ oc_side_ghost_cell_center, oc_side_ghost_cell_related_type, oc_side_ghost_cell_solution_related_cell_index, oc_side_ghost_cell_face_share_cell_index });

		const auto nc_side_ghost_cell_center = cell_center_nodes[nc_index] + nc_center_to_vnode + oc_vnode_to_center;
		const auto nc_side_ghost_cell_related_type = ElementType::periodic;		
		const auto nc_side_ghost_cell_solution_related_cell_index = oc_index;
		const auto nc_side_ghost_cell_face_share_cell_index = nc_index;
		ghost_cells.push_back({ nc_side_ghost_cell_center, nc_side_ghost_cell_related_type, nc_side_ghost_cell_solution_related_cell_index, nc_side_ghost_cell_face_share_cell_index });
	}

	return ghost_cells;
}

template <ushort space_dimension>
std::vector<std::vector<uint>> Grid<space_dimension>::ANN_indexes(void) const {
	const auto& cell_elements = this->elements.cell_elements;
	
	const auto bdry_oc_indexes = this->boundary_owner_cell_indexes();

	const auto num_cell = cell_elements.size();
	std::vector<std::vector<uint>> set_of_ANN_indexes(num_cell);
		
	//only work RQ mesh
	std::array<uint, 4> location1 = { 5,3,1,7 };
	std::array<uint, 4> location2 = { 6,4,2,8 };

	for (uint i = 0; i < num_cell; ++i) {
		if (ms::contains(bdry_oc_indexes, i))
			continue;

		const auto& cell_element = cell_elements[i];
		
		std::vector<uint> ANN_indexes(9, -1);
		ANN_indexes[0] = i;
		
		auto set_of_face_vnode_indexes = cell_element.set_of_face_vertex_node_indexes();
		const auto num_face = set_of_face_vnode_indexes.size();
		
		for (ushort j = 0; j < num_face; ++j) {
			const auto face_vnode_indexes = set_of_face_vnode_indexes[j];						
			const auto face_share_cell_index = this->find_face_share_cell_index_consider_pbdry(i, face_vnode_indexes);
			ANN_indexes[location1[j]] = face_share_cell_index.value();
		}

		const auto vnode_indexes = cell_element.vertex_node_indexes();
		const auto num_vnode = vnode_indexes.size();

		for (ushort j = 0; j < num_vnode; ++j) {
			const auto vnode_index = vnode_indexes[j];
			auto vnode_share_cell_index_set = this->vnode_index_to_share_cell_index_set_consider_pbdry_.at(vnode_index);
			vnode_share_cell_index_set.erase(i);

			std::set<uint> temp(ANN_indexes.begin(), ANN_indexes.end());

			auto difference_indexes = ms::set_difference(vnode_share_cell_index_set, temp);
			dynamic_require(difference_indexes.size() == 1, "difference indexes should be unique");			

			ANN_indexes[location2[j]] = difference_indexes.front();
		}

		set_of_ANN_indexes[i] = ANN_indexes;
	}

	return set_of_ANN_indexes;
	//only work RQ mesh
}

template <ushort space_dimension>
std::vector<std::vector<uint>> Grid<space_dimension>::set_of_face_share_cell_indexes_ignore_pbdry(void) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto num_cell = cell_elements.size();

	std::vector<std::vector<uint>> set_of_face_share_cell_indexes;
	set_of_face_share_cell_indexes.reserve(num_cell);

	for (uint i = 0; i < num_cell; ++i) {
		const auto& element = cell_elements[i];
		const auto& geometry = element.geometry_;

		const auto face_vnode_indexes_set = element.set_of_face_vertex_node_indexes();
		const auto num_face = face_vnode_indexes_set.size();

		std::vector<uint> face_share_cell_indexes;
		face_share_cell_indexes.reserve(num_face);

		for (const auto& face_vnode_indexes : face_vnode_indexes_set) {
			auto share_cell_index_opt = this->find_face_share_cell_index_ignore_pbdry(i, face_vnode_indexes);

			if (share_cell_index_opt.has_value())
				face_share_cell_indexes.push_back(share_cell_index_opt.value());
		}

		set_of_face_share_cell_indexes.push_back(std::move(face_share_cell_indexes));
	}

	return set_of_face_share_cell_indexes;
}

template <ushort space_dimension>
std::vector<std::vector<size_t>> Grid<space_dimension>::set_of_face_share_cell_indexes_consider_pbdry(void) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto num_cell = cell_elements.size();

	std::vector<std::vector<size_t>> set_of_face_share_cell_indexes;
	set_of_face_share_cell_indexes.reserve(num_cell);

	for (size_t i = 0; i < num_cell; ++i) {
		const auto& element = cell_elements[i];
		const auto& geometry = element.geometry_;

		const auto face_vnode_indexes_set = element.set_of_face_vertex_node_indexes();
		const auto num_face = face_vnode_indexes_set.size();

		std::vector<size_t> face_share_cell_indexes;
		face_share_cell_indexes.reserve(num_face);

		for (const auto& face_vnode_indexes : face_vnode_indexes_set) {			
			auto this_face_share_cell_indexes = this->find_cell_indexes_have_these_vnodes_consider_pbdry(face_vnode_indexes);

			const auto my_index_pos_iter = std::find(this_face_share_cell_indexes.begin(), this_face_share_cell_indexes.end(), i);
			dynamic_require(my_index_pos_iter != this_face_share_cell_indexes.end(), "my index should be included in this face share cell indexes");

			this_face_share_cell_indexes.erase(my_index_pos_iter);
			dynamic_require(this_face_share_cell_indexes.size() <= 1, "face share cell should be unique or absent");

			face_share_cell_indexes.push_back(this_face_share_cell_indexes.front());//need to fix
		}

		set_of_face_share_cell_indexes.push_back(std::move(face_share_cell_indexes));
	}

	return set_of_face_share_cell_indexes; //this is not ordered set
}


template <ushort space_dimension>
std::vector<std::vector<size_t>> Grid<space_dimension>::set_of_vertex_share_cell_indexes_consider_pbdry(void) const {
	const auto& cell_elements = this->elements.cell_elements;
	const auto num_cell = cell_elements.size();

	std::vector<std::vector<size_t>> set_of_vertex_share_cell_indexes;
	set_of_vertex_share_cell_indexes.reserve(num_cell);

	for (size_t i = 0; i < num_cell; ++i) {
		const auto& element = cell_elements[i];
		const auto& geometry = element.geometry_;
				
		auto vnode_indexes = element.vertex_node_indexes();
		std::set<size_t> vertex_share_cell_index_set;
		for (const auto vnode_index : vnode_indexes) {
			const auto& vnode_share_cell_index_set = this->vnode_index_to_share_cell_index_set_consider_pbdry_.at(vnode_index);
			vertex_share_cell_index_set.insert(vnode_share_cell_index_set.begin(), vnode_share_cell_index_set.end());
		}
		vertex_share_cell_index_set.erase(i);

		set_of_vertex_share_cell_indexes.push_back({ vertex_share_cell_index_set.begin(), vertex_share_cell_index_set.end() });
	}

	return set_of_vertex_share_cell_indexes;
}

template <ushort space_dimension>
std::vector<std::vector<uint>> Grid<space_dimension>::set_of_face_share_ghost_cell_indexes(const std::vector<Ghost_Cell<space_dimension>>& ghost_cells) const {
	std::vector<std::vector<uint>> set_of_face_share_ghost_cell_indexes(this->elements.cell_elements.size());

	const auto num_ghost_cell = ghost_cells.size();

	for (uint i = 0; i < num_ghost_cell; ++i) 
		set_of_face_share_ghost_cell_indexes[ghost_cells[i].face_share_cell_index].push_back(i);

	return set_of_face_share_ghost_cell_indexes;
}



template<ushort space_dimension>
std::vector<uint> Grid<space_dimension>::find_cell_indexes_have_these_vnodes_ignore_pbdry(const std::vector<uint>& vnode_indexes) const {
	std::vector<uint> this_face_share_cell_indexes;

	const auto& set0 = this->vnode_index_to_share_cell_index_set_ignore_pbdry_.at(vnode_indexes[0]);
	const auto& set1 = this->vnode_index_to_share_cell_index_set_ignore_pbdry_.at(vnode_indexes[1]);
	std::set_intersection(set0.begin(), set0.end(), set1.begin(), set1.end(), std::back_inserter(this_face_share_cell_indexes));

	const auto num_face_vnode = vnode_indexes.size();

	if (2 < num_face_vnode) {
		std::vector<uint> temp;
		for (ushort i = 2; i < num_face_vnode; ++i) {
			const auto& set_i = this->vnode_index_to_share_cell_index_set_ignore_pbdry_.at(vnode_indexes[i]);

			std::set_intersection(this_face_share_cell_indexes.begin(), this_face_share_cell_indexes.end(), set_i.begin(), set_i.end(), std::back_inserter(temp));
			std::swap(this_face_share_cell_indexes, temp);
			temp.clear();
		}
	}

	return this_face_share_cell_indexes;
}

template<ushort space_dimension>
std::vector<uint> Grid<space_dimension>::find_cell_indexes_have_these_vnodes_consider_pbdry(const std::vector<uint>& vnode_indexes) const {
	std::vector<uint> this_face_share_cell_indexes;

	const auto& set0 = this->vnode_index_to_share_cell_index_set_consider_pbdry_.at(vnode_indexes[0]);
	const auto& set1 = this->vnode_index_to_share_cell_index_set_consider_pbdry_.at(vnode_indexes[1]);
	std::set_intersection(set0.begin(), set0.end(), set1.begin(), set1.end(), std::back_inserter(this_face_share_cell_indexes));

	const auto num_face_vnode = vnode_indexes.size();

	if (2 < num_face_vnode) {
		std::vector<uint> temp;
		for (ushort i = 2; i < num_face_vnode; ++i) {
			const auto& set_i = this->vnode_index_to_share_cell_index_set_consider_pbdry_.at(vnode_indexes[i]);

			std::set_intersection(this_face_share_cell_indexes.begin(), this_face_share_cell_indexes.end(), set_i.begin(), set_i.end(), std::back_inserter(temp));
			std::swap(this_face_share_cell_indexes, temp);
			temp.clear();
		}
	}

	return this_face_share_cell_indexes;
}

template<ushort space_dimension>
std::optional<uint> Grid<space_dimension>::find_face_share_cell_index_ignore_pbdry(const uint my_index, const std::vector<uint>& my_face_vnode_indexes) const {
	auto cell_indexes = this->find_cell_indexes_have_these_vnodes_ignore_pbdry(my_face_vnode_indexes);

	const auto my_index_iter = ms::find(cell_indexes, my_index);
	dynamic_require(my_index_iter != cell_indexes.end(), "my index should be included in face share cell indexes");

	cell_indexes.erase(my_index_iter);
	dynamic_require(cell_indexes.size() <= 1, "face share cell should be unique or absent");

	if (cell_indexes.empty())
		return std::nullopt;
	else
		return cell_indexes.front();
}

template<ushort space_dimension>
std::optional<uint> Grid<space_dimension>::find_face_share_cell_index_consider_pbdry(const uint my_index, const std::vector<uint>& my_face_vnode_indexes) const {
	auto cell_indexes = this->find_cell_indexes_have_these_vnodes_consider_pbdry(my_face_vnode_indexes);

	const auto my_index_iter = ms::find(cell_indexes, my_index);
	dynamic_require(my_index_iter != cell_indexes.end(), "my index should be included in face share cell indexes");

	cell_indexes.erase(my_index_iter);
	dynamic_require(cell_indexes.size() <= 1, "face share cell should be unique or absent");

	if (cell_indexes.empty())
		return std::nullopt;
	else
		return cell_indexes.front();
}

template<ushort space_dimension>
std::vector<std::vector<uint>> Grid<space_dimension>::set_of_boundary_vnode_indexes(void) const {
	const auto& boundary_elements = this->elements.boundary_elements;

	const auto num_bdry = boundary_elements.size();
	std::vector<std::vector<uint>> set_of_vnode_indexes;
	set_of_vnode_indexes.reserve(num_bdry);

	for (const auto& element : boundary_elements)
		set_of_vnode_indexes.push_back(element.vertex_node_indexes());

	return set_of_vnode_indexes;
}

template<ushort space_dimension>
std::vector<std::vector<uint>> Grid<space_dimension>::set_of_periodic_boundary_vnode_indexes(void) const {
	const auto& pbdry_element_pairs = this->elements.periodic_boundary_element_pairs;

	const auto num_pbdry = pbdry_element_pairs.size() * 2;
	std::vector<std::vector<uint>> set_of_vnode_indexes;
	set_of_vnode_indexes.reserve(num_pbdry);

	for (const auto& [element1, element2] : pbdry_element_pairs) {
		set_of_vnode_indexes.push_back(element1.vertex_node_indexes());
		set_of_vnode_indexes.push_back(element2.vertex_node_indexes());
	}

	return set_of_vnode_indexes;
}