#pragma once
#include <type_traits>
#include <string>

class SDM {};// spatial discrete meethod

namespace ms {
	template <typename T>
	inline constexpr bool is_spatial_discrete_method = std::is_base_of_v<SDM, T>;
}

class FVM : public SDM 
{
public:
	static std::string name(void) { return "FVM"; };
};


class HOM : public SDM {};

// Spatial Discrete Method(SDM)�� ��ȭ�� ���� ��Ÿ���� Semi_Discrete_Equation(SDE)�� Grid_Info_Extractor(GIE)�� ��������
// Method ���� ����� ��ȭ������ ǥ������ �ʱ� ������ ��Ӱ� �ռ����� �������� �ٷ��� �ʰ�
// SDE�� GIE�� template class�� ������ �� Ư��ȭ�� ���� �������� �ٷ��.