#pragma once

#include <cstdint>
#include <new>

#if defined( _MSC_VER )
#   define SLALIGNED(x) __declspec(align(x))
#	define SLALIGNED_STRUCT(x) struct __declspec(align(x))
#	define SLALIGNED_CLASS(x) class __declspec(align(x)))
#   define SLASSEMBLY __asm
#pragma warning(disable: 4996)

#elif defined( __GNUC__ )
#   define SL_ALIGNED(x) __attribute__((aligned(x)))
#	define SL_ALIGNED_STRUCT(x) struct __attribute__((aligned(x)))
#	define SL_ALIGNED_CLASS(x) class __attribute__((aligned(x)))
#   define SL_ASSEMBLY __asm__
#endif

template <class T>
inline T *sl_aligned_malloc(std::size_t size, size_t align)
{
	auto ptr = _aligned_malloc(size * sizeof(T), align);
	return ptr ? static_cast<T *>(ptr) : throw std::bad_alloc{};
}

inline void sl_aligned_free(void* ptr)
{
	_aligned_free(ptr);
}

namespace sl
{
inline namespace type
{
	using INT8   = signed   char;
	using UINT8  = unsigned char;
	using INT16  = signed   short;
	using UINT16 = unsigned short;
	using INT32  = signed   int;
	using UINT32 = unsigned int;
	using INT64  = int64_t;
	using UINT64 = uint64_t;
	
	template <class T1, class T2>
	inline constexpr bool typeof()
	{
		return std::is_same_v<T1, T2>;
	}

	template <class T1, class T2, class T3>
	inline constexpr bool typeof()
	{
		if constexpr (std::is_same_v<T1, T2> || std::is_same_v<T1, T3>)
		{
			return true;
		}
		return false;
	}

	template <class T1, class T2, class T3, class T4>
	inline constexpr bool typeof()
	{
		if constexpr (std::is_same_v<T1, T2> ||
			std::is_same_v<T1, T3> ||
			std::is_same_v<T1, T4>)
		{
			return true;
		}
		return false;
	}
}

	template <class T>
	inline constexpr size_t asize(T &&t)
	{
		return sizeof(t) / sizeof(t[0]);
	}
}
