#pragma once

#include <memory>
#include <cstdio>

#pragma warning(push, 0)
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#pragma warning(pop)

namespace sl {

	class  Log
	{
	public:
		static void Launch();

		template <class... Args>
		static constexpr inline void Warn(Args&& ... args)
		{
			logger->warn(std::forward<Args>(args)...);
		}

		template <class... Args>
		static constexpr inline void Info(Args&& ... args)
		{
			logger->info(std::forward<Args>(args)...);
		}

		template <class... Args>
		static constexpr inline void Debug(Args&& ... args)
		{
			logger->debug(std::forward<Args>(args)...);
		}

		template <class... Args>
		static constexpr inline void Error(Args&& ... args)
		{
			logger->error(std::forward<Args>(args)...);
		}

		template <class... Args>
		static constexpr inline void Critical(Args&& ... args)
		{
			logger->critical(std::forward<Args>(args)...);
		}

	private:
		static std::shared_ptr<spdlog::logger> logger;
	};

}

