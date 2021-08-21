#pragma once

#include <iostream>
#include <chrono>

#include "Log.h"

namespace sl
{
#if defined( _DEBUG ) || defined( DEBUG )
#    define TIME_SUPERVISED Timer t{ __FILE__, __LINE__, __func__ };
#else
#    define	TIME_SUPERVISED
#endif

    class Timer
    {
	public:
		using Seconds      = std::ratio<1>;
		using Milliseconds = std::ratio<1, 1000>;
		using Microseconds = std::ratio<1, 1000000>;
		using Nanoseconds  = std::ratio<1, 1000000000>;

		using Clock             = std::chrono::steady_clock;
		using DefaultResolution = Seconds;

		Timer()
		{
			start = Clock::now();
			Log::Info("Timer Start");
		}

		Timer(const char *file, const int line, const char *func)
		{
			start = Clock::now();
			Log::Info("Running => {0}  at  {1}  in  {2}", file, line, func);
		}

		virtual ~Timer()
		{
			end = Clock::now();
			auto duration = std::chrono::duration<double, DefaultResolution>(end - start);
			Log::Info("Duration: {0}", duration.count());
		}

	private:
		Clock::time_point start;
		Clock::time_point end;
	};
}
