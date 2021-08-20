#pragma once

#include <iostream>
#include <chrono>

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
			fprintf(stdout, "Timer Start\n");
		}

		Timer(const char *file, const int line, const char *func)
		{
			start = Clock::now();
			fprintf(stdout, "\nRunning => %s  at  %d  in  %s\n", file, line, func);
		}

		virtual ~Timer()
		{
			end = Clock::now();
			auto duration = std::chrono::duration<double, DefaultResolution>(end - start);
			fprintf(stdout, "Duration: %.10gs\n\n", duration.count());
		}

	private:
		Clock::time_point start;
		Clock::time_point end;
	};
}
