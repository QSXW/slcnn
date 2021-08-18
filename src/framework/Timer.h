#pragma once

#include <iostream>
#include <chrono>

namespace sl
{
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

		virtual ~Timer()
		{
			end = Clock::now();
			auto duration = std::chrono::duration<double, DefaultResolution>(end - start);
			fprintf(stdout, "Time End\nDuration: %.10f\n", duration);
		}

	  private:
		Clock::time_point start;
		Clock::time_point end;
	};
}
