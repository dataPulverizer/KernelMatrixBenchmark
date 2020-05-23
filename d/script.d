import arrays;
import kernelmatrix;
import std.algorithm : sum;
import std.stdio: writeln;
import std.datetime.stopwatch: AutoStart, StopWatch;

/**
  To compile
  /usr/bin/time -v ldc2 script.d kernelmatrix.d arrays.d -O5 --boundscheck=off --ffast-math && ./script
*/

auto bench(alias K, T)(K!T kernel, long[] n, bool verbose = true)
{
  auto times = new double[n.length];
  auto sw = StopWatch(AutoStart.no);
  foreach(i; 0..n.length)
  {
    double[3] _times;
    auto data = createRandomMatrix!T(784L, n[i]);
    foreach(ref t; _times[])
    {
      sw.start();
      auto mat = calculateKernelMatrix!(K!T, T)(kernel, data);
      sw.stop();
      t = sw.peek.total!"usecs"/1000_000.0;
      sw.reset();
    }
    times[i] = sum(_times[])/3.0;
    if(verbose)
    {
      writeln("Average time for n = ", n[i], ", ", times[i], " seconds.");
      writeln("Detailed times: ", _times);
    }
  }
  return times;
}

//bench: [0.009412, 0.271136, 1.43589, 11.267, 31.0559]
void main()
{
  auto kernel = new DotProduct!float();
  writeln("bench: ", bench(kernel, [1000L, 5000L, 
      10_000L, 20_000L, 30_000L]));
}

