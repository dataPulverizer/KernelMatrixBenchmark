import arrays;
import kernelmatrix;
import std.stdio: writeln;
import std.datetime.stopwatch: AutoStart, StopWatch;

/**
  To compile
  ldc2 script.d kernelmatrix.d arrays.d -O5 --boundscheck=off --ffast-math && ./script
*/

auto bench(alias K, T)(K!T Kernel, long[] n, bool verbose = true)
{
  auto times = new double[n.length];
  auto sw = StopWatch(AutoStart.no);
  foreach(i; 0..n.length)
  {
    double[3] _times;
    auto data = createRandomMatrix!T(784L, n[i]);
    foreach(j; 0..3)
    {
      sw.start();
      auto mat = calculateKernelMatrix!(K!T, T)(Kernel, data);
      sw.stop();
      _times[j] = cast(double)sw.peek.total!"usecs"/1000_000;
      sw.reset();
    }
    times[i] = (_times[0] + _times[1] + _times[2])/3;
    if(verbose)
      writeln("Average time for n = ", n[i], ", ", times[i], " seconds.");
  }
  return times;
}

//bench: [0.009412, 0.271136, 1.43589, 11.267, 31.0559]
void main()
{
  auto Kernel = new DotProduct!float();
  writeln("bench: ", bench(Kernel, [1000L, 5000L, 
      10_000L, 20_000L, 30_000L]));
}

