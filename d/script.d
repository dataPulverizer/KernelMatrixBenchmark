import arrays;
import kernel;

import std.conv: to;
import std.meta: AliasSeq;
import std.algorithm : sum;
import std.stdio: File, writeln;
import std.typecons: tuple, Tuple;
import std.datetime.stopwatch: AutoStart, StopWatch;

/**
  To compile:
  ldc2 script.d kernel.d math.d arrays.d -O --boundscheck=off --ffast-math -mcpu=native
  /usr/bin/time -v ./script
  
  ldc2 --mcpu=help
  ldc2 script.d kernel.d math.d arrays.d -O --boundscheck=off --ffast-math --mcpu=core-avx2 -mattr=+avx2,+sse4.1,+sse4.2
  /usr/bin/time -v ./script
*/

auto bench(alias K, T)(K!T kernel, long[] n)
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
      t = sw.peek.total!"nsecs"/1000_000_000.0;
      sw.reset();
    }
    times[i] = sum(_times[])/3.0;
    version(verbose)
    {
      writeln("Average time for n = ", n[i], ", ", times[i], " seconds.");
      writeln("Detailed times: ", _times, "\n");
    }
  }
  return tuple(n, times);
}

auto runKernelBenchmark(KS)(KS kernels, long[] n)
{
  auto tmp = bench(kernels[0], n);
  alias R = typeof(tmp);
  R[kernels.length] results;
  results[0] = tmp;
  static foreach(i; 1..kernels.length)
  {
    version(verbose)
    {
      writeln("Running benchmarks for ", kernels[i]);
    }
    results[i] = bench(kernels[i], n);
  }
  return results;
}

void writeRow(File file, string[] row)
{
  string line = "";
  foreach(i; 0..(row.length - 1))
    line ~= row[i] ~ ",";
  line ~= row[row.length - 1] ~ "\n";
  file.write(line);
  return;
}

void runAllKernelBenchmarks(T = float)()
{
  auto kernels = tuple(DotProduct!(T)(),   Gaussian!(T)(1), Polynomial!(T)(2.5f, 1),
                            Exponential!(T)(1), Log!(T)(3),      Cauchy!(T)(1),
                            Power!(T)(2.5f),     Wave!(T)(1),     Sigmoid!(T)(1, 1));
  auto kernelNames = ["DotProduct",  "Gaussian", "Polynomial",
                       "Exponential", "Log",      "Cauchy",
                       "Power",       "Wave",     "Sigmoid"];
  //long[] n = [100L, 500L, 1000L];
  long[] n = [1000L, 5000L, 10_000L, 20_000L, 30_000L];
  auto results = runKernelBenchmark(kernels, n);

  auto table = new string[][] (n.length * kernels.length + 1, 4);
  table[0][] = ["language", "kernel", "nitems", "time"];
  auto tmp = ["D", "", "", ""];
  while(true)
  {
    auto k = 1;
    foreach(i; 0..kernels.length)
    {
      tmp = ["D", kernelNames[i], "", ""];
      foreach(j; 0..n.length)
      {
        tmp[2] = to!(string)(results[i][0][j]);
        tmp[3] = to!(string)(results[i][1][j]);
        table[k][] = tmp.dup;
        k += 1;
      }
    }
    if(k > (table.length - 1))
    {
      break;
    }
  }
  version(fastmath)
  {
    auto file = File("../fmdata/dBench.csv", "w");
  }else{
    auto file = File("../data/dBench.csv", "w");
  }
  foreach(row; table)
    file.writeRow(row);
}

void main()
{
  runAllKernelBenchmarks();
}
