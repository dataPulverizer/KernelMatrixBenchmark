import ndslice.kernels;

import mir.ndslice.slice;
import mir.ndslice.allocation;
import mir.ndslice.topology: iota;
import mir.random.algorithm: randomSlice;
import mir.random.variable: UniformVariable;

import std.conv: to;
import std.meta: AliasSeq;
import std.algorithm : sum;
import std.stdio: File, writeln;
import std.typecons: tuple, Tuple;
import std.datetime.stopwatch: AutoStart, StopWatch;

/**
  To compile:
  dub run --compiler=ldc2 --build=release
*/

auto bench(alias K, T)(K!(T) kernel, long[] n, bool verbose = true)
{
  auto times = new double[n.length];
  auto sw = StopWatch(AutoStart.no);
  foreach(i; 0..n.length)
  {
    double[3] _times;
    auto data = UniformVariable!T(0, 1).randomSlice(n[i], 784L);
    foreach(ref t; _times[])
    {
      sw.start();
      auto mat = calculateKernelMatrix!(K, T)(kernel, data);
      sw.stop();
      t = sw.peek.total!"nsecs"/1000_000_000.0;
      sw.reset();
    }
    times[i] = sum(_times[])/3.0;
    if(verbose)
    {
      writeln("Average time for n = ", n[i], ", ", times[i], " seconds.");
      writeln("Detailed times: ", _times, "\n");
    }
  }
  return tuple(n, times);
}

auto runKernelBenchmark(KS)(KS kernels, long[] n, bool verbose = true)
{
  auto tmp = bench(kernels[0], n, verbose);
  alias R = typeof(tmp);
  R[kernels.length] results;
  results[0] = tmp;
  static foreach(i; 1..kernels.length)
  {
    if(verbose)
    {
      writeln("Running benchmarks for ", kernels[i]);
    }
    results[i] = bench(kernels[i], n, verbose);
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


void runAllKernelBenchmarks(T = float)(bool verbose = true)
{
  auto kernels = tuple(DotProduct!(T)(),   Gaussian!(T)(1), Polynomial!(T)(2.5f, 1),
                            Exponential!(T)(1), Log!(T)(3),      Cauchy!(T)(1),
                            Power!(T)(2.5f),     Wave!(T)(1),     Sigmoid!(T)(1, 1));
  auto kernelNames = ["DotProduct",  "Gaussian", "Polynomial",
                       "Exponential", "Log",      "Cauchy",
                       "Power",       "Wave",     "Sigmoid"];
  //long[] n = [100L, 500L, 1000L];
  long[] n = [1000L, 5000L, 10_000L, 20_000L, 30_000L];
  auto results = runKernelBenchmark(kernels, n, verbose);

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
  auto file = File("../data/dNDSliceBench.csv", "w");
  foreach(row; table)
    file.writeRow(row);

  writeln("table: ", table);
}

void main()
{
  runAllKernelBenchmarks();
}
