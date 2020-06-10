use kernel;

use IO;
use Time;
use Random;

config const fastmath: bool = false;
config const verbose: bool = true;
const folder = if fastmath then "fmdata" else "data";

record BenchRecord {
  var D: domain(1);
  var n: [D] int(64);
  var times: [D] real(64);
  proc init()
  {
    this.D = {0..1};
    this.n = [0, 1];
    this.times = [0.0, 1.0];
  }
}

proc bench(type T, Kernel, n: [?D] int(64))
{
  var nitems: int(64) = D.dim(0).last: int(64);
  var times: [0..nitems] real(64);

  var result: BenchRecord;
  result.D = {0..nitems};

  for i in 0..nitems {
    var _times: [0..2] real(64);
    var data: [1..n[i], 1..784] T;
    fillRandom(data);
    for j in 0..2 {
      var sw = new Timer();
      sw.start();
      var mat = calculateKernelMatrix(Kernel, data);
      sw.stop();
      _times[j] = (sw.elapsed(TimeUnits.microseconds)/1000_000): real(64);
    }
    times[i] = (_times[0] + _times[1] + _times[2])/3;
    if verbose {
      writeln("Average time for n = ", n[i], ", ", times[i], " seconds.");
      writeln("Detailed times: ", _times);
    }
  }
  result.n = n;
  result.times = times;
  return result;
}

proc runKernelBenchmarks(type T, kernels, n: [?D] int(64))
{
  var results: [0..#kernels.size] BenchRecord;
  for param i in 0..(kernels.size - 1) {
    const kernel = kernels(i);
    if verbose {
      writeln("\n\nRunning benchmarks for ", kernel.type: string, kernel: string);
    }
    results[i] = bench(T, kernel, n);
  }
  return results;
}

/**
  To compile:
  chpl script.chpl kernel.chpl --fast && ./script
*/
proc runAllKernelBenchmarks(type T, folder: string)
{
  //var n = [100, 500, 1000];
  var n = [1000, 5000, 10000, 20000, 30000];
  
  var kernels = (new DotProduct(T),        new Gaussian(T, 1: T), new Polynomial(T, 2.5: T, 1: T),
                 new Exponential(T, 1: T), new Log(T, 3: T),      new Cauchy(T, 1: T),
                 new Power(T, 2.5: T),     new Wave(T, 1: T),     new Sigmoid(T, 1: T, 1: T));
  var kernelNames = ["DotProduct",  "Gaussian", "Polynomial",
                       "Exponential", "Log",      "Cauchy",
                       "Power",       "Wave",     "Sigmoid"];
  var results = runKernelBenchmarks(T, kernels, n);
  

  var last: int(64) = n.domain.dim(0).last: int(64);
  var tabLen = n.size * kernels.size;
  var table: [0..tabLen, 0..3] string;
  table[0, ..] = ["language,", "kernel,", "nitems,", "time"];
  while (true)
  {
    var k = 1;
    for i in 0..#kernels.size {
      var tmp = ["Chapel,", kernelNames[i] + ",", "", ""];
      for j in 0..(n.size - 1) {
        tmp[2] = results[i].n[j]: string + ",";
        tmp[3] = results[i].times[j]: string;
        table[k, ..] = tmp;
        k += 1;
      }
    }
    if k > tabLen
    {
      break;
    }
  }
  var file = open("../" + folder + "/chapelBench.csv", iomode.cw);
  var _channel = file.writer();
  _channel.write(table);
  _channel.close();
  file.close();
  return;
}

proc main()
{
  writeln("folder: ", folder);
  runAllKernelBenchmarks(real(32), folder);
}
