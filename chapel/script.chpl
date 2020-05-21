use kernelmatrix;

use Random;
use Time;

proc bench(type T, Kernel, n: [?D] int(64), verbose: bool = true)
{
  var nitems: int(64) = D.dim(0).last: int(64);
  var times: [0..nitems] real(64);
  for i in 0..nitems {
    var _times: [1..3] real(64);
    var data: [1..n[i], 1..784] T;
    fillRandom(data);
    for j in 1..3 {
      var sw = new Timer();
      sw.start();
      var mat = calculateKernelMatrix(Kernel, data);
      sw.stop();
      _times[j] = (sw.elapsed(TimeUnits.microseconds)/1000_000): real(64);
    }
    times[i] = (_times[1] + _times[2] + _times[3])/3;
    if verbose {
      writeln("Average time for n = ", n[i], ", ", times[i], " seconds.");
      writeln("Detailed times: ", _times);
    }
  }
  return times;
}


/**
  To compile:
  chpl script.chpl kernelmatrix.chpl --fast && ./script
  bench: 0.040345 0.978326 3.85939 16.7087 40.2499
*/

proc main(){
  var K = new DotProduct();
  writeln("bench: ", bench(real(32), K, [1000, 5000, 
      10_000, 20_000, 30_000]));
  return;
}
