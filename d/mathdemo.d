import std.math: dlog = log;
import std.stdio: writeln;
import std.random: uniform01;
import core.stdc.math: logf, log, logl;
import std.datetime.stopwatch: AutoStart, StopWatch;

T log(T)(T x)
if(is(T == float))
{
  return logf(x);
}
T log(T)(T x)
if(is(T == double))
{
  return log(x);
}
T log(T)(T x)
if(is(T == real))
{
  return logl(x);
}

auto makeRandomArray(T)(size_t n)
{
  T[] arr = new T[n];
  foreach(ref el; arr)
  {
    el = uniform01!(T)();
  }
  return arr;
}

auto apply(alias fun, T)(T[] arr)
{
  foreach(ref el; arr)
  {
    el = fun(el);
  }
  return;
}

/**
  ldc2 -O --boundscheck=off --ffast-math --mcpu=native --boundscheck=off mathdemo.d && ./mathdemo
  Time taken for c log: 0.324789 seconds.
  Time taken for d log: 2.30737 seconds.
*/
void main()
{
  auto sw = StopWatch(AutoStart.no);

  /* For C's log function */
  auto arr = makeRandomArray!(float)(100_000_000);
  sw.start();
  apply!(log)(arr);
  sw.stop();
  writeln("Time taken for c log: ", sw.peek.total!"nsecs"/1000_000_000.0, " seconds.");
  sw.reset();
  
  /* For D's log function */
  arr = makeRandomArray!(float)(100_000_000);
  sw.start();
  apply!(dlog)(arr);
  sw.stop();
  writeln("Time taken for d log: ", sw.peek.total!"nsecs"/1000_000_000.0, " seconds.");
  sw.reset();
}
