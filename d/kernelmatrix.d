import arrays;
import std.parallelism;
import std.range : iota;
import std.math: exp, sqrt, tanh;
import std.stdio: writeln;
import std.datetime.stopwatch: AutoStart, StopWatch;

/*
  Reference for Kernel functions
  http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/
*/
interface AbstractKernel(T){
  T opCall(T[] x, T[] y) const;
}

class DotProduct(T): AbstractKernel!(T)
{
  @nogc:
  public:
  //pragma(inline, true):
  //@fastmath
  T opCall(T[] x, T[] y) const
  {
    //assert(x.length == y.length, "x and y are not the same length.");
    T ret = 0;
    foreach(long i; 0..x.length)
    {
      ret += x[i]*y[i];
    }
    return ret;
  }
}

class Polynomial(T): AbstractKernel!(T)
{
  @nogc:
  private:
  T d;
  public:
  T opCall(T[] x, T[] y) const
  {
    //assert(x.length == y.length, "x and y are not the same length.");
    T ret = 0;
    for(long i = 0; i < x.length; ++i)
    {
      ret += x[i]*y[i];
    }
    return (ret + 1)^^d;
  }
}

class Gaussian(T): AbstractKernel!(T)
{
  @nogc:
  private:
  T gamma;
  public:
  this(T _gamma)
  {
    gamma = _gamma;
  }
  T opCall(T[] x, T[] y) const
  {
    //assert(x.length == y.length, "x and y are not the same length.");
    T ret = 0;
    for(long i = 0; i < x.length; ++i)
    {
      auto tmp = x[i] - y[i];
      ret += tmp*tmp;
    }
    return exp(-ret*gamma);
  }
}


class Laplace(T): AbstractKernel!(T)
{
  @nogc:
  private:
  T sigma;
  public:
  this(T _sigma)
  {
    sigma = _sigma;
  }
  T opCall(T[] x, T[] y) const
  {
    //assert(x.length == y.length, "x and y are not the same length.");
    T ret = 0;
    for(long i = 0; i < x.length; ++i)
    {
      auto tmp = x[i] - y[i];
      ret += tmp*tmp;
    }
    ret = sqrt(ret);
    return exp(-ret/sigma);
  }
}


class HyperbolicTan(T): AbstractKernel!(T)
{
  @nogc:
  private:
  T alpha;
  T c;
  public:
  this(_alpha, _c)
  {
    alpha =  _alpha; c = _c;
  }
  T opCall(T[] x, T[] y) const
  {
    //assert(x.length == y.length, "x and y are not the same length.");
    T ret = 0;
    for(long i = 0; i < x.length; ++i)
    {
      ret += x[i]*y[i];
    }
    ret *= alpha;
    ret += c;
    return tanh(ret);
  }
}

auto calculateKernelMatrix(K, T)(K!(T) kernel, Matrix!(T) data)
{
  long n = data.ncol;
  auto mat = Matrix!(T)(n, n);

  foreach(j; taskPool.parallel(iota(n)))
  {
    auto arrj = data.refColumnSelect(j).array;
    //for(long i = j; i < n; ++i)
    foreach(long i; j..n)
    {
      mat[i, j] = kernel(data.refColumnSelect(i).array, arrj);
      mat[j, i] = mat[i, j];
    }
  }
  return mat;
}



