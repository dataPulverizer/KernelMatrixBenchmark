module ndslice.kernels;
import ndslice.math;

import mir.ndslice.slice;
import mir.ndslice.allocation;
import mir.ndslice.topology: iota;

import std.parallelism;
//import std.range : iota;
//import std.stdio: writeln;
//import std.datetime.stopwatch: AutoStart, StopWatch;

/**
  Kernel Function Types:
*/
struct DotProduct(T)
{
  public:
  this(T _nothing)
  {}
  T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
  {
    T dist = 0;
    auto m = x.length;
    for(size_t i = 0; i < m; ++i)
    {
      dist += x[i] * y[i];
    }
    return dist;
  }
}

struct Gaussian(T)
{
  private:
    T theta;
  public:
    this(T _theta)
    {
      theta = _theta;
    }
    T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        auto tmp = x[i] - y[i];
        dist += tmp * tmp;
      }
      return exp(-sqrt(dist)/theta);
    }
}

struct Polynomial(T)
{
  private:
    T d;
    T offset;
  public:
    this(T _d, T _offset)
    {
      d = _d;
      offset = _offset;
    }
    T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist += x[i] * y[i];
      }
      return pow(dist + offset, d);
    }
}

struct Exponential(T)
{
  private:
    T theta;
  public:
    this(T _theta)
    {
      theta = _theta;
    }
    T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist -= abs(x[i] - y[i]);
      }
      return exp(dist/theta);
    }
}

struct Log(T)
{
  private:
    T beta;
  public:
    this(T _beta)
    {
      beta = _beta;
    }
    T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist += pow(abs(x[i] - y[i]), beta);
      }
      dist = pow(dist, 1/beta);
      return -log(1 + dist);
    }
}

struct Cauchy(T)
{
  private:
    T theta;
  public:
    this(T _theta)
    {
      theta = _theta;
    }
    T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        auto tmp = x[i] - y[i];
        dist += tmp * tmp;
      }
      dist = sqrt(dist)/theta;
      return 1/(1 + dist);
    }
}

struct Power(T)
{
  private:
    T beta;
  public:
    this(T _beta)
    {
      beta = _beta;
    }
    T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist += pow(abs(x[i] - y[i]), beta);
      }
      return -pow(dist, 1/beta);
    }
}

struct Wave(T)
{
  private:
    T theta;
  public:
    this(T _theta)
    {
      theta = _theta;
    }
    T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist += abs(x[i] - y[i]);
      }
      auto tmp = theta/dist;
      return tmp*sin(1/tmp);
    }
}

struct Sigmoid(T)
{
  private:
    T beta0;
    T beta1;
  public:
    this(T _beta0, T _beta1)
    {
      beta0 = _beta0;
      beta1 = _beta1;
    }
    T opCall(U...)(Slice!(T*, U) x, Slice!(T*, U) y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist += x[i] * y[i];
      }
      return tanh(beta0 * dist + beta1);
    }
}

/************************************************************************************/

auto calculateKernelMatrix(alias K, T, U...)(K!(T) kernel, Slice!(T*, U) data)
{
  size_t n = data.length!0;
  auto mat = slice!(T)(n, n);

  foreach(j; taskPool.parallel(iota(n)))
  {
    auto arrj = data[j, 0..$];
    foreach(size_t i; j..n)
    {
      mat[i, j] = kernel(data[i, 0..$], arrj);
      mat[j, i] = mat[i, j];
    }
  }
  return mat;
}
