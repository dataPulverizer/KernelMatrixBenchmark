import arrays;
import std.parallelism;
import std.range : iota;
import std.stdio: writeln;
import std.datetime.stopwatch: AutoStart, StopWatch;

import core.stdc.math: exp,   exp  = expf,   exp = expl,
                       fabs,  fabs = fabsf,  fabs = fabsl,
                       log,   log  = logf,   log  = logl, 
                       pow,   pow  = powf,   pow  = powl,
                       sin,   sin  = sinf,   sin  = sinhl,
                       sqrt,  sqrt = sqrtf,  sqrt = sqrtl,
                       tanh,  tanh = tanhf,  tanh = tanhl;

/**
  Kernel Function Types:
*/
struct DotProduct(T)
{
  public:
  this(T _nothing)
  {}
  T opCall(T[] x, T[] y) const
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
    T opCall(T[] x, T[] y) const
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
    T opCall(T[] x, T[] y) const
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
    T opCall(T[] x, T[] y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist -= fabs(x[i] - y[i]);
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
    T opCall(T[] x, T[] y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist += pow(fabs(x[i] - y[i]), beta);
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
    T opCall(T[] x, T[] y) const
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
    T opCall(T[] x, T[] y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist += pow(fabs(x[i] - y[i]), beta);
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
    T opCall(T[] x, T[] y) const
    {
      T dist = 0;
      auto m = x.length;
      for(size_t i = 0; i < m; ++i)
      {
        dist += fabs(x[i] - y[i]);
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
    T opCall(T[] x, T[] y) const
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

auto calculateKernelMatrix(alias K, T)(K!(T) kernel, Matrix!(T) data)
{
  size_t n = data.ncol;
  auto mat = Matrix!(T)(n, n);

  foreach(j; taskPool.parallel(iota(n)))
  {
    auto arrj = data.refColumnSelect(j).array;
    foreach(size_t i; j..n)
    {
      mat[i, j] = kernel(data.refColumnSelect(i).array, arrj);
      mat[j, i] = mat[i, j];
    }
  }
  return mat;
}
