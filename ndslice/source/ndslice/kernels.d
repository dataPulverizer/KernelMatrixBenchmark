module ndslice.kernels;
import core.stdc.tgmath: tanh;
import mir.algorithm.iteration;
import mir.math.common;
import mir.ndslice;

import std.parallelism;

/**
  Kernel Function Types:
*/
struct DotProduct(T)
{
  public:
  this(T _nothing)
  {}
  T opCall(Slice!(T*) x, Slice!(T*) y) const
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
    T opCall(Slice!(T*) x, Slice!(T*) y) const
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
    T opCall(Slice!(T*) x, Slice!(T*) y) const
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
    T opCall(Slice!(T*) x, Slice!(T*) y) const
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
    T opCall(Slice!(T*) x, Slice!(T*) y) const
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
    T opCall(Slice!(T*) x, Slice!(T*) y) const
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
    T opCall(Slice!(T*) x, Slice!(T*) y) const
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
    T opCall(Slice!(T*) x, Slice!(T*) y) const
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
    T opCall(Slice!(T*) x, Slice!(T*) y) const
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

auto calculateKernelMatrix(alias K, T)(K!(T) kernel, Slice!(T*, 2) data)
{
  size_t n = data.length!0;
  auto mat = uninitSlice!(T)(n, n);
  foreach(j, arrj; taskPool.parallel(data))
    foreach (i; j .. n)
      mat[j, i] = mat[i, j] = kernel(data[i], arrj);
  return mat;
}
