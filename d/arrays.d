/*
  This module contains implementations for vectors and matrices an altered version from my glmsolverd package
*/

module arrays;

import std.conv: to;
import std.format: format;
import std.traits: isFloatingPoint, isIntegral, isNumeric;
import std.algorithm: min, max;
import std.math: modf;
import core.memory: GC;
import std.stdio: writeln;
import std.random;

/********************************************* Printer Utility Functions *********************************************/
auto getRange(T)(const(T[]) data)
if(isFloatingPoint!T)
{
  real[2] range = [cast(real)data[0], cast(real)data[0]];
  foreach(el; data)
  {
    range[0] = min(range[0], el);
    range[1] = max(range[1], el);
  }
  return range;
}
string getFormat(real[] range, long maxLength = 8, long gap = 2)
{
  writeln("range: ", range);
  string form = "";
  if((range[0] > 0.01) & (range[1] < 1000_000))
  {
    form = "%" ~ to!(string)(gap + 2 + maxLength) ~ "." ~ to!(string)(maxLength) ~ "g";
  }else if((range[0] < 0.0001) | (range[1] > 1000_000))
  {
    form = "%" ~ to!(string)(gap + 1 + maxLength) ~ "." ~ to!(string)(4) ~ "g";
  }
  return form;
}
/********************************************* Matrix Class *********************************************/

/*
  Faster Array Creation
*/
auto newArray(T)(long n)
{
  auto data = (cast(T*)GC.malloc(T.sizeof*n, GC.BlkAttr.NO_SCAN))[0..n];
    if(data == null)
      assert(0, "Array Allocation Failed!");
  return data;
}

/*
  Matrix will be column major
*/
mixin template MatrixGubbings(T)
{
  private:
  T[] data;
  long[] dim;
  
  public:
  this(T[] _data, long rows, long cols)
  {
    assert(rows*cols == _data.length, 
          "dimension of matrix inconsistent with length of array");
    data = _data; dim = [rows, cols];
  }
  this(long n, long m)
  {
    long _len = n*m;
    data = newArray!(T)(_len);
    dim = [n, m];
  }
  this(T[] _data, long[] _dim)
  {
    long tlen = _dim[0]*_dim[1];
    assert(tlen == _data.length, 
          "dimension of matrix inconsistent with length of array");
    data = _data; dim = _dim;
  }
  this(Matrix!(T) mat)
  {
    data = mat.data.dup;
    dim = mat.dim.dup;
  }
  @property Matrix!(T) dup() const
  {
    return Matrix!(T)(data.dup, dim.dup);
  }
  T opIndex(long i, long j) const
  {
    return data[dim[0]*j + i];
  }
  void opIndexAssign(T x, long i, long j)
  {
    data[dim[0]*j + i] = x;
  }
  T opIndexOpAssign(string op)(T x, long i, long j)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
      mixin("return data[dim[0]*j + i] " ~ op ~ "= x;");
    else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
  }
  Matrix!(T) opBinary(string op)(Matrix!(T) x)
  {
    assert( data.length == x.array.length,
          "Number of rows and columns in matrices not equal.");
    long n = data.length;
    auto ret = Matrix!(T)(dim[0], dim[1]);
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      for(long i = 0; i < n; ++i)
      {
        mixin("ret.array[i] = " ~ "data[i] " ~ op ~ " x.array[i];");
      }
    }else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
    return ret;
  }
  Matrix!(T) opBinary(string op)(T rhs)
  {
    ulong n = data.length;
    Matrix!(T) ret = Matrix!(T)(dim[0], dim[1]);
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      for(ulong i = 0; i < n; ++i)
      {
        mixin("ret.array[i] = " ~ "data[i] " ~ op ~ " rhs;");
      }
    }else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
    return ret;
  }
  Matrix!(T) opBinaryRight(string op)(T lhs)
  {
    long n = data.length;
    Matrix!(T) ret = Matrix!(T)(dim[0], dim[1]);
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      for(long i = 0; i < n; ++i)
      {
        mixin("ret.array[i] = " ~ "lhs " ~ op ~ " data[i];");
      }
    }else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
    return ret;
  }
  void opOpAssign(string op)(Matrix!(T) x)
  {
    assert( data.length == x.array.length,
          "Number of rows and columns in matrices not equal.");
    long n = data.length;
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      for(long i = 0; i < n; ++i)
      {
        mixin("data[i] " ~ op ~ "= x.array[i];");
      }
    }else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
  }
  /* mat "op"= rhs */
  void opOpAssign(string op)(T rhs)
  {
    long n = data.length;
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      for(long i = 0; i < n; ++i)
      {
        mixin("data[i] " ~ op ~ "= rhs;");
      }
    }else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
  }
  @property long nrow() const
  {
    return dim[0];
  }
  @property long ncol() const
  {
    return dim[1];
  }
  @property T[] array()
  {
    return data;
  }
  @property long len() const
  {
    return data.length;
  }
  @property long length() const
  {
    return data.length;
  }
  @property size() const
  {
    return dim.dup;
  }
  /* Returns transposed matrix (duplicated) */
  Matrix!(T) t() const
  {
    auto _data = data.dup;
    long[] _dim = new long[2];
    _dim[0] = dim[1]; _dim[1] = dim[0];
    if((dim[0] == 1) & (dim[1] == 1)){
    } else if(dim[0] != dim[1]) {
      for(long j = 0; j < dim[1]; ++j)
      {
        for(long i = 0; i < dim[0]; ++i)
        {
          _data[_dim[0]*i + j] = data[dim[0]*j + i];
        }
      }
    } else  if(dim[0] == dim[1]) {
      for(long j = 0; j < dim[1]; ++j)
      {
        for(long i = 0; i < dim[0]; ++i)
        {
          if(i == j)
            continue;
          _data[_dim[0]*i + j] = data[dim[0]*j + i];
        }
      }
    }
    return Matrix!(T)(_data, _dim);
  }
  
  /* Appends Vector to the END of the matrix */
  void appendColumn(T[] rhs)
  {
    assert(rhs.length == nrow,
      "Vector is not of the same length as number of rows.");
    data ~= rhs;
    dim[1] += 1;
    return;
  }
  void appendColumn(Matrix!(T) rhs)
  {
    assert((rhs.nrow == 1) | (rhs.ncol == 1), 
      "Matrix does not have 1 row or 1 column");
    appendColumn(rhs.array);
  }
  void appendColumn(T _rhs)
  {
    auto rhs = newArray!(T)(nrow);
    rhs[] = _rhs;
    appendColumn(rhs);
  }
  /* Prepends Column Vector to the START of the matrix */
  void prependColumn(T[] rhs)
  {
    assert(rhs.length == nrow,
      "Vector is not of the same length as  number of rows.");
    data = rhs ~ data;
    dim[1] += 1;
    return;
  }
  void prependColumn(Matrix!(T) rhs)
  {
    assert((rhs.nrow == 1) | (rhs.ncol == 1), 
      "Matrix does not have 1 row or 1 column");
    prependColumn(rhs.array);
  }
  void prependColumn(T _rhs)
  {
    auto rhs = newArray!(T)(nrow);
    rhs[] = _rhs;
    prependColumn(rhs);
  }
  /* Contiguous column select copies the column */
  auto columnSelect(long start, long end)
  {
    assert(end > start, "Starting column is not less than end column");
    long nCol = end - start;
    long _len = nrow * nCol;
    auto arr = newArray!(T)(_len);
    auto startIndex = start*nrow;
    long iStart = 0;
    for(long i = 0; i < nCol; ++i)
    {
      arr[iStart..((iStart + nrow))] = data[startIndex..(startIndex + nrow)];
      startIndex += nrow;
      iStart += nrow;
    }
    return Matrix!(T)(arr, [nrow, nCol]);
  }
  auto columnSelect(long index)
  {
    assert(index < ncol, "Selected index is not less than number of columns.");
    auto arr = newArray!(T)(nrow);
    auto startIndex = index*nrow;
    arr[] = data[startIndex..(startIndex + nrow)];
    return Matrix!(T)(arr, [nrow, 1]);
  }
  auto refColumnSelect(long index)
  {
    assert(index < ncol, "Selected index is not less than number of columns.");
    auto startIndex = index*nrow;
    return Matrix!(T)(data[startIndex..(startIndex + nrow)], [nrow, 1]);
  }
  auto refColumnSelectArr(long index)
  {
    //assert(index < ncol, "Selected index is not less than number of columns.");
    auto startIndex = index*nrow;
    return data[startIndex..(startIndex + nrow)];
  }
  /*
    Function to remove a column from the matrix.
  */
  Matrix!(T) refColumnRemove(long index)
  {
    /* Remove first column */
    if(index == 0)
    {
      data = data[nrow..$];
      dim[1] -= 1;
      return this;
    /* Remove last column */
    }else if(index == (ncol - 1))
    {
      data = data[0..($ - nrow)];
      dim[1]-= 1;
      return this;
    /* Remove any other column */
    }else{
      auto start = index*nrow;
      long _len = data.length - nrow;
      auto _data = newArray!(T)(_len);
      _data[0..start] = data[0..start];
      _data[start..$] = data[(start + nrow)..$];
      data = _data;
      dim[1] -= 1;
      return this;
    }
  }
  /* Assigns vector in-place to a specific column */
  /* Refactor these two methods */
  Matrix!(T) refColumnAssign(T[] col, long index)
  {
    assert(col.length == nrow, "Length of vector is not the same as number of rows");
    /* Replace first column */
    if(index == 0)
    {
      data[0..nrow] = col;
      return this;
    /* Replace last column */
    }else if(index == (ncol - 1))
    {
      data[($ - nrow)..$] = col;
      return this;
    /* Replace any other column */
    }else{
      auto start = index*nrow;
      data[start..(start + nrow)] = col;
      return this;
    }
  }
  Matrix!(T) refColumnAssign(T col, long index)
  {
    /* Replace first column */
    if(index == 0)
    {
      data[0..nrow] = col;
      return this;
    /* Replace last column */
    }else if(index == (ncol - 1))
    {
      data[($ - nrow)..$] = col;
      return this;
    /* Replace any other column */
    }else{
      auto start = index*nrow;
      data[start..(start + nrow)] = col;
      return this;
    }
  }
}
/* Assuming column major */
struct Matrix(T)
if(isFloatingPoint!T)
{
  mixin MatrixGubbings!(T);
  string toString() const
  {
    string dform = getFormat(getRange(data));
    string repr = format(" Matrix(%d x %d)\n", dim[0], dim[1]);
    for(long i = 0; i < dim[0]; ++i)
    {
      for(long j = 0; j < dim[1]; ++j)
      {
        repr ~= format(dform, opIndex(i, j));
      }
      repr ~= "\n";
    }
    return repr;
  }
}

// Create random matrix
/****************************************************************************/
Matrix!T createRandomMatrix(T)(ulong rows, ulong cols)
{
  Mt19937_64 gen;
  gen.seed(unpredictableSeed);
  ulong len = rows*cols;
  T[] data = newArray!(T)(len);
  for(int i = 0; i < len; ++i)
    data[i] = uniform01!(T)(gen);
  return Matrix!T(data, rows, cols);
}
Matrix!T createRandomMatrix(T)(ulong m)
{
  return createRandomMatrix!(T)(m, m);
}
Matrix!T createRandomMatrix(T)(ulong[] dim)
{
  return createRandomMatrix!(T)(dim[0], dim[1]);
}

