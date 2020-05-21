/*
  This module contains implementations for vectors and matrices
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
long getMaxLength(T)(const(T[]) v)
if(isIntegral!T)
{
  long x = 0;
  foreach(el; v)
    x = max(x, cast(long)to!string(el).length);
  return x;
}

long[] getMaxLength(T)(const(T[]) v)
if(isFloatingPoint!T)
{
  long[] x = [0, 0];
  real intpart, frac;
  foreach(el; v)
  {
    frac = modf(cast(real)el, intpart);
    x[0] = max(x[0], cast(long)to!string(intpart).length);
    x[1] = max(x[1], cast(long)to!string(frac).length);
  }
  return x;
}

string floatFormat(long[] mlen, long dp = 6, long dig = 7, long gap = 2)
{
  string dform = "";
  long tot = mlen[0] + mlen[1];

  if((tot > dig) && (mlen[0] > 1))
  {
    dform = "%" ~ to!string(dp + 4*gap) ~ "." ~ to!string(dp) ~ "e";
  } else if(tot > dig){
    dform = "%" ~ to!string(dp + 2*gap) ~ "." ~ to!string(dp) ~ "f";
  } else {
    dform = "%" ~ to!string(mlen[0] + mlen[1] + gap) ~ "." ~ to!string(mlen[1]) ~ "f";
  }

  return dform;
}

/* Function to pad to length */
string pad(string num, long len)
{
  while(len > num.length)
    num ~= " ";
  return num;
}


/********************************************* Matrix Class *********************************************/

/*
  Create an array in an "unsafe" but blazingly fast way.
*/
auto newArray(T)(long n)
{
  auto data = (cast(T*)GC.malloc(T.sizeof*n, GC.BlkAttr.NO_SCAN))[0..n];
    if(data == null)
      assert(0, "Array Allocation Failed!");
  return data;
}

/*
  Matrices and Vectors will be column major
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
    //data = new T[_len];
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
  @property Matrix!(T) dup()
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
  @property long nrow()
  {
    return dim[0];
  }
  @property long ncol()
  {
    return dim[1];
  }
  @property T[] array()
  {
    return data;
  }
  @property long len()
  {
    return data.length;
  }
  @property long length()
  {
    return data.length;
  }
  @property size() const
  {
    return dim.dup;
  }
  /* Returns transposed matrix (duplicated) */
  Matrix!(T) t()
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
  /* Cast to Column Vector */
  ColumnVector!(T) opCast(V: ColumnVector!(T))() {
    assert(ncol == 1, "The number of columns in the matrix 
         is not == 1 and so can not be converted to a matrix.");
    return new ColumnVector!(T)(data);
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
  void appendColumn(Vector!(T) rhs)
  {
    appendColumn(rhs.array);
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
    //auto rhs = new T[nrow];
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
  void prependColumn(Vector!(T) rhs)
  {
    prependColumn(rhs.array);
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
    //auto rhs = new T[nrow];
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
    //auto arr = new T[_len];
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
      //T[] _data = new T[_len];
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
      //T[] newdata = new T[data.length - nrow];
      data[start..(start + nrow)] = col;
      return this;
    }
  }
  Matrix!(T) refColumnAssign(Vector!(T) col, long index)
  {
    return refColumnAssign(col.array, index);
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
      //T[] newdata = new T[data.length - nrow];
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
    long[] mlen = getMaxLength!T(data);
    string dform = floatFormat(mlen);
    writeln(dform);
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
  string printSubArray(long[] rows, long[] cols) const
  {
    string repr = format(" Matrix(%d x %d)\n", rows[1] - rows[0], cols[1] - cols[0]);
    string[] reprArr = new string[(rows[1] - rows[0])*(cols[1] - cols[0])];
    long k = 0; long _max = 0;
    /* Get the maximum string length */
    for(long i = rows[0]; i < rows[1]; ++i)
    {
      for(long j = cols[0]; j < cols[1]; ++j)
      {
        reprArr[k] = to!(string)(opIndex(i, j));
        _max = max(cast(long)reprArr[k].length, _max);
        k += 1;
      }
    }
    k = 0;
    for(long i = rows[0]; i < rows[1]; ++i)
    {
      for(long j = cols[0]; j < cols[1]; ++j)
      {
        repr ~= pad(reprArr[k], _max) ~ "  ";
        k += 1;
      }
      repr ~= "\n";
    }
    return repr;
  }
}

struct Matrix(T)
if(isIntegral!T)
{
  mixin MatrixGubbings!(T);
  string toString() const
  {
    long dig = 6;
    long mlen = getMaxLength!T(data);
    long gap = 2;
    dig = mlen < dig ? mlen : dig;
    string dform = "%" ~ to!string(dig + gap) ~ "d";
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
/********************************************* Vector Classes *********************************************/
interface Vector(T)
{
  @property long len() const;
  @property long length() const;
  T opIndex(long i) const;
  void opIndexAssign(T x, long i);
  void opIndexOpAssign(string op)(T x, long i);
  @property T[] array();
  Matrix!(T) opCast(M: Matrix!(T))();
}
mixin template VectorGubbings(T)
{
  T[] data;
  @property long len() const
  {
    return data.length;
  }
  @property long length() const
  {
    return data.length;
  }
  this(T)(T[] dat)
  {
    data = dat;
  }
  this(long n)
  {
    auto data = newArray!(T)(n);
  }
  T opIndex(long i) const
  {
    return data[i];
  }
  void opIndexAssign(T x, long i)
  {
    data[i] = x;
  }
  T opIndexOpAssign(string op)(T x, long i)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
      mixin("return data[i] " ~ op ~ "= x;");
    else static assert(0, "Operator "~ op ~" not implemented");
  }
  @property T[] array()
  {
    return data;
  }
  Matrix!(T) opCast(M: Matrix!(T))()
  {
    return Matrix!(T)(data, [len, 1]);
  }
}
class ColumnVector(T) : Vector!T
if(isNumeric!T)
{
  mixin VectorGubbings!(T);

  override string toString() const
  {
    auto n = len();
    string repr = format("ColumnVector(%d)", n) ~ "\n";
    for(long i = 0; i < n; ++i)
    {
      repr ~= to!string(data[i]) ~ "\n";
    }
    return repr;
  }
  ColumnVector!T opBinary(string op)(T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      auto ret = data.dup;
      for(long i = 0; i < data.length; ++i)
        mixin("ret[i] = ret[i] " ~ op ~ " rhs;");
      return new ColumnVector!(T)(ret);
    } else static assert(0, "Operator " ~ op ~ " not implemented");
  }
  ColumnVector!T opBinaryRight(string op)(T lhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      auto ret = data.dup;
      for(long i = 0; i < data.length; ++i)
        mixin("ret[i] = lhs " ~ op ~ " ret[i];");
      return new ColumnVector!(T)(ret);
    } else static assert(0, "Operator " ~ op ~ " not implemented");
  }
  ColumnVector!T opBinary(string op)(ColumnVector!T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      assert(data.length == rhs.data.length, "Vector lengths are not the same.");
      auto ret = new ColumnVector!T(rhs.data.dup);
      for(long i = 0; i < data.length; ++i)
        mixin("ret.data[i] = data[i] " ~ op ~ " ret.data[i];");
      return ret;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  void opOpAssign(string op)(ColumnVector!T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      assert(data.length == rhs.data.length, "Vector lengths are not the same.");
      for(long i = 0; i < data.length; ++i)
        mixin("data[i] = data[i] " ~ op ~ " rhs.data[i];");
      return;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  void opOpAssign(string op)(T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      for(long i = 0; i < data.length; ++i)
        mixin("data[i] = data[i] " ~ op ~ " rhs;");
      return;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  @property ColumnVector!(T) dup()
  {
    return new ColumnVector!T(data.dup);
  }
}
class RowVector(T): Vector!T
if(isNumeric!T)
{
  mixin VectorGubbings!(T);

  override string toString() const
  {
    long dig = 5;
    string repr = format("RowVector(%d)", len()) ~ "\n" ~ to!string(data) ~ "\n";
    return repr;
  }
  RowVector!T opBinary(string op)(T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      auto ret = data.dup;
      for(long i = 0; i < data.length; ++i)
        mixin("ret[i] = ret[i] " ~ op ~ " rhs;");
      return new RowVector!(T)(ret);
    } else static assert(0, "Operator " ~ op ~ " not implemented");
  }
  RowVector!T opBinaryRight(string op)(T lhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      auto ret = data.dup;
      for(long i = 0; i < data.length; ++i)
        mixin("ret[i] = lhs " ~ op ~ " ret[i];");
      return new RowVector!(T)(ret);
    } else static assert(0, "Operator " ~ op ~ " not implemented");
  }
  RowVector!T opBinary(string op)(RowVector!T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      assert(data.len == rhs.data.len, "Vector lengths are not the same.");
      auto ret = RowVector!T(rhs.data.dup);
      for(long i = 0; i < data.len; ++i)
        mixin("ret.data[i] = data[i] "~ op ~ " ret.data[i];");
      return ret;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  void opOpAssign(string op)(RowVector!T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      assert(data.len == rhs.data.len, "Vector lengths are not the same.");
      for(long i = 0; i < data.len; ++i)
        mixin("data[i] = data[i] "~ op ~ " rhs.data[i]");
      return;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  void opOpAssign(string op)(T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
    {
      for(long i = 0; i < data.len; ++i)
        mixin("data[i] = data[i] "~ op ~ " rhs");
      return;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  @property RowVector!(T) dup()
  {
    return new RowVector!T(data.dup);
  }
}

// Convenience functions for matrices
/****************************************************************************/
/* Convinient function for constructor with type inference */
Matrix!(T) matrix(T)(T[] data, long rows, long cols)
{
  assert(rows*cols == data.length, 
        "dimension of matrix inconsistent with length of array");
  return Matrix!(T)(data, [rows, cols]);
}
/* Constructor for matrix with data and dimension array */
Matrix!(T) matrix(T)(T[] data, long[] dim)
{
  assert(dim[0]*dim[1] == data.length, 
        "dimension of matrix inconsistent with length of array");
  return Matrix!(T)(data, dim);
}
Matrix!(T) matrix(T)(T[] dat, long rows)
{
  assert(rows * rows == dat.length, 
        "dimension of matrix inconsistent with length of array");
  return Matrix!(T)(dat, [rows, rows]);
}


Matrix!(T) fillMatrix(T)(T x, long[] dim)
{
  long n = dim[0] * dim[1];
  auto data = newArray!(T)(n);
  data[] = x;
  return Matrix!(T)(data, dim.dup);
}
Matrix!(T) fillMatrix(T)(T x, long nrow, long ncol)
{
  return fillMatrix!(T)(x, [nrow, ncol]);
}

Matrix!(T) matrix(T)(long rows, long cols)
{
  long n = rows * cols;
  auto data = newArray!(T)(n);
  return Matrix!(T)(data, [rows, cols]);
}
Matrix!(T) matrix(T)(long[] dim)
{
  return matrix(dim[0], dim[1]);
}

/* Constructor for square matrix */
Matrix!(T) matrix(T)(Matrix!(T) m)
{
  return Matrix!(T)(m);
}


Matrix!(T) zerosMatrix(T)(long[] dim)
{
  return fillMatrix!(T)(cast(T)0, dim[0], dim[1]);
}
Matrix!(T) zerosMatrix(T)(long nrow, long ncol)
{
  return fillMatrix!(T)(cast(T)0, nrow, ncol);
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

// Convenience functions for vectors
/****************************************************************************/
ColumnVector!(T) fillColumn(T)(T x, long n)
{
  if(n > 0)
  {
    auto data = newArray!(T)(n);
    data[] = x;
    return new ColumnVector!(T)(data);
  }
  return new ColumnVector!(T)(new T[0]);
}
ColumnVector!(T) zerosColumn(T)(long n)
{
  return fillColumn!(T)(cast(T)0, n);
}
auto onesColumn(T)(long n)
{
  return fillColumn!(T)(cast(T)1, n);
}

RowVector!(T) rowVector(T)(T[] data)
{
  return new RowVector!(T)(data);
}
