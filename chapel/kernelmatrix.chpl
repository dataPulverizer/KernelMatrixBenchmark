use CPtr;

record DotProduct {
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var ret: T = 0: T;
    for i in 0..#p {
      ret += xrow[i] * yrow[i];
    }
    return ret;
  }
}

record Gaussian {
  type T;
  const gamma: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var ret: T = 0: T;
    for i in 0..#p {
      var tmp = xrow[i] - yrow[i];
      ret -= tmp * tmp;
    }
    return exp(this.gamma * ret);
  }
}

record Polynomial {
  type T;
  const d: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var ret: T = 0: T;
    for i in 0..#p {
      ret += xrow[i]*yrow[i];
    }
    return (ret + 1)**this.d;
  }
}

/***************************************************************************/
use DynamicIters;
proc calculateKernelMatrix(K, data: [?D] ?T) /* : [?E] T */
{
  var n = D.dim(0).last;
  var p = D.dim(1).last;
  var E: domain(2) = {D.dim(0), D.dim(0)};
  var mat: [E] T;
  // code below assumes data starts at 1,1
  var rowPointers: [1..n] c_ptr(T) =
    forall i in 1..n do c_ptrTo(data[i, 1]);

  forall j in guided(1..n by -1) {
    for i in j..n {
      mat[i, j] = K.kernel(rowPointers[i], rowPointers[j], p);
      mat[j, i] = mat[i, j];
    }
  }
  return mat;
}

