use CPtr;
use Math;

record DotProduct {
  type T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      dist += xrow[i] * yrow[i];
    }
    return dist;
  }
}

record Gaussian {
  type T;
  const theta: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      var tmp = xrow[i] - yrow[i];
      dist += tmp * tmp;
    }
    return exp(-sqrt(dist)/this.theta);
  }
}

record Polynomial {
  type T;
  const d: T;
  const offset: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      dist += xrow[i]*yrow[i];
    }
    return (dist + this.offset)**this.d;
  }
}

record Exponential {
  type T;
  const theta: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      dist -= abs(xrow[i] - yrow[i]);
    }
    return exp(dist/this.theta);
  }
}

record Log {
  type T;
  const beta: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      dist += abs(xrow[i] - yrow[i])**this.beta;
    }
    dist = dist**(1/this.beta);
    return -log(1 + dist);
  }
}

record Cauchy {
  type T;
  const theta: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      var tmp = xrow[i] - yrow[i];
      dist += tmp * tmp;
    }
    dist = sqrt(dist)/this.theta;
    return 1/(1 + dist);
  }
}

record Power {
  type T;
  const beta: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      dist += abs(xrow[i] - yrow[i])**this.beta;
    }
    return -dist**(1/this.beta);
  }
}

record Wave {
  type T;
  const theta: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      dist += abs(xrow[i] - yrow[i]);
    }
    var tmp = this.theta/dist;
    return tmp*sin(1/tmp);
  }
}

record Sigmoid {
  type T;
  const beta0: T;
  const beta1: T;
  proc kernel(xrow:c_ptr, yrow:c_ptr(?T), p: int): T
  {
    var dist: T = 0: T;
    for i in 0..#p {
      dist += xrow[i] * yrow[i];
    }
    return tanh(this.beta0 * dist + this.beta1);
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

