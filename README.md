# A look at Chapel, D, and Julia using kernel matrix calculations

## Introduction

It seems each time you turn around there is a new programming language aimed at solving some specific problem set. Increased proliferation of programming languages and data are deeply connected in a fundamental way and increasing demand for “data science” computing is a related phenomenon. In the field of scientific computing Chapel, D, and Julia  are highly relevant programming languages. They arise from different needs and are aimed at different problem sets. Chapel focuses on data parallelism on single multicore machines and large clusters, D initially was developed as a more productive safer alternative to C++ and Julia was developed for technical and scientific computing and aimed at getting the best of both worlds from static programming languages which have high performance and safety with the flexibility of dynamic programming languages, however they all emphasize performance a feature. In this article, we look at how their performance varies over kernel matrix calculations, investigate approaches to code optimization and other usability features of the languages.

Kernel matrix calculations form the basis of kernel methods in machine learning applications. They scale rather poorly `O(m n^2)`, `n` is the number of items and `m` is the number of elements in each item. In our exercsie `m` will be constant and we will be looking at execution time in each implementation as `n` increases. Here `m = 784`, and `n = 1k, 5k, 10k, 20k, 30k`, each calculation is run 3 times and an average is taken. We disallow any use of BLAS and only allow use of packages or modules from the language standard library of each language. A reference for kernel functions is located [here](http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/).

While preparing the code for this article, the Chapel, D, and Julia communities were very helpful and patient with all enquiries, so they are acknowledged here.

## Language Benchmarks for Kernel Matrix Calculation
<img class="plot" src="https://github.com/dataPulverizer/KernelMatrixBenchmark/blob/master/images/benchplot.jpg" width="600">

|     n |         D |      Julia |    Chapel |
| ----- |:---------:|:----------:| ---------:|
|  1000 |  0.0094   |  0.0105    |  0.0403   |
|  5000 |  0.2711   |  0.3133    |  0.9783   |
| 10000 |  1.4359   |  2.1713    |  3.8594   |
| 20000 | 11.2670   | 11.1537    | 16.7087   |
| 30000 | 31.0559   | 29.2007    | 40.2499   |

## Environment

The code was run on a computer with an Ubuntu 20.04 OS, 32 GB memory and an Intel® Core™ i9-8950HK CPU @ 2.90GHz with 6 cores and 12 threads.

```
$ julia --version
julia version 1.4.1
```

```
$ dmd --version
DMD64 D Compiler v2.090.1
```

```
ldc2 --version
LDC - the LLVM D compiler (1.18.0):
  based on DMD v2.088.1 and LLVM 9.0.0
```

```
$ chpl --version
chpl version 1.22.0
```

## Implementations 

Efforts were made to avoid non standard libraries while implementing these kernel functions. The reasons for this are:

* Making it easy for a reader after installing the language to copy and run the code. Having to install external libraries can be a bit of a "faff".
* Packages outside standard libraries can go extinct so avoiding external libraries keeps the article and code relevant.
* It's completely transparent and shows how each language works.

### Chapel

Chapel uses a `forall` loop to parallelizes the code over threads. Also C pointers to each item is used rather than the default array notation and `guided` iteration over the indexes:

```
//Chapel
proc calculateKernelMatrix(K, data: [?D] ?T)
{
  var n = D.dim(0).last;
  var p = D.dim(1).last;
  var E: domain(2) = {D.dim(0), D.dim(0)};
  var mat: [E] T;
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
```

Chapel code was the most difficult to optimise for performance and required the most in terms of code changes.

### D

D uses a `taskPool` of threads from its `std.parallel` package to parallelize code. The D code underwent the least amount of change for performance optimization, a lot of the benefit came from the compiler (discussed later). My implementation of a `Matrix` allows columns to be selected by reference `refColumnSelect`.

```
auto calculateKernelMatrix(alias K, T)(K!(T) kernel, Matrix!(T) data)
{
  long n = data.ncol;
  auto mat = Matrix!(T)(n, n);

  foreach(j; taskPool.parallel(iota(n)))
  {
    auto arrj = data.refColumnSelect(j).array;
    foreach(long i; j..n)
    {
      mat[i, j] = kernel(data.refColumnSelect(i).array, arrj);
      mat[j, i] = mat[i, j];
    }
  }
  return mat;
}
```

### Julia

The Julia code uses `@threads` macro for parallelising the code and `@views` macro for referencing arrays. One confusing thing about Julia's arrays is their reference status. Sometimes as in this case arrays will behave like value objects and they have to be referenced by using the `@views` macro otherwise they generate copies, at other times they behave like reference objects, for example passing them into a function. It can be a little tricky dealing with this because you don't always know what set of operations will generate a copy, but where this occurs `@views` provides a good solution.

The `Symmetric` type saves the small bit of extra work needed for allocating to both sides of the matrix.

```
//Julia
function calculateKernelMatrix(Kernel::K, data::Array{T}) where {K <: AbstractKernel,T <: AbstractFloat}
  n = size(data)[2]
  mat = zeros(T, n, n)
  @threads for j in 1:n
      @views for i in j:n
          mat[i,j] = kernel(Kernel, data[:, i], data[:, j])
      end
  end
  return Symmetric(mat, :L)
end
```

The `@bounds` and `@simd` macros in the kernel functions were used to turn bounds checking off and apply SIMD optimization to the calculations:

```
struct DotProduct <: AbstractKernel end
@inline function kernel(K::DotProduct, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
  ret = zero(T)
  m = length(x)
  @inbounds @simd for k in 1:m
      ret += x[k] * y[k]
  end
  return ret
end
```

These optimizations are quite visible but very easy to apply.

## Performance optimization

The process of performance optimization in all three languages was very different and all three communities were very helpful in the process. But there were some common themes.

* Direct dispatching of kernel function types rather than using polymorphism. This means that when passing the kernel function, use parametric (static compile time) polymorphism rather than runtime (dynamic) polymorphism were dispatch with virtual functions carries a performance penalty. In this case it made little or no difference.
* Using views rather than copying data over multiple threads – makes a big difference
* Parallelising the calculations – makes a huge difference
* Knowing if your array is row/column major and using that in your calculation makes a huge difference.
* Bounds checks and compiler optimizations – makes a huge difference especially in Chapel and D.

In terms of language specific issues, getting to performant code in Chapel was by far the most challenging and the Chapel code changed the most from easy to read array operations to using pointers and guided iterations. But on the compiler size it was relatively easy to add `--fast`.

In D the code changed very little and most of the performance was gained in compiler optimizations. D’s LDC compiler is rich in terms of options for performance optimization. It has 8 different `-O` optimization levels and a myriad of other flags that affect performance in different ways. In this case the flags used were `-O5 --boundscheck=off –ffast-math` representing bounds checking and LLVM’s fast-math.

In the Julia the macro changes discussed previously markedly improved the performance but they were not too intrusive. I did attempt changing the optimization `-O` level but it didn’t improve the performance.

## Quality Of Life

This section examines the relative pros and cons around the convenience and ease of use of each language. People underestimate the effort it takes to use a language day to day, the support and infrastructure it takes is a lot so it’s worth comparing various facets of each language. To avoid the TLDR the end of this paragraph has a handy table comparing the language features discussed here. Every effort has been made to be as objective as possible but comparing programming languages is difficult, bias prone, and contentious so read this section with that in mind. Also some elements are from the “data science”/technical/scientific computing and others are more general.

### Interactivity

When developing code you generally want a fast code/compile/result loop so that you can quickly observe the results of what you’ve written so that you can progress or make necessary changes; Julia’s interpreter is hands down the best for this and offers a smooth and feature-rich development experience, but D comes a close second. If you’ve every compiled anything using a C++ compiler this loop can be rather tedious even when compiling rather simple code. D has three compilers, the standard DMD compiler, the LLVM-based LDC compiler, and the GCC-based GDC. In this process, I used the DMD and LDC compilers. DMD has **very** fast compilation times which is great for the development process. Once you have developed your code, the LDC compiler is great for getting great performance from what you’ve written. The Chapel compiler is very slow in comparison, to give you an idea running Linux’s `time` command on DMD vs Chapel’s compiler with no optimizations gives us for D:

```
real	0m0.545s
user	0m0.447s
sys	0m0.101s
```
Compared with Chapel:

```
real	0m5.980s
user	0m5.787s
sys	0m0.206s
```
That’s a large actual and *psychological* difference, it makes you more reluctant to check your work if you have to wait for outputs especially when they could run faster than it takes to compile.

### Documentation and examples

This is partially personal but resources such as books, blogs and documentation play a large role also. Julia’s documentation is again hands down the best of the bunch. I guess the best way is to compare them all with Python’s official documentation which is *the* gold standard for programming languages, it combines examples with formal definitions and tutorials in a seamless and user friendly way. It is **not** fair to compare documentation from relatively new languages to Python which  has been around for many years and has had a lot of resources poured into it but it gives an idea of how they compare.

 Julia’s documentation the is closest to Python’s documentation quality and gives the user a very smooth detailed and relatively painless transition into the language, it also has a rich ecosystem of blogs and topics on many aspects of the language are easy to come by. D’s official documentation is not as good and can be challenging and frustrating, however there is a *very* good free book [“Programming in D”](https://wiki.dlang.org/Books) which is a great introduction to the language. D takes after C++ and was intended to be a replacement of sorts so one great book can not cover every aspect of the language in detail, so even after reading the book expect to still struggle with advanced concepts and there isn’t enough material out there that covers it. Chapel’s documentation is actually pretty good from a getting things done perspective though the examples varies in presence and quality, often you have to really know what you are doing so that you can look for the right thing and often there won’t be any or enough examples. By way of comparison, compare the file i/o libraries in Chapel, D, and Julia. Chapel’s i/o library has too few examples but is relatively clear and straightforward, D’s i/o documentation is kind of spread across a few modules and difficult to follow, Julia’s i/o documentation has lots of examples and is clear and easy to follow.

One of the factors affecting Chapel’s adoption is lack of examples, for example its arrays have a very non-standard way of interfacing with them, so you have to work very hard to become familiar with them, were as even though D’s documentation may not be as good in places it is often very similar to C/C++ and so maybe gets away it a bit more.

### Multi-dimensional Array support

“Arrays” here do not refer to native C/C++ style arrays but mathematical arrays. Julia and Chapel ship with array support and D does not. In the implementation of kernel matrix, I wrote my own matrix object in D – which is not difficult if you understand how to in principle, but there is a linear algebra library in D called Lubeck which has impressive performance characteristics however as I outlined before, we are avoiding non standard packages. Julia’s arrays are by far the easiest and most familiar to people that frequently use these things, Chapel arrays are more difficult to get started than Julia’s but are designed to be run on single core, multicore and computer clusters using the same code which is a feature Julia’s arrays lack.

### Language Power

Since Julia is a dynamic programming language some might say, “well Julia’s is a dynamic language so you can do whatever you want therefore the debate is over” but it’s more complicated than that. There is power in type system, Julia has a type system similar in nature to a type system from a static language so you can write code as if you were using a static language but you can do things reserved only for dynamic languages, it has a highly developed generic and meta-programming syntax and powerful macros. It also has highly flexible object system and multiple dispatch. When you combine this with being a dynamic programming language, you can do things in Julia impossible in D or Chapel; Julia is the most powerful language out of the three.

D was intended to be a replacement for C++ and takes very much after C++ (and borrows from Java) but makes template programming and compile time evaluation much more user friendly than in C++, it is a single dispatch language (though multi-methods are available in a package), instead of macros D has string and template “mixins” which serve a similar purpose.

Chapel has some generic programming support and nascent support for single dispatch OOP, no macro support, and is not yet as mature as D in terms of these features.

### Concurrency & Parallel Programming

Nowadays new languages tout support for concurrency and it’s popular subset parallelism but the detail varies a lot between languages. Parallelism is more relevant in this example and all three languages deliver. Writing the parallel for loop required here is very straightforward in all three languages.

Chapel’s concurrency model has much more emphasis on data parallelism but has tools for task based parallelism and ships with support for cluster-based concurrency.

Julia has good support for both concurrency and parallelism, but consumes the most system resources in terms of memory out of the three when running things in parallel.

D has industry strength support for parallelism and concurrency, it’s the most efficient from a memory and execution point of view, each thread in D will use the least memory out of the three – though it’s support for threading is much less well documented with examples.

Basically if you write parallel code in Chapel, D, and Julia, you could get more out of your system resources using D, then Chapel, and finally Julia – though this is separate from execution times.

### Standard Library

How good is the standard library of all three languages in general? What range of tasks do they allow me to do? It’s a tough question because library quality and documentation factor in also. All three languages have very good standard libraries, D has the most comprehensive standard library, but Julia is great second then Chapel, but things are never that simple. For example if I was writing code to do binary i/o I’d probably choose Julia, it’s the most straightforward clear interface and documentation, followed by Chapel and then D, though I’ve done a simple this for an IDX file reader and found D’s i/o the fastest to execute, and Julia’s easy to write for cases unavailable in the other two languages.

### Package Managers & Package Ecosystems

In terms of documentation, usage and features, D’s dub package manager is the best. D also has a rich package ecosystem in the [Dub website](https://code.dlang.org/), Julia’s package manager runs very tightly integrated with GitHub and is a good package system with good documentation. Chapel has a package manager but does not a highly developed package ecosystem.

### C Integration

C interop is easy in all three languages; though Chapel’s is not as well popularised but has good documentation but D’s documentation is better though Julia’s documentation is the best for this. Oddly enough though, none of the language documentations show the commands required to compile your own C code and integrate it with the language which is an oversight especially when it comes to novices. It’s easy to search for and find examples for D and Julia though.

### Community

All three languages have convenient places where users can ask questions. For Chapel, the easiest place is Gitter, for Julia it’s Discourse (though there is a Julia Gitter) and for D it’s the official website forum. In terms of activity Julia community is by far the most active, followed by D and then Chapel. Though I’ve found that you’ll get good responses from all three communities but you’ll probably get quicker answers from D and Julia communities.

|                             | Chapel        | D             | Julia    |
| --------------------------- |:-------------:|:-------------:| --------:|
| Compilation/Interactivty    | Slow          | Fast          | Best     |
| Documentation & Examples    | Detailed      | Patchy        | Best     |
| Multi-dimensional Arrays    | Yes           | Native Only   | Yes      |
| Language Power              | Good          | Great         | Best     |
| Concurrency & Parallelism   | Great         | Great         | Good     |
| Standard Library            | Good          | Great         | Great    |
| Package Manager & Ecosystem | Nascent       | Best          | Great    |
| C Integration               | Great         | Great         | Great    |
| Community                   | Small         | Large         | Biggest  |
Table for quality of life features in Chapel, D & Julia

## Summary

If you are a novice programmer writing numerical algorithms and doing calculations based in scientific computing and want a fast language that's easy to use Julia is your best bet. If you are an experienced programmer working in the same space Julia is still a great option. If you specifically want a more conventional static compiled high performance language with all the "bells and whistles" but want something more productive, safer and less painful than C++ then D is your best bet. You can write "anything" in D and get great performance from its compilers. If you need to get array calculations happening on clusters then Chapel is probably the easiest place to go.

In terms of raw performance on this task Julia was the winner for the larger data objects with D a very close second but was the best for smaller data - but both are  *very* close, my feeling is that adding SIMD support and a slightly different implementation of my D Matrix object using a different memory model could easily put D ahead. This exercise reveals that Julia's label as a high performance language is more than hype it has held it's own against highly competitve languages. It was quite hard to get competitve performance from Chapel which I did not expect - it took a lot of investigation from the Chapel team to come up with the current solution, however it's very early in the development of the language and I look forward to further improvements.
