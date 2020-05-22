# A look at Chapel, D, and Julia using kernel matrix calculations

*Date: 2020-05-22*

## Introduction

It seems each time you turn around there is a new programming language aimed at solving some specific problem set. Increased proliferation of programming languages and data are deeply connected in a fundamental way and increasing demand for “data science” computing is a related phenomenon. In the field of scientific computing Chapel, D, and Julia  are highly relevant programming languages. They arise from different needs and are aimed at different problem sets. Chapel focuses on data parallelism on single multicore machines and large clusters, D initially was developed as a more productive safer alternative to C++ and Julia was developed for technical and scientific computing and aimed at getting the best of both worlds from static programming languages which have high performance and safety with the flexibility of dynamic programming languages, however they all emphasize performance as a feature. In this article, we look at how their performance varies over kernel matrix calculations, and present approaches to performance optimization and other usability features of the languages.

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

Table shows the average execution times (seconds) for each language for varying number of items (n).

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

### Compilation

Chapel
```
chpl script.chpl kernelmatrix.chpl --fast && ./script
```

D
```
ldc2 script.d kernelmatrix.d arrays.d -O5 --boundscheck=off --ffast-math && ./script
```

Julia - no compilation required but can be run from command line:
```
julia script.jl
```

## Implementations 

Efforts were made to avoid non standard libraries while implementing these kernel functions. The reasons for this are:

* Making it easy for a reader after installing the language to copy and run the code. Having to install external libraries can be a bit of a "faff".
* Packages outside standard libraries can go extinct so avoiding external libraries keeps the article and code relevant.
* It's completely transparent and shows how each language works.

### Chapel

Chapel uses a `forall` loop to parallelize over threads. Also C pointers to each item is used rather than the default array notation and `guided` iteration over indices are used:

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

D uses a `taskPool` of threads from its `std.parallel` package to parallelize code. The D code underwent the least amount of change for performance optimization, a lot of the benefit came from the compiler used and flags selected (discussed later). My implementation of a `Matrix` allows columns to be selected by reference `refColumnSelect`.

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

* Static dispatching of kernel functions instead of using polymorphism. This means that when passing the kernel function, use parametric (static compile time) polymorphism rather than runtime (dynamic) polymorphism were dispatch with virtual functions carries a performance penalty. In this case it made little or no difference.
* Using views/references rather than copying data over multiple threads – makes a big difference.
* Parallelising the calculations – makes a huge difference.
* Knowing if your array is row/column major and using that in your calculation makes a huge difference.
* Bounds checks and compiler optimizations – makes a huge difference especially in Chapel and D.

In terms of language specific issues, getting to performant code in Chapel was by far the most challenging and the Chapel code changed the most from easy to read array operations to using pointers and guided iterations. But on the compiler side it was relatively easy to add `--fast` and get a very large optimization boost.

In D the code changed very little and most of the performance was gained in compiler optimizations. D’s LDC compiler is rich in terms of options for performance optimization. It has 8 different `-O` optimization levels and a myriad of other flags that affect performance in various ways. In this case the flags used were `-O5 --boundscheck=off –ffast-math` representing aggressive compiler optimizations, bounds checking, and LLVM’s fast-math.

In the Julia the macro changes discussed previously markedly improved the performance but they were not too intrusive. I tried changing the optimization `-O` level but this did not improve performance.

## Quality of life

This section examines the relative pros and cons around the convenience and ease of use of each language. People underestimate the effort it takes to use a language day to day, the support and infrastructure required is a lot so it is worth comparing various facets of each language. Readers seeking to avoid the TLDR should scroll to the end of this section for the table comparing the language features discussed here. Every effort has been made to be as objective as possible but comparing programming languages is difficult, bias prone, and contentious so read this section with that in mind. Some elements looked at such as arrays are from the “data science”/technical/scientific computing point of view and others are more general.

### Interactivity

Programmers want a fast code/compile/result loop during development to quickly observe results and outputs in order to make progress or necessary changes; Julia’s interpreter is hands down the best for this and offers a smooth and feature-rich development experience, and D comes a close second. This code/compile/result loop in compilers for example gcc's C++ can be slow even when compiling simple code. D has three compilers, the standard DMD compiler, the LLVM-based LDC compiler, and the GCC-based GDC. In this development process, the DMD and LDC compilers were used. DMD has **very** fast compilation times which is great for development. The LDC compiler is great for getting great performance. Chapel's compiler is very slow in comparison, to give an example, running Linux’s `time` command on DMD vs Chapel’s compiler for the kernel matrix code with no optimizations gives us for D:

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
That’s a large actual and *psychological* difference, it can make programmers reluctant to check their work and delay the development loop if they have to wait for outputs especially when source code increases in volume and compilation times become significant.

It is worth mentioning however that when developing packages in Julia compilation times can be very long, and users have noticed that when they load some packages compilation times can stretch so the experience of the development loop in Julia could vary, but in this specific case the process was seamless.

### Documentation and examples

One way of comparing documentation in the different languages is to compare them all with Python’s official documentation which is *the* gold standard for programming languages, it combines examples with formal definitions and tutorials in a seamless and user friendly way. Since many programmers are familiar with the Python documentation this approach gives an idea of how they compare.

Julia’s documentation the is closest to Python’s documentation quality and gives the user a very smooth detailed and relatively painless transition into the language, it also has a rich ecosystem of blogs and topics on many aspects of the language are easy to come by. D’s official documentation is not as good and can be challenging and frustrating, however there is a *very* good free book [“Programming in D”](https://wiki.dlang.org/Books) which is a great introduction to the language but no single book can cover a programming language and there are not many sources for advanced topics. Chapel’s documentation is quite good for getting things done though examples vary in presence and quality, often the programmer needs a lot of knowledge to look in the right place. A good topic for comparison is file i/o libraries in Chapel, D, and Julia. Chapel’s i/o library has too few examples but is relatively clear and straightforward, D’s i/o is kind of spread across a few modules and documentation is more difficult to follow, Julia’s i/o documentation has lots of examples and is clear and easy to follow.

Perhaps one factor affecting Chapel’s adoption is lack of examples, since its arrays have a non-standard interface the user has to work hard to become familiar with them, were as even though D’s documentation may not be as good in places it is often similar to C/C++ and so gets away with more sparse documentation.

### Multi-dimensional Array support

“Arrays” here do not refer to native C/C++ style arrays but mathematical arrays. Julia and Chapel ship with array support and D does not. In the implementation of kernel matrix, I wrote my own matrix object in D – which is not difficult if you understand the principle but it's not something a user wants to do, however D has a linear algebra library called Lubeck which has impressive performance characteristics and interfaces with all the usual BLAS implementations. Julia’s arrays are by far the easiest and most familiar, Chapel arrays are more difficult to get started than Julia’s but are designed to be run on single core, multicore and computer clusters using the same or very similar code which is a good unique selling point.

### Language power

Since Julia is a dynamic programming language some might say, “well Julia’s is a dynamic language which is far more permissive than static programming languages therefore the debate is over” but it’s more complicated than that. There is power in static type systems, Julia has a type system similar in nature to type systems from static languages so you can write code as if you were using a static language but you can do things reserved only for dynamic languages, it has a highly developed generic and meta-programming syntax and powerful macros. It also has highly flexible object system and multiple dispatch. This mix of features is what makes Julia is the most powerful language of the three.

D was intended to be a replacement for C++ and takes very much after C++ (and also borrows from Java) but makes template programming and compile time evaluation much more user friendly than in C++, it is a single dispatch language (though multi-methods are available in a package), instead of macros D has string and template “mixins” which serve a similar purpose.

Chapel has generic programming support and nascent support for single dispatch OOP, no macro support, and is not yet as mature as D or Julia in these terms.

### Concurrency & Parallel Programming

Nowadays new languages tout support for concurrency and it’s popular subset parallelism but the detail varies a lot between languages. Parallelism is more relevant in this example and all three languages deliver. Writing parallel for loops required is straightforward in all three languages.

Chapel’s concurrency model has much more emphasis on data parallelism but has tools for task parallelism and ships with support for cluster-based concurrency.

Julia has good support for both concurrency and parallelism, but consumes the most system resources in terms of memory out of the three when running processes in parallel.

D has industry strength support for parallelism and concurrency, it’s the most efficient from a memory and execution point of view, each thread in D will usually use the least memory out of the three – though its support for threading is much less well documented with examples.

Gvien the choice of wriitng parallel code in Chapel, D, and Julia, you could get more out of your system resources using D, then Chapel, and finally Julia – though this is separate from execution times.

### Standard Library

How good is the standard library of all three languages in general? What range of tasks do they allow users to easily tend to? It’s a tough question because library quality and documentation factor in. All three languages have very good standard libraries, D has the most comprehensive standard library, but Julia is great second then Chapel, but things are never that simple. For example for a user writing code for binary i/o Julia may be the best choice, its the most straightforward clear interface and documentation, followed by Chapel and then D, though in my implementation of an IDX file reader, D’s i/o was the fastest, and the Julia code was easy to write for cases unavailable in the other two languages.

### Package Managers & Package Ecosystems

In terms of documentation, usage and features, D’s Dub package manager is the most comprehensive. D also has a rich package ecosystem in the [Dub website](https://code.dlang.org/), Julia’s package manager runs tightly integrated with GitHub and is a good package system with good documentation. Chapel has a package manager but does not a highly developed package ecosystem.

### C Integration

C interop is easy in all three languages; though Chapel’s is not as well popularised but has good documentation but D’s documentation is better though Julia’s documentation is the most comprehensive. Oddly enough though, none of the language documentations show the commands required to compile your own C code and integrate it with the language which is an oversight especially when it comes to novices. It’s easy to search for and find examples for D and Julia though.

### Community

All three languages have convenient places where users can ask questions. For Chapel, the easiest place is Gitter, for Julia it’s Discourse (though there is a Julia Gitter) and for D it’s the official website forum. In terms of activity Julia community is the most active, followed by D and then Chapel. Though I’ve found that you’ll get good responses from all three communities but you’ll probably get quicker answers from D and Julia communities.

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

If you are a novice programmer writing numerical algorithms and doing calculations based in scientific computing and want a fast language that's easy to use Julia is your best bet. If you are an experienced programmer working in the same space Julia is still a great option. If you specifically want a more conventional industrual strength static compiled high performance language with all the "bells and whistles" but want something more productive, safer and less painful than C++ then D is your best bet. You can write "anything" in D and get great performance from its compilers. If you need to get array calculations happening on clusters then Chapel is probably the easiest place to go.

In terms of raw performance on this task Julia was the winner for the larger data objects with D a very close second but D was the best for smaller data - but both are  *very* close, my feeling is that adding SIMD support could easily put D ahead or on par with Julia at the larger data sizes. This exercise reveals that Julia's label as a high performance language is more than just hype it has held it's own against highly competitve languages. It was quite hard to get competitve performance from Chapel which I did not expect - it took a lot of investigation from the Chapel team to come up with the current solution, however it is early in the development of this language and we should look forward to further improvements.
