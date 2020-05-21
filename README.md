# Kernel Matrix Benchmarking in Chapel, D, and Julia

Kernel matrix calculations forms the basis of many kernel methods in machine learning applications. While writing kernel matrix 
code, I wandered how the performance of these calculations would compare between Chapel, D, and Julia. All these languages are relatively new languages compared to those frequently used in machine learning. They were all built for slightly different reasons but all target high performance programming so it would be interesting to see how they all perform in this one task.

The original problem was looking at generating kernel matrices for the test MNIST data set (http://yann.lecun.com/exdb/mnist/), though those data sets won't be used here, that is where the problem size range comes in, so it is not "arbitrary". The code presented are have generic interfaces to the kernel function ... so some arbitrary kernel function can be passed and the kernel matrix calculated. Kernel matrix calculations are very interesting because (apart from the dot kernel) they can't be completely abastracted away to BLAS libraries, so a good part of the calculation relies on the efficiency of the programming language being used. They also scale rather poorly `O(mn^2)` where `n` is the number of items and `m` is the number of elements in each item, in this case `m` will be constant and we will be looking at what happens to execution times as `n` increases. So `m = 784`, and `n = 1k, 5k, 10k, 20k, 30k`. Each calculation is run 3 times and an average is taken, this approach will suffice for our purpose.

We'll look at the times for the dot kernel only, but we don't allow use of BLAS or anything that removes the generalization of of the functional form ... meaning that the calculation is performed in the same way as it would be for any other kernel function. There is a lovely article for kernels located [here](http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/).

## Language Benchmarks for Kernel Matrix Calculation
<img class="plot" src="https://github.com/dataPulverizer/KernelMatrixBenchmark/blob/master/images/benchplot.jpg">

<style>
.plot {
   width: 40vw;
}
</style>
