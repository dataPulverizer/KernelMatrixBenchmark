using Base.Threads: @threads, @spawn
using Random: shuffle!
using LinearAlgebra
#using LoopVectorization: @avx

#=
  Avoid conflicts when multi-threading
  when using BLAS
=#
# BLAS.set_num_threads(1)

# Kernel Methods
#===============#
abstract type AbstractKernel end

# For regular kmeans clustering
struct DotProduct <: AbstractKernel end
@inline function kernel(K::DotProduct, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
  ret = zero(T)
  m = length(x)
  @inbounds @simd for k in 1:m
      ret += x[k] * y[k]
  end
  return ret
end

struct Polynomial{T} <: AbstractKernel
  d::T
end
@inline function kernel(K::Polynomial{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  ret::eltype(x) = 0
  @inbounds @simd for i = 1:length(x)
    ret += x[i] * y[i]
  end
  return (ret + 1)^K.d
end

struct Gaussian{T} <: AbstractKernel
  gamma::T
end
@inline function kernel(K::Gaussian{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
  ret = zero(T)
  m = length(x)
  @inbounds @simd for k in 1:m
    tmp = x[k] - y[k]
    ret -= tmp * tmp
  end
  return exp(K.gamma * ret)
end

struct GaussianRBF{T} <: AbstractKernel
  gamma::T
end


struct LaplaceRBF{T} <: AbstractKernel
  sigma::T
end
@inline function kernel(K::LaplaceRBF, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  ret::eltype(x) = 0
  @inbounds @simd for i in 1:length(x)
    tmp = x[i] - y[i]
    ret -= tmp * tmp
  end
  return exp(ret/K.sigma)
end

struct Sigmoid{T} <: AbstractKernel
  alpha::T
  c::T
end
@inline function kernel(K::Sigmoid, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  ret::eltype(x) = 0
  @inbounds @simd for i = 1:length(x)
    ret += x[i] * y[i]
  end
  return tanh(K.alpha * ret + K.c)
end

#=======================================================================================#

"""
  Function to calculate the kernel matrix
  It's a symmetrix matrix so only the diagonal and lower part of the 
  matrix are actually calculated

  T - The element type to return - we accept that the user may want to output
      a different element type than the type of data because the kernel
      matrix can be very large so the output may be down-sampled.
  K - The kernel to use
  data - 2D Arrays/Matrix with the data
  trans - does the array/matrix need to be transposed the kernel matrix
      is calculated over columns (not rows) for performance reasons because 
      arrays in Julia are column major
"""
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


