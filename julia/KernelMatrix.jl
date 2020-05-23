using Base.Threads: @threads, @spawn
using Random: shuffle!
using LinearAlgebra

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

