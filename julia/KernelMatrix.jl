using Base.Threads: @threads, @spawn
using Random: shuffle!
using LinearAlgebra: Symmetric

# Kernel Function Types
#======================#
abstract type AbstractKernel{T <: AbstractFloat} end

struct DotProduct{T} <: AbstractKernel{T} end
@inline function kernel(K::DotProduct{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
  dist = T(0)
  m = length(x)
  @inbounds @simd for i in 1:m
    dist += x[i] * y[i]
  end
  return dist
end

struct Gaussian{T} <: AbstractKernel{T}
  theta::T
end
@inline function kernel(K::Gaussian{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
  dist::T = T(0)
  tmp::T = T(0)
  m = length(x)
  @inbounds @simd for i in 1:m
    tmp = x[i] - y[i]
    dist += tmp * tmp
  end
  return exp(-sqrt(dist)/K.theta)
end

struct Polynomial{T} <: AbstractKernel{T}
  d::T
  offset::T
end
@inline function kernel(K::Polynomial{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  dist::T = T(0)
  m = length(x)
  @inbounds @simd for i = 1:m
    dist += x[i] * y[i]
  end
  return (dist + K.offset)^K.d
end

struct Exponential{T} <: AbstractKernel{T}
  theta::T
end
@inline function kernel(K::Exponential{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  dist::T = T(0)
  m = length(x)
  @inbounds @simd for i in 1:m
    dist -= abs(x[i] - y[i])
  end
  return exp(dist/K.theta)
end

struct Log{T} <: AbstractKernel{T}
  beta::T
end
@inline function kernel(K::Log{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  dist::T = T(0)
  m = length(x)
  @inbounds @simd for i in 1:m
    dist += abs(x[i] - y[i])^K.beta
  end
  dist ^= (1/K.beta)
  return -log(1 + dist)
end

struct Cauchy{T} <: AbstractKernel{T}
  theta::T
end
@inline function kernel(K::Cauchy{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  dist::T = T(0)
  tmp::T = T(0)
  m = length(x)
  @inbounds @simd for i in 1:m
    tmp = x[i] - y[i]
    dist += tmp*tmp
  end
  dist = sqrt(dist)/K.theta
  return 1/(1 + dist)
end

struct Power{T} <: AbstractKernel{T}
  beta::T
end
@inline function kernel(K::Power{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  dist::T = T(0)
  m = length(x)
  @inbounds @simd for i in 1:m
    dist += abs(x[i] - y[i])^K.beta
  end
  return -dist^(1/K.beta)
end

struct Wave{T} <: AbstractKernel{T}
  theta::T
end
@inline function kernel(K::Wave{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  dist::T = T(0)
  m = length(x)
  @inbounds @simd for i in 1:m
    dist += abs(x[i] - y[i])
  end
  tmp = K.theta/dist;
  return tmp*sin(1/tmp);
end

struct Sigmoid{T} <: AbstractKernel{T}
  beta0::T
  beta1::T
end
@inline function kernel(K::Sigmoid{T}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
  dist::T = T(0)
  m = length(x)
  @inbounds @simd for i = 1:m
    dist += x[i] * y[i]
  end
  return tanh(K.beta0 * dist + K.beta1)
end

#=======================================================================================#

function calculateKernelMatrix(Kernel::AbstractKernel{T}, data::AbstractArray{T, N}) where {T, N}
  n = size(data)[2]
  mat::Array{T, 2} = zeros(T, n, n)
  @threads for j in 1:n
      @views for i in j:n
          mat[i,j] = kernel(Kernel, data[:, i], data[:, j])
      end
  end
  return Symmetric(mat, :L)
end
