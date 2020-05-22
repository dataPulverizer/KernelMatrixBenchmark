include("KernelMatrix.jl")


function bench(::Type{T}, Kernel::K, n::Array{Int64, 1}, verbose::Bool = true) where {K, T}
  times::Array{Float64, 1} = zeros(Float64, length(n))
  for i in 1:length(n)
    _times::Array{Float64, 1} = zeros(Float64, 3)
    data = rand(T, (784, n[i]))
    for j in 1:3
      t1 = time()
      mat = calculateKernelMatrix(Kernel, data);
      t2 = time()
      _times[j] = t2 - t1
    end
    times[i] = (_times[1] + _times[2] + _times[3])/3
    if verbose
      println("Average time for n = ", n[i], ", ", times[i], " seconds.")
      println("Detailed times: ", _times);
    end
  end
  return times
end


function main()
  K = DotProduct();
  
  precompile(kernel, (typeof(K), Array{Float32, 1}, Array{Float32, 1}))
  precompile(calculateKernelMatrix, (typeof(K), Array{Float32, 2}))
  precompile(bench, (DataType, typeof(K), Array{Int64, 1}))
  
  tt = bench(Float32, K, [1000, 5000, 
  10_000, 20_000, 30_000])
  println("bench: ", tt)
  return
end

#=
  To run:
  julia script.jl
  bench: [0.01053166389465332, 0.3132596015930176, 2.171336015065511, 11.15370806058248, 29.200666030248005]
=#
main()
