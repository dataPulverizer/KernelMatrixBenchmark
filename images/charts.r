require(ggplot2)

# For language benchmarks
results = Map(read.csv, c("../data/chapelBench.csv", "../data/dBench.csv", 
                     "../data/juliaBench.csv"))
results = Reduce(rbind.data.frame, results)
results$kernel = gsub(" ", "", results$kernel)

p = ggplot(results, aes(x = nitems, y = time, color = language)) + geom_line() + 
       geom_point() + theme(legend.position="top") + ylab("time (s)") + 
       xlab("#items") + facet_wrap(~ kernel, scale = "free_y")

jpeg(file = "benchplot.jpg", width = 7, height = 5, units = "in", res = 200)
plot(p)
dev.off()

# Difference between NDSlice and My basic matrix implementation
results = Map(read.csv, c("../data/dNDSliceBench.csv", "../data/dBench.csv"))
results[[1]]$language = "NDSlice"
results[[2]]$time = results[[2]]$time - results[[1]]$time
results = results[[2]]

p = ggplot(results, aes(x = nitems, y = time, color = kernel)) + geom_line() + 
       geom_point() + theme(legend.position="top") + ylab("time (s)") + 
       xlab("#items")

jpeg(file = "ndsliceDiagnostic.jpg", width = 6, height = 4, units = "in", res = 200)
plot(p)
dev.off()
