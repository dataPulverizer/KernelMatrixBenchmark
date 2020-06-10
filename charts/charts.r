require(data.table)
require(ggplot2)
require(scales)

# For language benchmarks
createPlot = function(folder)
{
  results = Map(fread, c(paste0("../", folder,"/chapelBench.csv"), 
                         paste0("../", folder,"/dBench.csv"), 
                     paste0("../", folder,"/juliaBench.csv")))
  results = rbindlist(results)
  results[, kernel := gsub(" ", "", kernel)]
  p = ggplot(results, aes(x = nitems, y = time, color = language)) + geom_line() + 
         geom_point() + scale_y_continuous(trans = "log10",
                  labels = trans_format("log10", math_format(10^.x))) + 
         theme(legend.position="top") + ylab("time (s)") + 
         xlab("Number Of Items") + facet_wrap(~ kernel, scale = "free_y")
  return(p)
}

p1 = createPlot("data")
p2 = createPlot("fmdata")

jpeg(file = "benchplot.jpg", width = 9, height = 7, units = "in", res = 200)
plot(p1)
dev.off()

jpeg(file = "fmbenchplot.jpg", width = 9, height = 7, units = "in", res = 200)
plot(p2)
dev.off()


# Difference between NDSlice and My basic matrix implementation
results = Map(fread, c("../data/dNDSliceBench.csv", "../data/dBench.csv"))
results[[1]][, language := "NDSlice"]
results[[2]][, time := 100*(time - results[[1]][,time])/time]
results = results[[2]]

p3 = ggplot(results, aes(x = nitems, y = time, fill = kernel)) + geom_col() + 
       theme(legend.position="none", plot.title = element_text(hjust = 0.5)) + 
       ylab("% Difference\n(+ive = ndslice is faster)") + 
       xlab("Number Of Items") + facet_wrap(~ kernel, scale = "free_y") + 
       ggtitle("Matrix and NDSlice percentage time difference")

jpeg(file = "ndsliceDiagnostic.jpg", width = 7, height = 7, units = "in", res = 200)
plot(p3)
dev.off()
