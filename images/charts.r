require(ggplot2)

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

