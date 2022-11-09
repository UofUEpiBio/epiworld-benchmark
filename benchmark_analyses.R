library(ggplot2)
library(data.table)

benchmark <- fread("benchmark.csv")

benchmark[, speed := (50 * size) / time * 1e-6]

figdim <- 1024

ggplot(benchmark, aes(y = time, x = size / 1e6)) +
  geom_jitter(height = 0, alpha = .5) +
  scale_y_log10(breaks = c(0.1, 1, 10, 50)) +
  scale_x_log10() +
  labs(
    x = "Mill. Agents\n(log10 scale)",
    y = "Time in Seconds\n(log10)"
    )
ggsave("benchmark_analyses_elapsed_sir.pdf", width = figdim, height = figdim, units = "px")

ggplot(benchmark, aes(y = speed, x = size / 1e6)) +
  geom_jitter(height = 0, alpha = .5) +
  scale_x_log10() +
  labs(
    x = "Mill. Agents\n(log10 scale)",
    y = "Speed (Mill. Agents x Day x Second)\n(log10 scale)"
    )
ggsave("benchmark_analyses_speed_sir.pdf", width = figdim, height = figdim, units = "px")

