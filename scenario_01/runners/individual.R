#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(individual)
  library(jsonlite)
})

parse_args <- function(x) {
  if (length(x) %% 2L != 0L) stop("Arguments must be --key value pairs")
  keys <- sub("^--", "", x[seq(1L, length(x), by = 2L)])
  values <- x[seq(2L, length(x), by = 2L)]
  stats::setNames(as.list(values), gsub("-", "_", keys))
}

# Resident memory in bytes from /proc/self/status: VmRSS now and VmHWM, its
# high-water mark. NA off Linux. Uncollected garbage counts as resident; no
# collection is forced beyond the one before the simulate timer.
memory_status <- function() {
  lines <- tryCatch(readLines("/proc/self/status"), error = function(e) character(),
                    warning = function(w) character())
  kib <- function(key) {
    line <- grep(paste0("^", key, ":"), lines, value = TRUE)
    if (length(line)) as.numeric(gsub("[^0-9]", "", line)) * 1024 else NA_real_
  }
  c(rss = kib("VmRSS"), peak = kib("VmHWM"))
}
# Writing 5 to clear_refs resets VmHWM to the current RSS (Linux 4.0 and later).
reset_peak <- function() {
  tryCatch({
    cat(5, file = "/proc/self/clear_refs")
    TRUE
  }, error = function(e) FALSE, warning = function(w) FALSE)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
number <- function(name) as.numeric(args[[name]])
integer <- function(name) as.integer(args[[name]])

n <- integer("n")
days <- integer("days")
mean_degree <- number("mean_degree")
target_r0 <- number("target_r0")
latent_days <- number("latent_days")
infectious_days <- number("infectious_days")
hospital_probability <- number("hospitalization_probability")
hospital_days <- number("hospital_days")
vaccine_coverage <- number("vaccine_coverage")
vaccine_efficacy <- number("vaccine_efficacy")
recovery <- 1 / infectious_days
transmissibility <- min(0.999, target_r0 / max(1, mean_degree - 1))
beta <- transmissibility * recovery / (1 - transmissibility * (1 - recovery))
beta <- beta * number("transmission_multiplier")

memory_baseline <- memory_status()
total_started <- proc.time()[["elapsed"]]
edges <- utils::read.delim(gzfile(args$network), colClasses = "integer")
if (nrow(edges) != integer("network_edges")) stop("Network edge count mismatch")
read_seconds <- proc.time()[["elapsed"]] - total_started
memory_after_read <- memory_status()

# individual has no contact networks, so the network is an adjacency list:
# agent i's neighbours are neighbours[first[i] + 0:(degree[i] - 1)].
from <- c(edges$source, edges$target) + 1L
to <- c(edges$target, edges$source) + 1L
neighbours <- to[order(from, method = "radix")]
degree <- tabulate(from, nbins = n)
first <- cumsum(c(1L, degree))[seq_len(n)]
neighbours_of <- function(agents) neighbours[sequence(degree[agents], from = first[agents])]
rm(edges, from, to)
invisible(gc(FALSE))

# The initial conditions differ in every run, so building them is timed with
# the simulation, as epiworld seeds its infections inside run().
memory_after_setup <- memory_status()
peak_reset <- reset_peak()
simulate_started <- proc.time()[["elapsed"]]
set.seed(integer("seed"))
states <- c("S", "E", "I", "H", "R")
initial <- rep("S", n)
initial[sample.int(n, min(integer("initial_infected"), n))] <- "I"
health <- CategoricalVariable$new(categories = states, initial_values = initial)

# All-or-nothing vaccine. individual has no interventions, so the protected
# agents are a Bitset that the infection process leaves out; unprotected
# vaccinees behave exactly like unvaccinated agents.
vaccinated <- sample.int(n, round(vaccine_coverage * n))
protected <- Bitset$new(n)$insert(vaccinated[stats::runif(length(vaccinated)) < vaccine_efficacy])
unprotected <- protected$copy()$not(inplace = TRUE)

# A susceptible agent with k infectious neighbours is infected with
# probability 1 - (1 - beta)^k, which is epiworld's per-contact rule.
infection_process <- function(t) {
  contacts <- tabulate(neighbours_of(health$get_index_of("I")$to_vector()), nbins = n)
  exposed <- health$get_index_of("S")$and(unprotected)$and(Bitset$new(n)$insert(which(contacts > 0L)))
  exposed$sample(1 - (1 - beta)^contacts[exposed$to_vector()])
  health$queue_update("E", exposed)
}
render <- Render$new(timesteps = days)
processes <- list(
  infection_process,
  bernoulli_process(health, "E", "I", 1 / latent_days),
  # Leave I at the recovery rate, for hospital with the lifetime probability.
  fixed_probability_multinomial_process(
    health, "I", c("H", "R"), recovery, c(hospital_probability, 1 - hospital_probability)
  ),
  bernoulli_process(health, "H", "R", 1 / hospital_days),
  categorical_count_renderer_process(render, health, "H")
)
simulation_loop(variables = list(health), processes = processes, timesteps = days)
simulate_seconds <- proc.time()[["elapsed"]] - simulate_started
memory_simulate <- memory_status()
total_seconds <- proc.time()[["elapsed"]] - total_started

final <- vapply(states, health$get_size_of, numeric(1))
# The renderer records the counts at the start of each day, before its
# updates, so the final day's count is added separately.
peak_hospitalized <- max(render$to_dataframe()$H_count, final[["H"]])

record <- list(
  status = "ok",
  engine = "individual",
  engine_version = args$engine_version,
  n = n,
  days = days,
  replicate = integer("replicate"),
  seed = integer("seed"),
  network_sha256 = args$network_sha256,
  network_edges = integer("network_edges"),
  mean_degree = mean_degree,
  target_r0 = target_r0,
  transmission_multiplier = number("transmission_multiplier"),
  read_seconds = read_seconds,
  setup_seconds = total_seconds - simulate_seconds,
  simulate_seconds = simulate_seconds,
  total_seconds = total_seconds,
  rss_baseline_bytes = memory_baseline[["rss"]],
  rss_after_read_bytes = memory_after_read[["rss"]],
  rss_after_setup_bytes = memory_after_setup[["rss"]],
  peak_rss_setup_bytes = memory_after_setup[["peak"]],
  peak_rss_simulate_bytes = if (peak_reset) memory_simulate[["peak"]] else NA_real_,
  final_susceptible = final[["S"]],
  final_exposed = final[["E"]],
  final_infected = final[["I"]],
  final_hospitalized = final[["H"]],
  final_recovered = final[["R"]],
  peak_hospitalized = peak_hospitalized,
  vaccinated = length(vaccinated),
  vaccine_protected = protected$size(),
  fingerprint = args$fingerprint,
  timestamp_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)

stopifnot(sum(unlist(record[c(
  "final_susceptible", "final_exposed", "final_infected",
  "final_hospitalized", "final_recovered"
)])) == n)

dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
temporary <- tempfile(pattern = paste0(".", basename(args$output), "."), tmpdir = dirname(args$output))
write_json(record, temporary, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null")
if (!file.rename(temporary, args$output)) stop("Could not atomically install result file")
