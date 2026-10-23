#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(epiworldR)
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
recovery <- 1 / infectious_days
transmissibility <- min(0.999, target_r0 / max(1, mean_degree - 1))
beta <- transmissibility * recovery / (1 - transmissibility * (1 - recovery))
beta <- beta * number("transmission_multiplier")
hospital_rate <- hospital_probability * recovery /
  (1 - hospital_probability * (1 - recovery))

memory_baseline <- memory_status()
total_started <- proc.time()[["elapsed"]]
edges <- utils::read.delim(gzfile(args$network), colClasses = "integer")
if (nrow(edges) != integer("network_edges")) stop("Network edge count mismatch")
read_seconds <- proc.time()[["elapsed"]] - total_started
memory_after_read <- memory_status()

model <- Model() |>
  add_param("Transmission rate", beta) |>
  add_param("Incubation rate", 1 / latent_days) |>
  add_param("Hospitalization rate", hospital_rate) |>
  add_param("Recovery rate", recovery) |>
  add_param("Hospital recovery rate", 1 / hospital_days)

update_s <- update_fun_susceptible(exclude = c(1L, 3L, 4L))
update_e <- update_fun_rate("Incubation rate", 2L)
update_i <- update_fun_rate(
  c("Hospitalization rate", "Recovery rate"),
  c(3L, 4L)
)
update_h <- update_fun_rate("Hospital recovery rate", 4L)

model |>
  add_state("Susceptible", update_s) |>
  add_state("Exposed", update_e) |>
  add_state("Infected", update_i) |>
  add_state("Hospitalized", update_h) |>
  add_state("Recovered", NULL)

pathogen <- virus(
  name = "Benchmark pathogen",
  prevalence = min(integer("initial_infected"), n),
  as_proportion = FALSE,
  prob_infecting = beta,
  recovery_rate = recovery
) |>
  # New infections enter Exposed (state 1). The R API offers no way to seed a
  # custom model's initial cases in a different state, so the initial cases
  # start Exposed too, unlike the other engines, which seed them Infected.
  virus_set_state(init = 1L, end = 4L, removed = 4L) |>
  set_prob_infecting_ptr(model, "Transmission rate") |>
  set_distribution_virus(distribute_virus_randomly(
    min(integer("initial_infected"), n), FALSE
  ))

add_virus(model, pathogen)
agents_from_edgelist(
  model,
  source = edges$source,
  target = edges$target,
  size = n,
  directed = FALSE
)
verbose_off(model)
rm(edges)
invisible(gc(FALSE))

memory_after_setup <- memory_status()
peak_reset <- reset_peak()
simulate_started <- proc.time()[["elapsed"]]
run(model, ndays = days, seed = integer("seed"))
simulate_seconds <- proc.time()[["elapsed"]] - simulate_started
memory_simulate <- memory_status()
total_seconds <- proc.time()[["elapsed"]] - total_started

today <- get_today_total(model)
names(today) <- get_states(model)
history <- get_hist_total(model)
hospital_history <- history[history$state == "Hospitalized", , drop = FALSE]
peak_hospitalized <- if (nrow(hospital_history)) max(hospital_history$counts) else 0L

record <- list(
  status = "ok",
  engine = "epiworldR",
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
  final_susceptible = unname(today[["Susceptible"]]),
  final_exposed = unname(today[["Exposed"]]),
  final_infected = unname(today[["Infected"]]),
  final_hospitalized = unname(today[["Hospitalized"]]),
  final_recovered = unname(today[["Recovered"]]),
  peak_hospitalized = peak_hospitalized,
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
