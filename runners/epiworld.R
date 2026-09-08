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

total_started <- proc.time()[["elapsed"]]
edges <- utils::read.delim(gzfile(args$network), colClasses = "integer")
if (nrow(edges) != integer("network_edges")) stop("Network edge count mismatch")

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
  virus_set_state(init = 2L, end = 4L, removed = 4L) |>
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

simulate_started <- proc.time()[["elapsed"]]
run(model, ndays = days, seed = integer("seed"))
simulate_seconds <- proc.time()[["elapsed"]] - simulate_started
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
  setup_seconds = total_seconds - simulate_seconds,
  simulate_seconds = simulate_seconds,
  total_seconds = total_seconds,
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
write_json(record, temporary, auto_unbox = TRUE, pretty = TRUE, digits = NA)
if (!file.rename(temporary, args$output)) stop("Could not atomically install result file")
