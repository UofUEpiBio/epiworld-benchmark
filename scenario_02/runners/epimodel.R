#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(EpiModel)
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

# EpiModel draws its networks from a fitted ERGM. The benchmark's network is
# fixed, so the formation model is a placeholder: an edges term held at the
# shared network's density, which leaves ergm nothing to estimate. The network
# is never drawn from it; the initialization module below installs the edge
# list itself, in the tergmLite edge-list form EpiModel simulates on.
estimate <- suppressMessages(suppressWarnings(netest(
  network_initialize(n), formation = ~offset(edges),
  coef.form = stats::qlogis(nrow(edges) / choose(n, 2)), target.stats = NULL,
  coef.diss = dissolution_coefs(~offset(edges), duration = 1)
)))
network <- as.edgelist(cbind(edges$source, edges$target) + 1L, n = n, directed = FALSE)
rm(edges)
invisible(gc(FALSE))

# initialize.net without its first network draw: the shared edge list stands
# in for it, and the seed cases are EpiModel's own (init.net's i.num).
initialize_fixed <- function(x, param, init, control, s) {
  dat <- create_dat_object(param, init, control)
  dat <- init_nets(dat, x)
  dat$run$el[[1L]] <- network
  dat <- init_status.net(dat)
  vaccinate(dat)
}

# All-or-nothing vaccine on day 0, as attributes in the style of the EpiModel
# Gallery's vaccination examples: the infection module skips protected
# agents, and unprotected vaccinees behave exactly like unvaccinated agents.
# The day-0 epidemic statistics keep the counts.
vaccinate <- function(dat) {
  vaccinated <- sample.int(n, round(get_param(dat, "vaccine_coverage") * n))
  protected <- vaccinated[stats::runif(length(vaccinated)) < get_param(dat, "vaccine_efficacy")]
  protection <- logical(n)
  protection[protected] <- TRUE
  dat <- set_attr(dat, "protected", protection)
  dat <- set_epi(dat, "vaccinated", 1L, length(vaccinated))
  dat <- set_epi(dat, "vaccine_protected", 1L, length(protected))
  prevalence(dat, at = 1L)
}

# A susceptible agent with k infectious neighbours is infected with
# probability 1 - (1 - beta)^k, which is epiworld's per-contact rule:
# discord_edgelist() lists every susceptible-infectious edge. EpiModel keeps
# the transmission tree in its transmission matrix, whose source is one of the
# transmitting neighbours, chosen at random as epiworld does (the edge list
# comes shuffled), and in each agent's infTime.
infect <- function(dat, at) {
  pairs <- discord_edgelist(dat, at)
  if (!is.null(pairs)) pairs <- pairs[!get_attr(dat, "protected")[pairs$sus], ]
  if (NROW(pairs) == 0L) return(set_epi(dat, "se.flow", at, 0))
  transmitted <- pairs[stats::runif(nrow(pairs)) < get_param(dat, "beta"), ]
  exposed <- unique(transmitted$sus)
  status <- get_attr(dat, "status")
  status[exposed] <- "e"
  dat <- set_attr(dat, "status", status)
  infection_time <- get_attr(dat, "infTime")
  infection_time[exposed] <- at
  dat <- set_attr(dat, "infTime", infection_time)
  if (length(exposed)) dat <- set_transmat(dat, transmitted, at)
  set_epi(dat, "se.flow", at, length(exposed))
}

# Every transition is drawn from the states at the start of the step. Agents
# leave I at the recovery rate, for H with the lifetime probability.
progress <- function(dat, at) {
  status <- get_attr(dat, "status")
  latent <- which(status == "e")
  infectious <- which(status == "i")
  hospitalized <- which(status == "h")
  onset <- latent[stats::runif(length(latent)) < 1 / get_param(dat, "latent_days")]
  leave <- infectious[stats::runif(length(infectious)) < get_param(dat, "recovery")]
  admitted <- stats::runif(length(leave)) < get_param(dat, "hospital_probability")
  discharged <- hospitalized[stats::runif(length(hospitalized)) < 1 / get_param(dat, "hospital_days")]
  status[onset] <- "i"
  status[leave[admitted]] <- "h"
  status[leave[!admitted]] <- "r"
  status[discharged] <- "r"
  # The daily transition counts, as epidemic statistics.
  dat <- set_epi(dat, "ei.flow", at, length(onset))
  dat <- set_epi(dat, "ih.flow", at, sum(admitted))
  dat <- set_epi(dat, "ir.flow", at, sum(!admitted))
  dat <- set_epi(dat, "hr.flow", at, length(discharged))
  set_attr(dat, "status", status)
}

states <- c("s", "e", "i", "h", "r")
prevalence <- function(dat, at) {
  status <- get_attr(dat, "status")
  for (state in states) dat <- set_epi(dat, paste0(state, ".num"), at, sum(status == state))
  dat
}

param <- param.net(
  beta = beta, latent_days = latent_days, recovery = recovery,
  hospital_probability = hospital_probability, hospital_days = hospital_days,
  vaccine_coverage = vaccine_coverage, vaccine_efficacy = vaccine_efficacy
)
init <- init.net(i.num = min(integer("initial_infected"), n))
# Step 1 is day 0, so days steps follow it. User modules (progress) run before
# the built-in ones (infection, then prevalence) within each step. The network
# is fixed, so resimulating it does nothing; tergmLite makes EpiModel keep it
# as an edge list rather than a networkDynamic object.
control <- suppressMessages(control.net(
  type = NULL, nsims = 1, ncores = 1, nsteps = days + 1L, tergmLite = TRUE,
  initialize.FUN = initialize_fixed, resim_nets.FUN = function(dat, at) dat,
  infection.FUN = infect, progress.FUN = progress, prevalence.FUN = prevalence,
  save.nwstats = FALSE, save.network = FALSE, save.transmat = TRUE, save.run = TRUE, verbose = FALSE
))

# The initial conditions differ in every run, so building them is timed with
# the simulation, as epiworld seeds its infections inside run(). netsim()
# builds them in its initialization module.
memory_after_setup <- memory_status()
peak_reset <- reset_peak()
simulate_started <- proc.time()[["elapsed"]]
set.seed(integer("seed"))
sim <- netsim(estimate, param, init, control)
simulate_seconds <- proc.time()[["elapsed"]] - simulate_started
memory_simulate <- memory_status()

# The final compartments are the prevalence module's counts on the last day;
# protected agents never leave S.
trajectory <- as.data.frame(sim)
final <- vapply(states, function(state) trajectory[[paste0(state, ".num")]][days + 1L], numeric(1))
# The peak is the largest end-of-day count of hospitalized agents.
peak_hospitalized <- max(trajectory$h.num)

# Extracting the outputs comes after the simulation and is timed apart. Step
# at is day at - 1; the seed cases are those infected at step 1.
extract_started <- proc.time()[["elapsed"]]
pairs <- c("SE", "EI", "IH", "IR", "HR")
transitions <- as.matrix(trajectory[-1L, paste0(tolower(pairs), ".flow")])
colnames(transitions) <- pairs
# get_transmat() fails on a run without transmissions.
transmissions <- if (sum(transitions[, "SE"]) > 0) {
  get_transmat(sim)
} else {
  data.frame(at = 0L[0], sus = 0L[0], inf = 0L[0])
}
seeds <- which(sim$run[[1L]]$attr$infTime == 1)
tree <- data.frame(
  source = c(rep(-1L, length(seeds)), transmissions$inf),
  target = c(seeds, transmissions$sus),
  day = c(rep(0L, length(seeds)), transmissions$at - 1L)
)
daily_incidence <- transitions[, "SE"]
# epiworld's reproductive number: the mean number of secondary infections
# caused by the cases infected on each day.
secondary <- tabulate(tree$source[tree$source != -1L], nbins = n)
repnum <- tapply(secondary[tree$target], tree$day, mean)
reproductive_number <- rep(NA_real_, days + 1L)
reproductive_number[as.integer(names(repnum)) + 1L] <- unname(repnum)
extract_seconds <- proc.time()[["elapsed"]] - extract_started
total_seconds <- proc.time()[["elapsed"]] - total_started

record <- list(
  status = "ok",
  engine = "EpiModel",
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
  setup_seconds = total_seconds - simulate_seconds - extract_seconds,
  simulate_seconds = simulate_seconds,
  total_seconds = total_seconds,
  rss_baseline_bytes = memory_baseline[["rss"]],
  rss_after_read_bytes = memory_after_read[["rss"]],
  rss_after_setup_bytes = memory_after_setup[["rss"]],
  peak_rss_setup_bytes = memory_after_setup[["peak"]],
  peak_rss_simulate_bytes = if (peak_reset) memory_simulate[["peak"]] else NA_real_,
  final_susceptible = final[["s"]],
  final_exposed = final[["e"]],
  final_infected = final[["i"]],
  final_hospitalized = final[["h"]],
  final_recovered = final[["r"]],
  peak_hospitalized = peak_hospitalized,
  vaccinated = trajectory$vaccinated[1L],
  vaccine_protected = trajectory$vaccine_protected[1L],
  extract_seconds = extract_seconds,
  transmissions = sum(tree$source != -1L),
  transitions_se = sum(transitions[, "SE"]),
  transitions_ei = sum(transitions[, "EI"]),
  transitions_ih = sum(transitions[, "IH"]),
  transitions_ir = sum(transitions[, "IR"]),
  transitions_hr = sum(transitions[, "HR"]),
  daily_incidence = unname(daily_incidence),
  reproductive_number = reproductive_number,
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
