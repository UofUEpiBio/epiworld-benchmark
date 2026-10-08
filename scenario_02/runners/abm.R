#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ABM)
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
# ABM runs in continuous time, so the rates follow EoN's mapping from R0 to the
# per-contact transmission rate, with the competing hospitalization rate.
transmissibility <- min(0.999, target_r0 / max(1, mean_degree - 1))
tau <- transmissibility * recovery / (1 - transmissibility)
tau <- tau * number("transmission_multiplier")
hospital_rate <- hospital_probability / (1 - hospital_probability) * recovery

memory_baseline <- memory_status()
total_started <- proc.time()[["elapsed"]]
edges <- utils::read.delim(gzfile(args$network), colClasses = "integer")
if (nrow(edges) != integer("network_edges")) stop("Network edge count mismatch")
read_seconds <- proc.time()[["elapsed"]] - total_started
memory_after_read <- memory_status()

# ABM has no way to load a network, so the edge list is an adjacency list, as
# in the individual runner, served to ABM through a Contact subclass: agent i's
# neighbours are neighbours[first[i] + 0:(degree[i] - 1)].
from <- c(edges$source, edges$target) + 1L
to <- c(edges$target, edges$source) + 1L
neighbours <- to[order(from, method = "radix")]
degree <- tabulate(from, nbins = n)
first <- cumsum(c(1L, degree))[seq_len(n)]
rm(edges, from, to)
NetworkContact <- R6::R6Class("NetworkContact", inherit = Contact,
  public = list(
    # The agents ABM hands to contact() are external pointers, and agent IDs
    # run consecutively from the first agent's.
    build = function() {
      private$agents <- lapply(seq_len(n), function(i) getAgent(private$population, i))
      private$offset <- getID(private$agents[[1L]]) - 1L
    },
    index = function(agent) getID(agent) - private$offset,
    states = function() vapply(private$agents, function(a) unlist(getState(a))[[1L]], ""),
    contact = function(time, agent) {
      i <- getID(agent) - private$offset
      private$agents[neighbours[first[i] + seq_len(degree[i]) - 1L]]
    }
  ),
  private = list(agents = NULL, offset = NULL)
)
invisible(gc(FALSE))

# The initial conditions differ in every run, so building them is timed with
# the simulation, as epiworld seeds its infections inside run().
memory_after_setup <- memory_status()
peak_reset <- reset_peak()
simulate_started <- proc.time()[["elapsed"]]
set.seed(integer("seed"))
states <- c("S", "E", "I", "H", "R", "V")
initial <- rep("S", n)
initial[sample.int(n, min(integer("initial_infected"), n))] <- "I"
# All-or-nothing vaccine. ABM has no interventions, so protected agents sit in
# a state V that no transition touches; unprotected vaccinees behave exactly
# like unvaccinated agents. Protected agents who are already infected stay so.
vaccinated <- sample.int(n, round(vaccine_coverage * n))
protected <- vaccinated[stats::runif(length(vaccinated)) < vaccine_efficacy]
initial[protected[initial[protected] == "S"]] <- "V"
# The transmission tree, which ABM does not keep: the day each agent was
# infected (the day whose counts include it) and its source, -1 for the seed
# cases.
infection_day <- rep(-1L, n)
infection_source <- rep(NA_integer_, n)
seeds <- which(initial == "I")
infection_day[seeds] <- 0L
infection_source[seeds] <- -1L
# ABM's initializer must return states that are already lists: given a
# character state it builds a temporary list that its garbage-collection
# handling can lose, which corrupts the state of an agent now and then. The
# agents share these lists. The initializer's index starts at 0.
state_lists <- lapply(stats::setNames(nm = states), list)
sim <- Simulation$new(n, function(i) state_lists[[initial[i + 1L]]])
network <- NetworkContact$new()
sim$addContact(network)
record_infection <- function(time, agent, contact) {
  target <- network$index(contact)
  infection_day[target] <<- as.integer(ceiling(time))
  infection_source[target] <<- network$index(agent)
}
# The counter reports the hospitalized agents at the end of each day.
sim$addLogger(newCounter("H", "H"))
# Daily transition counts, which ABM does not keep either: a counter with a
# from and a to state counts the transitions between two reports.
pairs <- c("SE", "EI", "IH", "IR", "HR")
# A counter keeps an unprotected reference to its "to" state, which ABM's
# garbage collection can then free, so these lists stay referenced here.
pair_from <- lapply(stats::setNames(pairs, pairs), function(pair) list(substr(pair, 1, 1)))
pair_to <- lapply(stats::setNames(pairs, pairs), function(pair) list(substr(pair, 2, 2)))
for (pair in pairs) sim$addLogger(newCounter(pair, pair_from[[pair]], pair_to[[pair]]))
sim$addTransition("I" + "S" -> "I" + "E" ~ network, newExpWaitingTime(tau), changed_callback = record_infection)
sim$addTransition("E" -> "I", newExpWaitingTime(1 / latent_days))
sim$addTransition("I" -> "H", newExpWaitingTime(hospital_rate))
sim$addTransition("I" -> "R", newExpWaitingTime(recovery))
sim$addTransition("H" -> "R", newExpWaitingTime(1 / hospital_days))
trajectory <- sim$run(0:days)
simulate_seconds <- proc.time()[["elapsed"]] - simulate_started
memory_simulate <- memory_status()

# Extracting the outputs comes after the simulation and is timed apart. The
# first row of the trajectory is day 0, which has no transitions.
extract_started <- proc.time()[["elapsed"]]
transitions <- as.matrix(trajectory[-1L, pairs])
infected <- which(infection_day >= 0L)
tree <- data.frame(
  source = infection_source[infected], target = infected, day = infection_day[infected]
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

# The final compartments are counted from the agents themselves, outside the
# simulate timer, like individual's.
final_states <- network$states()
final <- vapply(states, function(state) sum(final_states == state), numeric(1))
final[["S"]] <- final[["S"]] + final[["V"]]
# The peak is the largest end-of-day count of hospitalized agents.
peak_hospitalized <- max(trajectory$H)

record <- list(
  status = "ok",
  engine = "ABM",
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
  final_susceptible = final[["S"]],
  final_exposed = final[["E"]],
  final_infected = final[["I"]],
  final_hospitalized = final[["H"]],
  final_recovered = final[["R"]],
  peak_hospitalized = peak_hospitalized,
  vaccinated = length(vaccinated),
  vaccine_protected = length(protected),
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
