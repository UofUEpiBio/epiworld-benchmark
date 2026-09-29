# Shared helpers for the per-scenario reports (scenario_*/README.qmd).
#
# Quarto runs each report from its scenario folder, so paths default to the
# repository root one level up. Each function returns a knitr table or a
# ggplot object, or NULL when there is nothing to show yet.

# "epiworld" is the C++ library; epiworldR and epiworldpy wrap it.
engine_levels <- c(
  "epiworld", "epiworldR", "epiworldpy", "covasim", "starsim", "EoN", "epydemic", "ixa",
  "individual", "FRED", "Agents.jl"
)
# 165,865 is scenario 04's GeoPops population once isolated people are dropped.
size_levels <- c(10000, 100000, 165865, 1000000)
full_days <- 100

#' "10,000 agents" and so on, as a factor in size order.
agents_factor <- function(n, suffix = " agents") {
  sizes <- size_levels[size_levels %in% n]
  factor(n, levels = sizes, labels = paste0(format(sizes, big.mark = ",", scientific = FALSE,
                                                   trim = TRUE), suffix))
}

options(scipen = 999, knitr.kable.NA = "")

q <- function(x, p) unname(stats::quantile(x, p, na.rm = TRUE))

#' Load results for one scenario, plus the full-design rows of every scenario
#' (needed for comparisons against another scenario).
load_benchmark <- function(scenario, root = "..") {
  results_path <- file.path(root, "results", "results.csv")
  manifest_path <- file.path(root, "results", "run-manifest.json")
  scenarios_path <- file.path(root, "results", "scenarios.json")
  all_results <- if (file.exists(results_path)) {
    read.csv(results_path, stringsAsFactors = FALSE)
  } else {
    data.frame()
  }
  every <- if (nrow(all_results)) {
    subset(all_results, days == full_days & n %in% size_levels)
  } else {
    all_results
  }
  if (nrow(every)) {
    every$attack_rate <- 1 - every$final_susceptible / every$n
    every$protected_share <- every$vaccine_protected / every$n
  }
  full <- if (nrow(every)) every[every$scenario == scenario, ] else every
  info <- if (file.exists(scenarios_path)) jsonlite::read_json(scenarios_path) else list()
  # Replicates per engine at each size, from the design run.py publishes.
  design <- info[[scenario]]$design
  design <- if (length(design)) {
    data.frame(n = vapply(design, `[[`, 0, "n"),
               replicates = vapply(design, `[[`, 0, "replicates"))
  } else {
    data.frame(n = size_levels[1:2], replicates = 100)
  }
  # Each scenario runs its own engines; run.py publishes them.
  engines <- if (length(info[[scenario]]$engines)) {
    unlist(info[[scenario]]$engines)
  } else {
    engine_levels[1:9]
  }
  counts <- table(factor(full$engine, levels = engines),
                  factor(full$n, levels = design$n))
  complete <- nrow(full) > 0 && all(sweep(counts, 2, design$replicates, `>=`))
  list(
    scenario = scenario,
    root = root,
    all = all_results,
    every = every,
    full = full,
    has_full = nrow(full) > 0,
    complete = complete,
    design = design,
    design_runs = length(engines) * sum(design$replicates),
    manifest = if (file.exists(manifest_path)) {
      jsonlite::read_json(manifest_path, simplifyVector = TRUE)
    } else {
      NULL
    },
    info = info[[scenario]]
  )
}

completion_status <- function(bench) {
  if (!bench$has_full) {
    cat("::: {.callout-warning}\n",
        "No full-profile results are present for this scenario yet. Run ",
        "`make benchmark`, then render this report again.\n:::\n", sep = "")
  } else if (!bench$complete) {
    cat("::: {.callout-warning}\n",
        "The full-profile cache for this scenario is incomplete. Estimates ",
        "below are provisional; rerun `make benchmark` to resume missing cells.\n",
        ":::\n", sep = "")
  } else {
    cat("::: {.callout-tip}\nThe complete ", format(bench$design_runs, big.mark = ","),
        "-run design for this scenario is available.\n:::\n", sep = "")
  }
}

table_environment <- function(bench) {
  m <- bench$manifest
  if (is.null(m)) return(NULL)
  knitr::kable(data.frame(
    Field = c("Platform", "Python", "Workers", "Latest run failures"),
    Value = c(m$platform, m$python,
              if (is.null(bench$info$workers)) m$workers else bench$info$workers,
              m$failures)
  ), row.names = FALSE)
}

table_summary <- function(bench) {
  full <- bench$full
  if (!nrow(full)) return(NULL)
  rows <- do.call(rbind, lapply(split(full, list(full$engine, full$n), drop = TRUE),
    function(d) data.frame(
      engine = d$engine[[1]],
      agents = d$n[[1]],
      replicates = nrow(d),
      median = median(d$simulate_seconds),
      q25 = q(d$simulate_seconds, 0.25),
      q75 = q(d$simulate_seconds, 0.75)
    )))
  rows <- rows[order(rows$agents, rows$median), ]
  knitr::kable(rows, digits = 3,
    col.names = c("Engine", "Agents", "Runs", "Median simulation (s)", "Q1 (s)", "Q3 (s)"),
    row.names = FALSE)
}

plot_simulation_time <- function(bench) {
  full <- bench$full
  if (!nrow(full)) return(NULL)
  # Reversed so the first engine_levels entry lands at the top row: ggplot
  # draws a discrete y-axis's first factor level at the bottom.
  full$engine <- factor(full$engine, levels = rev(engine_levels))
  full$agents <- agents_factor(full$n)
  ggplot2::ggplot(full, ggplot2::aes(simulate_seconds, engine, fill = engine)) +
    ggridges::geom_density_ridges(
      scale = 0.75, rel_min_height = 0.01, alpha = 0.7,
      jittered_points = TRUE, point_alpha = 0.5, point_size = 1.2,
      position = ggridges::position_points_jitter(width = 0, height = 0.25)
    ) +
    ggplot2::facet_wrap(~ agents, ncol = 1, scales = "free_x") +
    ggplot2::scale_x_log10() +
    ggplot2::labs(
      x = "Simulation wall time (seconds, log scale)",
      y = NULL,
      subtitle = "Wall time distribution for 100-day simulations",
      caption = "Note: dots show individual replicates"
    ) +
    ggplot2::theme_minimal(base_size = 18) +
    ggplot2::theme(legend.position = "none")
}

#' Median time of each phase of a run. Reading the edge list is benchmark
#' plumbing whose cost depends on the language, not the engine, so it is shown
#' but left out of "build + simulate", the time to a first result once the
#' network is in memory.
table_phases <- function(bench) {
  full <- bench$full
  if (!nrow(full) || all(is.na(full$read_seconds))) return(NULL)
  full$build_seconds <- full$setup_seconds - full$read_seconds
  rows <- do.call(rbind, lapply(split(full, list(full$engine, full$n), drop = TRUE),
    function(d) data.frame(
      engine = d$engine[[1]],
      agents = d$n[[1]],
      read = median(d$read_seconds),
      build = median(d$build_seconds),
      simulate = median(d$simulate_seconds),
      result = median(d$build_seconds + d$simulate_seconds)
    )))
  rows <- rows[order(rows$agents, rows$result), ]
  knitr::kable(rows, digits = 3,
    col.names = c("Engine", "Agents", "Read edges (s)", "Build (s)", "Simulate (s)",
                  "Build + simulate (s)"),
    row.names = FALSE)
}

#' Paired simulate-time ratio of each engine against `reference`, matched by
#' size and seed. `engines` restricts the rows (default: every other engine).
table_speedup <- function(bench, reference = "epiworldR", engines = NULL) {
  full <- bench$full
  if (!nrow(full) || !any(full$engine == reference)) return(NULL)
  base <- full[full$engine == reference, c("n", "seed", "simulate_seconds")]
  names(base)[3] <- "reference_seconds"
  others <- full[full$engine != reference, ]
  if (!is.null(engines)) others <- others[others$engine %in% engines, ]
  if (!nrow(others)) return(NULL)
  ratios <- merge(others, base, by = c("n", "seed"))
  ratios$ratio <- ratios$simulate_seconds / ratios$reference_seconds
  rows <- do.call(rbind, lapply(split(ratios, list(ratios$engine, ratios$n), drop = TRUE),
    function(d) data.frame(
      engine = d$engine[[1]],
      agents = d$n[[1]],
      median = median(d$ratio),
      q25 = q(d$ratio, .25),
      q75 = q(d$ratio, .75)
    )))
  rows <- rows[order(rows$agents, factor(rows$engine, levels = engine_levels)), ]
  knitr::kable(rows, digits = 2,
    col.names = c("Engine", "Agents", paste("Median time /", reference), "Q1", "Q3"),
    row.names = FALSE)
}

outcome_medians <- function(data) {
  if (!nrow(data)) return(data.frame())
  rows <- do.call(rbind, lapply(split(data, list(data$scenario, data$engine, data$n), drop = TRUE),
    function(d) data.frame(
      scenario = d$scenario[[1]],
      engine = d$engine[[1]],
      n = d$n[[1]],
      attack_rate = median(d$attack_rate),
      peak_hospitalized = median(d$peak_hospitalized),
      protected_share = if (all(is.na(d$protected_share))) NA else
        median(d$protected_share, na.rm = TRUE)
    )))
  rows[order(rows$scenario, rows$n, rows$engine), ]
}

table_outcomes <- function(bench) {
  rows <- outcome_medians(bench$full)
  if (!nrow(rows)) return(NULL)
  columns <- c("engine", "n", "attack_rate", "peak_hospitalized")
  names <- c("Engine", "Agents", "Median final attack rate", "Median peak hospitalized")
  if (!all(is.na(rows$protected_share))) {
    columns <- c(columns, "protected_share")
    names <- c(names, "Median share protected")
  }
  knitr::kable(rows[columns], digits = 3, col.names = names, row.names = FALSE)
}

plot_outcomes <- function(bench, subtitle) {
  rows <- outcome_medians(bench$full)
  if (!nrow(rows)) return(NULL)
  rows$engine <- factor(rows$engine, levels = engine_levels)
  rows$agents <- agents_factor(rows$n, suffix = "")
  ggplot2::ggplot(rows, ggplot2::aes(engine, attack_rate, fill = engine)) +
    ggplot2::geom_col(width = 0.65) +
    ggplot2::facet_wrap(~ agents, scales = "free_y") +
    ggplot2::scale_y_continuous(limits = c(0, NA)) +
    ggplot2::labs(x = NULL, y = "Median final attack rate", subtitle = subtitle) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(legend.position = "none")
}

#' Paired simulate-time ratio of this scenario against `baseline`, matched by
#' engine, size, and seed (seeds do not depend on the scenario).
table_time_versus <- function(bench, baseline) {
  every <- bench$every
  if (!nrow(bench$full) || !any(every$scenario == baseline)) return(NULL)
  base <- every[every$scenario == baseline, c("engine", "n", "seed", "simulate_seconds")]
  names(base)[4] <- "baseline_seconds"
  paired <- merge(bench$full, base, by = c("engine", "n", "seed"))
  paired$ratio <- paired$simulate_seconds / paired$baseline_seconds
  rows <- do.call(rbind, lapply(split(paired, list(paired$engine, paired$n), drop = TRUE),
    function(d) data.frame(
      engine = d$engine[[1]],
      agents = d$n[[1]],
      baseline = median(d$baseline_seconds),
      this = median(d$simulate_seconds),
      median = median(d$ratio),
      q25 = q(d$ratio, .25),
      q75 = q(d$ratio, .75)
    )))
  rows <- rows[order(rows$agents, rows$median), ]
  label <- sub("scenario_", "scenario ", baseline)
  this <- sub("scenario_", "scenario ", bench$scenario)
  knitr::kable(rows, digits = c(0, 0, 3, 3, 2, 2, 2),
    col.names = c("Engine", "Agents", paste("Median", label, "(s)"),
                  paste("Median", this, "(s)"), paste("Median time /", label), "Q1", "Q3"),
    row.names = FALSE)
}

#' Median simulation time at every size of this scenario and `baseline`, which
#' runs the same model at other sizes, with the growth from the baseline's
#' largest size to this scenario's smallest, and this scenario's build time
#' (setup without reading the edge list).
table_scaling <- function(bench, baseline) {
  if (!nrow(bench$full)) return(NULL)
  every <- bench$every[bench$every$scenario %in% c(baseline, bench$scenario), ]
  medians <- tapply(every$simulate_seconds, list(every$engine, every$n), median)
  sizes <- as.numeric(colnames(medians))
  before <- max(sizes[sizes < min(bench$full$n)])
  after <- min(bench$full$n)
  setup <- tapply(bench$full$setup_seconds - bench$full$read_seconds, bench$full$engine, median)
  rows <- data.frame(
    engine = rownames(medians),
    medians,
    growth = medians[, as.character(after)] / medians[, as.character(before)],
    setup = setup[rownames(medians)],
    check.names = FALSE
  )
  rows <- rows[order(rows[[as.character(after)]]), ]
  size_label <- function(n) format(n, big.mark = ",", scientific = FALSE, trim = TRUE)
  knitr::kable(rows, digits = c(0, rep(3, length(sizes)), 1, 3), row.names = FALSE, col.names = c(
    "Engine", paste(size_label(sizes), "agents (s)"),
    paste0("Time at ", size_label(after), " / at ", size_label(before)),
    paste0("Median build at ", size_label(after), " (s)")
  ))
}

#' Count the model lines of one engine's runner. Each region runs from the
#' first line matching `from` to the next line matching `to` (both inclusive).
count_model_lines <- function(spec, scenario_dir) {
  path <- file.path(scenario_dir, spec$files[[1]])
  lines <- readLines(path, warn = FALSE)
  keep <- logical(length(lines))
  for (r in spec$regions) {
    to <- if (is.null(r$to)) r$from else r$to
    start <- grep(r$from, lines)[1]
    if (is.na(start)) stop("Anchor not found in ", path, ": ", r$from)
    end <- start - 1L + grep(to, lines[start:length(lines)])[1]
    if (is.na(end)) stop("Anchor not found in ", path, ": ", to)
    keep[start:end] <- TRUE
  }
  code <- trimws(lines[keep])
  timing <- "perf_counter|proc\\.time|Instant::now|elapsed"
  sum(nzchar(code) & !grepl("^(#|//|\"\"\")", code) & !grepl(timing, code))
}

code_effort <- function(scenario, root = "..") {
  dir <- file.path(root, scenario)
  specs <- yaml::read_yaml(file.path(dir, "code_regions.yml"))
  data.frame(
    engine = names(specs),
    language = vapply(specs, `[[`, "", "language"),
    files = vapply(specs, function(x) length(x$files), 0L),
    model_lines = vapply(specs, count_model_lines, 0L, scenario_dir = dir),
    row.names = NULL
  )
}

#' Model-line table for this scenario; with `baseline`, also the lines added.
table_code_effort <- function(bench, baseline = NULL) {
  effort <- code_effort(bench$scenario, bench$root)
  names <- c("Engine", "Language", "Files", "Model lines")
  if (!is.null(baseline)) {
    base <- code_effort(baseline, bench$root)[c("engine", "model_lines")]
    names(base)[2] <- "baseline_lines"
    effort <- merge(effort, base, by = "engine")
    effort$added <- effort$model_lines - effort$baseline_lines
    effort <- effort[c("engine", "language", "files", "baseline_lines", "model_lines", "added")]
    label <- sub("scenario_", "scenario ", baseline)
    this <- sub("scenario_", "scenario ", bench$scenario)
    names <- c("Engine", "Language", "Files", paste("Lines,", label),
               paste("Lines,", this), paste("Added since", label))
  }
  effort <- effort[order(effort$model_lines), ]
  knitr::kable(effort, col.names = names, row.names = FALSE)
}

table_versions <- function(bench) {
  if (!nrow(bench$full)) return(NULL)
  versions <- unique(bench$full[c("engine", "engine_version")])
  knitr::kable(versions[order(versions$engine), ],
               col.names = c("Engine", "Recorded version"), row.names = FALSE)
}

#' Median daily incidence and reproductive number across replicates, from
#' results/daily.csv (written by run.py for scenarios that report them).
load_daily <- function(bench) {
  path <- file.path(bench$root, "results", "daily.csv")
  if (!file.exists(path)) return(data.frame())
  daily <- read.csv(path, stringsAsFactors = FALSE)
  daily[daily$scenario == bench$scenario & daily$n %in% size_levels, ]
}

#' Time to extract the outputs after the run, which simulate_seconds excludes.
table_extraction <- function(bench) {
  full <- bench$full
  if (!nrow(full) || all(is.na(full$extract_seconds))) return(NULL)
  rows <- do.call(rbind, lapply(split(full, list(full$engine, full$n), drop = TRUE),
    function(d) data.frame(
      engine = d$engine[[1]],
      agents = d$n[[1]],
      simulate = median(d$simulate_seconds),
      extract = median(d$extract_seconds),
      ratio = median(d$extract_seconds / d$simulate_seconds)
    )))
  rows <- rows[order(rows$agents, rows$simulate), ]
  knitr::kable(rows, digits = c(0, 0, 3, 3, 2),
    col.names = c("Engine", "Agents", "Median simulation (s)",
                  "Median extraction (s)", "Median extraction / simulation"),
    row.names = FALSE)
}

#' Checks that each engine's outputs agree with its own final counts: every
#' agent who left S is a seed case or a transmission target, every
#' transmission is an S -> E transition, and every recovered agent came from
#' I or H.
table_output_checks <- function(bench) {
  full <- bench$full
  if (!nrow(full) || all(is.na(full$transmissions))) return(NULL)
  seeds <- as.integer(bench$info$parameters$initial_infected)
  full$tree_matches <- full$transmissions + seeds == full$n - full$final_susceptible
  full$matrix_matches <- full$transitions_se == full$transmissions &
    full$transitions_ir + full$transitions_hr == full$final_recovered
  rows <- do.call(rbind, lapply(split(full, list(full$engine, full$n), drop = TRUE),
    function(d) data.frame(
      engine = d$engine[[1]],
      agents = d$n[[1]],
      transmissions = median(d$transmissions),
      tree = mean(d$tree_matches),
      matrix = mean(d$matrix_matches)
    )))
  rows <- rows[order(rows$agents, factor(rows$engine, levels = engine_levels)), ]
  knitr::kable(rows, digits = c(0, 0, 0, 2, 2),
    col.names = c("Engine", "Agents", "Median transmissions",
                  "Share of runs: tree matches counts",
                  "Share of runs: matrix matches counts"),
    row.names = FALSE)
}

#' Median daily incidence and reproductive number by engine.
plot_daily <- function(bench) {
  daily <- load_daily(bench)
  if (!nrow(daily)) return(NULL)
  long <- rbind(
    data.frame(daily[c("engine", "n", "day")], measure = "Daily incidence",
               value = daily$median_incidence),
    data.frame(daily[c("engine", "n", "day")],
               measure = "Reproductive number, by day the case was infected",
               value = daily$median_reproductive_number)
  )
  long <- long[!is.na(long$value), ]
  long$engine <- factor(long$engine, levels = engine_levels)
  long$agents <- agents_factor(long$n)
  ggplot2::ggplot(long, ggplot2::aes(day, value, colour = engine)) +
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::facet_grid(measure ~ agents, scales = "free_y", switch = "y",
                        labeller = ggplot2::label_wrap_gen(28)) +
    ggplot2::labs(x = "Day", y = NULL, colour = NULL,
                  subtitle = "Medians across replicates") +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(legend.position = "bottom", strip.placement = "outside")
}
