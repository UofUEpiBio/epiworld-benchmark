library(epiworldR)
library(data.table)

prepare_data <- function(m) {

    err <- tryCatch({
        ans <- list(
            repnum    = epiworldR::plot_reproductive_number(m, plot = FALSE),
            incidence = epiworldR::plot_incidence(m, plot = FALSE),
            gentime   = epiworldR::plot_generation_time(m, plot = FALSE)
        )

        # Filling
        ans <- lapply(ans, data.table::as.data.table)

        # Replacing NaN and NAs with the previous value
        # in each element in the list
        ans$repnum[,
            "avg" := data.table::nafill(.SD[[1L]], "locf"), by = "variant",
            .SDcols = "avg"
            ]
        ans$gentime[,
            "avg" := data.table::nafill(.SD[[1L]], "locf"), by = "variant",
            .SDcols = "avg"
            ]

        # Reference table for merging
        ndays <- epiworldR::get_ndays(m)

        ref_table <- data.table::data.table(
            date = 0:ndays
        )

        # Replace the $ with the [[ ]] to avoid the warning in the next
        # two lines
        ans[["repnum"]] <- data.table::merge.data.table(
            ref_table, ans[["repnum"]], by = "date", all.x = TRUE
            )
        ans[["gentime"]] <- data.table::merge.data.table(
            ref_table, ans[["gentime"]], by = "date", all.x = TRUE
            )

        # Generating the arrays
        ans <- data.table::data.table(
            infected =  ans[["incidence"]][["Infected"]],
            recovered = ans[["incidence"]][["Recovered"]],
            repnum    = ans[["repnum"]][["avg"]],
            gentime   = ans[["gentime"]][["avg"]],
            repnum_sd = ans[["repnum"]][["sd"]],
            gentime_sd = ans[["gentime"]][["sd"]]
        )

        # Filling NAs with last obs
        ans[, "infected" := data.table::nafill(.SD[[1]], "locf"),
            .SDcols = "infected"]
        ans[, "recovered" := data.table::nafill(.SD[[1]], "locf"),
            .SDcols = "recovered"]
        ans[, "repnum" := data.table::nafill(.SD[[1]], "locf"),
            .SDcols = "repnum"]
        ans[, "gentime" := data.table::nafill(.SD[[1]], "locf"),
            .SDcols = "gentime"]
        ans[, "repnum_sd" := data.table::nafill(.SD[[1]], "locf"),
            .SDcols = "repnum_sd"]
        ans[, "gentime_sd" := data.table::nafill(.SD[[1]], "locf"),
            .SDcols = "gentime_sd"]

    }, error = function(e) e)

    # If there is an error, return NULL
    if (inherits(err, "error")) {
        return(err)
    }

    # Returning without the first observation (which is mostly zero)
    t(diff(as.matrix(ans[-1,])))

}