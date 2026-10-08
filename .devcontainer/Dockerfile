# Reproducible environment for developing and running the benchmark.
#
# Build from the repository root:
#   podman build -t epiworld-benchmark -f .devcontainer/Dockerfile .
FROM docker.io/rocker/r-ver:4.5.1

# Each ARG is declared just before the step that uses it: a changed ARG
# invalidates the build cache from its declaration on, so bumping one package
# rebuilds only the layers after it. The pinned commits are also ENV, so
# run.py can record them (scripts/environment.py).

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates cmake curl git make pkg-config zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# R packages for the reports and the epiworldR runner's dependencies.
RUN Rscript -e 'install.packages(c("cpp11", "jsonlite", "ggplot2", "knitr", "rmarkdown"), Ncpus = parallel::detectCores())'

# Quarto, for rendering README.qmd.
ARG QUARTO_VERSION=1.10.18
RUN arch="$(dpkg --print-architecture)" \
    && curl -fsSL "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${arch}.tar.gz" \
        | tar -xz -C /opt \
    && ln -s "/opt/quarto-${QUARTO_VERSION}/bin/quarto" /usr/local/bin/quarto

# Rust, for the ixa runner.
ARG RUST_VERSION=1.98.0
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rustc --version

# Julia and the Agents.jl environment, for every scenario's Agents.jl runner.
ARG JULIA_VERSION=1.13.1
RUN architecture="$(dpkg --print-architecture)" \
    && case "${architecture}" in \
        amd64) julia_arch=x86_64; julia_path=x64 ;; \
        arm64) julia_arch=aarch64; julia_path=aarch64 ;; \
        *) echo "Unsupported Julia architecture: ${architecture}" >&2; exit 1 ;; \
       esac \
    && curl -fsSL "https://julialang-s3.julialang.org/bin/linux/${julia_path}/1.13/julia-${JULIA_VERSION}-linux-${julia_arch}.tar.gz" \
        | tar -xz -C /opt \
    && ln -s "/opt/julia-${JULIA_VERSION}/bin/julia" /usr/local/bin/julia \
    && julia --version

# Python via uv. The environment and the Cargo target directory live outside
# the bind-mounted checkout so they never collide with a host .venv/target.
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHON=/opt/venv/bin/python \
    CARGO_TARGET_DIR=/opt/cargo-target
ARG PYTHON_VERSION=3.12
RUN uv python install "${PYTHON_VERSION}"

WORKDIR /opt/benchmark-deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen && rm -rf /root/.cache/uv

COPY julia/Project.toml julia/Manifest.toml ./julia/
RUN julia --project=/opt/benchmark-deps/julia --startup-file=no \
        -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
    && rm -rf /root/.julia/logs /root/.julia/registries/.cache

# Pre-fetch and pre-build the ixa runners' dependencies so `make setup` only
# compiles the runners themselves. Every scenario's crate pins the same
# dependency versions, so scenario_00's lock file warms the cache for all.
COPY scenario_00/runners/ixa/Cargo.toml scenario_00/runners/ixa/Cargo.lock scenario_00/runners/ixa/rust-toolchain.toml ./ixa/
RUN mkdir -p ixa/src && echo 'fn main() {}' > ixa/src/main.rs \
    && cargo build --release --locked --manifest-path ixa/Cargo.toml \
    && rm -rf ixa "${CARGO_TARGET_DIR}/release/ixa-scenario-"* \
        "${CARGO_TARGET_DIR}/release/deps/ixa_scenario_"* \
        "${CARGO_TARGET_DIR}/release/.fingerprint/ixa-scenario-"*

# individual (mrc-ide), for its R runner. It is no longer on CRAN, so it is
# installed from its GitHub release, pinned to the v0.1.19 tag's commit. It
# links against testthat's C++ headers, so testthat is needed to build it.
ARG INDIVIDUAL_SHA=50524f239e965d38b40f178290262760f252fc60
ENV INDIVIDUAL_SHA=${INDIVIDUAL_SHA}
RUN Rscript -e 'install.packages(c("R6", "Rcpp", "testthat"), Ncpus = parallel::detectCores())' \
    && curl -fsSL "https://github.com/mrc-ide/individual/archive/${INDIVIDUAL_SHA}.tar.gz" -o /tmp/individual.tar.gz \
    && R CMD INSTALL /tmp/individual.tar.gz \
    && rm /tmp/individual.tar.gz \
    && Rscript -e 'stopifnot(packageVersion("individual") == "0.1.19")'

# ABM (Junling Ma), for its R runner: the CRAN release, pinned to 0.4.3. The
# runner uses its 0.4 API (transitions without a contact rate), so a newer
# release needs the runner revisited. It depends on R6 and Rcpp, installed above.
RUN (curl -fsSL https://cran.r-project.org/src/contrib/ABM_0.4.3.tar.gz -o /tmp/ABM.tar.gz \
        || curl -fsSL https://cran.r-project.org/src/contrib/Archive/ABM/ABM_0.4.3.tar.gz -o /tmp/ABM.tar.gz) \
    && R CMD INSTALL /tmp/ABM.tar.gz \
    && rm /tmp/ABM.tar.gz \
    && Rscript -e 'stopifnot(packageVersion("ABM") == "0.4.3")'

# The epiworld family last, as it changes most often. All three are pinned to
# commits on their default branches at epiworld 0.17.1: epiworld's master and
# epiworldR's main here, epiworldpy in pyproject.toml above.
ARG EPIWORLDR_SHA=fca60f3010d5a018a11059bdf61a7ce21f4c42ca
ENV EPIWORLDR_SHA=${EPIWORLDR_SHA}
RUN curl -fsSL "https://github.com/UofUEpiBio/epiworldR/archive/${EPIWORLDR_SHA}.tar.gz" -o /tmp/epiworldR.tar.gz \
    && R CMD INSTALL /tmp/epiworldR.tar.gz \
    && rm /tmp/epiworldR.tar.gz \
    && Rscript -e 'stopifnot(packageVersion("epiworldR") == "0.17.1.0")'

# epiworld's C++ headers, for the epiworld (C++) runner. The library is
# header-only; each scenario's runner is compiled by `make setup`. Keep the
# commit in step with EPIWORLD_REF in the Makefile.
ARG EPIWORLD_SHA=04c4ad866b1866ca88332a888d941b395c96b8a0
ENV EPIWORLD_SHA=${EPIWORLD_SHA} \
    EPIWORLD_INCLUDE=/opt/epiworld/include \
    EPIWORLD_BUILD_DIR=/opt/epiworld-build
RUN mkdir -p /opt/epiworld \
    && curl -fsSL "https://github.com/UofUEpiBio/epiworld/archive/${EPIWORLD_SHA}.tar.gz" \
        | tar -xz -C /opt/epiworld --strip-components=1 \
    && grep -q 'EPIWORLD_VERSION_MINOR 17' /opt/epiworld/include/epiworld/epiworld.hpp \
    && grep -q 'EPIWORLD_VERSION_PATCH 1' /opt/epiworld/include/epiworld/epiworld.hpp

# FRED, for every scenario's FRED runner. Unlike epiworld, it
# takes no bind-mounted scenario inputs (the runner drives it entirely via a
# generated model file), so it is fully built into the image rather than by
# `make setup`. Keep the commit in step with FRED_REF in the Makefile. FRED's
# Makefile hardcodes -m64 for every Linux target, which gcc rejects on arm64;
# M64= clears it (redundant, not wrong, on x86_64). FRED always reads
# data/config.fred at startup for its place-type defaults, even though this
# runner supplies its own synthetic population and never touches FRED's
# bundled census data (data/country, the rest of data/'s size); that alone is
# dropped. Date::setup_dates() writes one entry past its date array on every
# run; glibc aborts on the corrupted heap at 200,000 agents, so the array gets
# one more entry. Keep this patch in step with the Makefile's.
ARG FRED_SHA=bd25f048f8e390ff697bdfa83a0bd3c2e19d6046
ENV FRED_SHA=${FRED_SHA} \
    FRED_HOME=/opt/fred
RUN mkdir -p /opt/fred \
    && curl -fsSL "https://github.com/PublicHealthDynamicsLab/FRED/archive/${FRED_SHA}.tar.gz" \
        | tar -xz -C /opt/fred --strip-components=1 \
    && sed -i 's/new date_t \[ Date::max_days \];/new date_t [ Date::max_days + 1 ];/' /opt/fred/src/Date.cc \
    && grep -q 'new date_t \[ Date::max_days + 1 \];' /opt/fred/src/Date.cc \
    && make -C /opt/fred/src FRED M64= \
    && echo "${FRED_SHA}" > /opt/fred/COMMIT \
    && rm -rf /opt/fred/tests /opt/fred/docs /opt/fred/pybin /opt/fred/data/country

# ggridges, for the scenario reports' ridgeline simulation-time plots. Last,
# alongside the other report-only packages, so bumping it never invalidates
# the epiworld family or FRED layers above.
RUN Rscript -e 'install.packages("ggridges", Ncpus = parallel::detectCores())'

# GNU time, which run.py and the FRED runners start each measured process
# through: it reports the process's peak memory without counting its parent's
# (see run_measured() in run.py). Late in the file, so adding it rebuilt
# nothing above.
RUN apt-get update \
    && apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/*

# Julia installed and precompiled the Agents.jl environment under root's home;
# pin the depot so it is found whatever HOME a runner sets (GitHub Actions job
# containers use /github/home).
ENV JULIA_DEPOT_PATH=/root/.julia

WORKDIR /workspace
CMD ["bash"]
