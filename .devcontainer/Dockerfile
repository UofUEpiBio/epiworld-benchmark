# Reproducible environment for developing and running the benchmark.
#
# Build from the repository root:
#   podman build -t epiworld-benchmark -f .devcontainer/Dockerfile .
FROM docker.io/rocker/r-ver:4.5.1

# Each ARG is declared just before the step that uses it: a changed ARG
# invalidates the build cache from its declaration on, so bumping one package
# rebuilds only the layers after it.

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

# Pre-fetch and pre-build the ixa runners' dependencies so `make setup` only
# compiles the runners themselves. Every scenario's crate pins the same
# dependency versions, so scenario_00's lock file warms the cache for all.
COPY scenario_00/runners/ixa/Cargo.toml scenario_00/runners/ixa/Cargo.lock scenario_00/runners/ixa/rust-toolchain.toml ./ixa/
RUN mkdir -p ixa/src && echo 'fn main() {}' > ixa/src/main.rs \
    && cargo build --release --locked --manifest-path ixa/Cargo.toml \
    && rm -rf ixa "${CARGO_TARGET_DIR}/release/ixa-scenario-"* \
        "${CARGO_TARGET_DIR}/release/deps/ixa_scenario_"* \
        "${CARGO_TARGET_DIR}/release/.fingerprint/ixa-scenario-"*

# The epiworld family last, as it changes most often. All three are pinned to
# commits: epiworld's master and epiworldR's main after the counting-sort
# network build (UofUEpiBio/epiworld#274), which still report version 0.17.0.
# epiworldpy is pinned in pyproject.toml above.
ARG EPIWORLDR_SHA=1f5e63b4b40297e969f3b0ee8ccea652f5c7d277
RUN curl -fsSL "https://github.com/UofUEpiBio/epiworldR/archive/${EPIWORLDR_SHA}.tar.gz" -o /tmp/epiworldR.tar.gz \
    && R CMD INSTALL /tmp/epiworldR.tar.gz \
    && rm /tmp/epiworldR.tar.gz \
    && Rscript -e 'stopifnot(packageVersion("epiworldR") == "0.17.0.0")'

# epiworld's C++ headers, for the epiworld (C++) runner. The library is
# header-only; each scenario's runner is compiled by `make setup`. Keep the
# commit in step with EPIWORLD_REF in the Makefile.
ARG EPIWORLD_SHA=092bad189b137e76f32ea19f1d6ed61f834389d0
ENV EPIWORLD_INCLUDE=/opt/epiworld/include \
    EPIWORLD_BUILD_DIR=/opt/epiworld-build
RUN mkdir -p /opt/epiworld \
    && curl -fsSL "https://github.com/UofUEpiBio/epiworld/archive/${EPIWORLD_SHA}.tar.gz" \
        | tar -xz -C /opt/epiworld --strip-components=1 \
    && grep -q 'EPIWORLD_VERSION_MINOR 17' /opt/epiworld/include/epiworld/epiworld.hpp

WORKDIR /workspace
CMD ["bash"]
