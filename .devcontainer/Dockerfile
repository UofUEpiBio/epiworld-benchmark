# Reproducible environment for developing and running the benchmark.
#
# Build from the repository root:
#   podman build -t epiworld-benchmark -f .devcontainer/Dockerfile .
FROM docker.io/rocker/r-ver:4.5.1

ARG EPIWORLDR_SHA=74266774a12475003ba8b1676d87123c30e74fcf
ARG QUARTO_VERSION=1.10.18
ARG RUST_VERSION=1.98.0
ARG PYTHON_VERSION=3.12

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git make pkg-config \
    && rm -rf /var/lib/apt/lists/*

# R: report dependencies plus epiworldR pinned to the commit that reports
# version 0.15.1.0 (CRAN only carries 0.14.0.0).
RUN Rscript -e 'install.packages(c("cpp11", "jsonlite", "ggplot2", "knitr", "rmarkdown"), Ncpus = parallel::detectCores())' \
    && curl -fsSL "https://github.com/UofUEpiBio/epiworldR/archive/${EPIWORLDR_SHA}.tar.gz" -o /tmp/epiworldR.tar.gz \
    && R CMD INSTALL /tmp/epiworldR.tar.gz \
    && rm /tmp/epiworldR.tar.gz \
    && Rscript -e 'stopifnot(packageVersion("epiworldR") == "0.15.1.0")'

# Quarto, for rendering README.qmd.
RUN arch="$(dpkg --print-architecture)" \
    && curl -fsSL "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${arch}.tar.gz" \
        | tar -xz -C /opt \
    && ln -s "/opt/quarto-${QUARTO_VERSION}/bin/quarto" /usr/local/bin/quarto

# Rust, for the ixa runner.
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

WORKDIR /workspace
CMD ["bash"]
