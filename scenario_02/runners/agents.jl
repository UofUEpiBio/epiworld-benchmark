#!/usr/bin/env julia
# Single-replicate Agents.jl runner for benchmark scenario 02.
#
# Scenario 02 extracts four outputs from every run: the transmission tree, daily
# incidence, the reproductive number, and the daily transition matrix.

using Agents
using CodecZlib
using Random
using Dates

if length(ARGS) == 1 && ARGS[1] == "--version"
    print(Base.pkgversion(Agents))
    exit()
end

function arguments(values)
    iseven(length(values)) || error("arguments must be --key value pairs")
    Dict(replace(values[i][3:end], "-" => "_") => values[i + 1] for i in 1:2:length(values))
end

arg = arguments(ARGS)
number(key) = parse(Float64, arg[key])
integer(key) = parse(Int, arg[key])

const S, E, I, H, R = UInt8(0), UInt8(1), UInt8(2), UInt8(3), UInt8(4)

# `protected` marks all-or-nothing vaccinees the vaccine protects; transmission
# skips them. Unprotected vaccinees need no marker.
@agent struct Person(NoSpaceAgent) <: AbstractAgent
    status::UInt8
    protected::Bool
end

mutable struct Parameters
    adjacency::Vector{Vector{Int}}
    beta::Float64
    incubation::Float64
    hospitalization::Float64
    recovery::Float64
    hospital_recovery::Float64
    peak_hospitalized::Int
    # The outputs: (day, source, target) for every transmission, and
    # (day, from, to) for every transition.
    day::Int
    tree::Vector{NTuple{3,Int}}
    transitions::Vector{NTuple{3,Int}}
end

function read_edges(path, n)
    adjacency = [Int[] for _ in 1:n]
    stream = GzipDecompressorStream(open(path))
    readline(stream)
    edges = 0
    for line in eachline(stream)
        source_text, target_text = split(line, '\t')
        source, target = parse(Int, source_text) + 1, parse(Int, target_text) + 1
        push!(adjacency[source], target)
        push!(adjacency[target], source)
        edges += 1
    end
    close(stream)
    adjacency, edges
end

function model_step!(model)
    p = abmproperties(model)
    rng = abmrng(model)
    p.day += 1
    # (agent, new status, source of an infection or 0)
    transitions = Tuple{Int,UInt8,Int}[]
    for agent in allagents(model)
        if agent.status == I
            for neighbor in p.adjacency[agent.id]
                contact = model[neighbor]
                if contact.status == S && !contact.protected && rand(rng) < p.beta
                    push!(transitions, (neighbor, E, agent.id))
                end
            end
            draw = rand(rng)
            none = (1 - p.hospitalization) * (1 - p.recovery)
            only_h = p.hospitalization * (1 - p.recovery)
            only_r = p.recovery * (1 - p.hospitalization)
            total = none + only_h + only_r
            if draw < only_h / total
                push!(transitions, (agent.id, H, 0))
            elseif draw < (only_h + only_r) / total
                push!(transitions, (agent.id, R, 0))
            end
        elseif agent.status == E && rand(rng) < p.incubation
            push!(transitions, (agent.id, I, 0))
        elseif agent.status == H && rand(rng) < p.hospital_recovery
            push!(transitions, (agent.id, R, 0))
        end
    end
    for (id, status, source) in transitions
        if status != E || model[id].status == S
            push!(p.transitions, (p.day, Int(model[id].status), Int(status)))
            status == E && push!(p.tree, (p.day, source, id))
            model[id].status = status
        end
    end
    hospitalized = count(agent -> agent.status == H, allagents(model))
    p.peak_hospitalized = max(p.peak_hospitalized, hospitalized)
end

"""
The daily transition matrix, indexed [day + 1, from + 1, to + 1] over S, E, I,
H, R; daily incidence for days 1 to `days`; and epiworld's reproductive number:
the mean number of secondary infections caused by the cases infected on each
day, with the seed cases on day 0.
"""
function outputs(p, seeds, n, days)
    matrix = zeros(Int, days + 1, 5, 5)
    for (day, from, to) in p.transitions
        matrix[day + 1, from + 1, to + 1] += 1
    end
    incidence = matrix[2:end, S + 1, E + 1]
    infection_day = fill(-1, n)
    infection_day[seeds] .= 0
    secondary = zeros(Int, n)
    for (day, source, target) in p.tree
        infection_day[target] = day
        secondary[source] += 1
    end
    cases, caused = zeros(Int, days + 1), zeros(Int, days + 1)
    for id in 1:n
        if infection_day[id] >= 0
            cases[infection_day[id] + 1] += 1
            caused[infection_day[id] + 1] += secondary[id]
        end
    end
    reproductive_number = [count == 0 ? nothing : total / count for (total, count) in zip(caused, cases)]
    matrix, incidence, reproductive_number
end

"""
All-or-nothing vaccine, drawn before the seed cases and independently of them:
the number of agents vaccinated and the number protected.
"""
function vaccinate!(model, coverage, efficacy)
    rng = abmrng(model)
    vaccinated = randperm(rng, nagents(model))[1:round(Int, coverage * nagents(model))]
    protected = 0
    for id in vaccinated
        if rand(rng) < efficacy
            model[id].protected = true
            protected += 1
        end
    end
    length(vaccinated), protected
end

# Compile the model's methods and the output extraction on a throwaway
# two-agent model, so that JIT compilation, like the Python runners' imports,
# stays outside every timer.
let warmup = StandardABM(
        Person; model_step!, rng = Xoshiro(0), container = Vector,
        properties = Parameters([[2], [1]], 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, [], []),
    )
    add_agent!(warmup, I, false)
    add_agent!(warmup, S, false)
    vaccinate!(warmup, 0.5, 0.5)
    randperm(abmrng(warmup), 2)
    step!(warmup, 2)
    outputs(abmproperties(warmup), [1], 2, 2)
end

n = integer("n")
total_started = time_ns()
adjacency, edge_count = read_edges(arg["network"], n)
edge_count == integer("network_edges") || error("edge count mismatch")
read_seconds = (time_ns() - total_started) / 1e9

recovery = 1 / number("infectious_days")
transmissibility = min(0.999, number("target_r0") / max(1.0, number("mean_degree") - 1))
beta = transmissibility * recovery / (1 - transmissibility * (1 - recovery))
beta *= number("transmission_multiplier")
hprob = number("hospitalization_probability")
hospitalization = hprob * recovery / (1 - hprob * (1 - recovery))
properties = Parameters(
    adjacency, beta, 1 / number("latent_days"), hospitalization, recovery,
    1 / number("hospital_days"), 0, 0, NTuple{3,Int}[], NTuple{3,Int}[],
)
model = StandardABM(
    Person; model_step!, properties, rng = Xoshiro(integer("seed")), container = Vector,
)
for _ in 1:n
    add_agent!(model, S, false)
end

simulate_started = time_ns()
vaccinated, protected = vaccinate!(model, number("vaccine_coverage"), number("vaccine_efficacy"))
seeds = randperm(abmrng(model), n)[1:min(integer("initial_infected"), n)]
for id in seeds
    model[id].status = I
end
days = integer("days")
step!(model, days)
simulate_seconds = (time_ns() - simulate_started) / 1e9

# Extracting the outputs comes after the simulation and is timed apart.
extract_started = time_ns()
matrix, incidence, reproductive_number = outputs(properties, seeds, n, days)
extract_seconds = (time_ns() - extract_started) / 1e9
total_seconds = (time_ns() - total_started) / 1e9
pair(from, to) = sum(matrix[2:end, from + 1, to + 1])

counts = Dict(state => count(a -> a.status == state, allagents(model)) for state in (S, E, I, H, R))
sum(values(counts)) == n || error("final compartment counts do not sum to population size")

function json_string(value)
    value isa AbstractString && return "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
    value isa Bool && return value ? "true" : "false"
    value === nothing && return "null"
    value isa AbstractVector && return "[" * join(json_string.(value), ", ") * "]"
    string(value)
end

record = [
    "status" => "ok", "engine" => "Agents.jl", "engine_version" => arg["engine_version"],
    "n" => n, "days" => integer("days"), "replicate" => integer("replicate"),
    "seed" => integer("seed"), "network_sha256" => arg["network_sha256"],
    "network_edges" => edge_count, "mean_degree" => number("mean_degree"),
    "target_r0" => number("target_r0"),
    "transmission_multiplier" => number("transmission_multiplier"),
    "read_seconds" => read_seconds, "setup_seconds" => total_seconds - simulate_seconds - extract_seconds,
    "simulate_seconds" => simulate_seconds, "total_seconds" => total_seconds,
    "final_susceptible" => counts[S], "final_exposed" => counts[E],
    "final_infected" => counts[I], "final_hospitalized" => counts[H],
    "final_recovered" => counts[R], "peak_hospitalized" => properties.peak_hospitalized,
    "vaccinated" => vaccinated, "vaccine_protected" => protected,
    "extract_seconds" => extract_seconds, "transmissions" => length(properties.tree),
    "transitions_se" => pair(S, E), "transitions_ei" => pair(E, I),
    "transitions_ih" => pair(I, H), "transitions_ir" => pair(I, R),
    "transitions_hr" => pair(H, R),
    "daily_incidence" => incidence, "reproductive_number" => reproductive_number,
    "fingerprint" => arg["fingerprint"],
    "timestamp_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
]
output = arg["output"]
mkpath(dirname(output))
temporary = output * "." * string(getpid()) * ".tmp"
open(temporary, "w") do stream
    println(stream, "{")
    for (index, (key, value)) in enumerate(record)
        suffix = index == length(record) ? "" : ","
        println(stream, "  ", json_string(key), ": ", json_string(value), suffix)
    end
    println(stream, "}")
end
mv(temporary, output; force = true)
