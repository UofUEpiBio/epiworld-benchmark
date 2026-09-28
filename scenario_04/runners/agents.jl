#!/usr/bin/env julia
# Single-replicate Agents.jl runner for the collapsed GeoPops scenario.

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

@agent struct Person(NoSpaceAgent) <: AbstractAgent
    status::UInt8
end

mutable struct Parameters
    adjacency::Vector{Vector{Int}}
    beta::Float64
    incubation::Float64
    hospitalization::Float64
    recovery::Float64
    hospital_recovery::Float64
    peak_hospitalized::Int
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
    transitions = Tuple{Int,UInt8}[]
    for agent in allagents(model)
        if agent.status == I
            for neighbor in p.adjacency[agent.id]
                if model[neighbor].status == S && rand(rng) < p.beta
                    push!(transitions, (neighbor, E))
                end
            end
            draw = rand(rng)
            none = (1 - p.hospitalization) * (1 - p.recovery)
            only_h = p.hospitalization * (1 - p.recovery)
            only_r = p.recovery * (1 - p.hospitalization)
            total = none + only_h + only_r
            if draw < only_h / total
                push!(transitions, (agent.id, H))
            elseif draw < (only_h + only_r) / total
                push!(transitions, (agent.id, R))
            end
        elseif agent.status == E && rand(rng) < p.incubation
            push!(transitions, (agent.id, I))
        elseif agent.status == H && rand(rng) < p.hospital_recovery
            push!(transitions, (agent.id, R))
        end
    end
    for (id, status) in transitions
        if status != E || model[id].status == S
            model[id].status = status
        end
    end
    hospitalized = count(agent -> agent.status == H, allagents(model))
    p.peak_hospitalized = max(p.peak_hospitalized, hospitalized)
end

# Compile the model's methods on a throwaway two-agent model, so that JIT
# compilation, like the Python runners' imports, stays outside every timer.
let warmup = StandardABM(
        Person; model_step!, rng = Xoshiro(0), container = Vector,
        properties = Parameters([[2], [1]], 0.5, 0.5, 0.5, 0.5, 0.5, 0),
    )
    add_agent!(warmup, I)
    add_agent!(warmup, S)
    randperm(abmrng(warmup), 2)
    step!(warmup, 2)
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
    1 / number("hospital_days"), 0,
)
model = StandardABM(
    Person; model_step!, properties, rng = Xoshiro(integer("seed")), container = Vector,
)
for _ in 1:n
    add_agent!(model, S)
end

simulate_started = time_ns()
for id in randperm(abmrng(model), n)[1:min(integer("initial_infected"), n)]
    model[id].status = I
end
step!(model, integer("days"))
simulate_seconds = (time_ns() - simulate_started) / 1e9
total_seconds = (time_ns() - total_started) / 1e9

counts = Dict(state => count(a -> a.status == state, allagents(model)) for state in (S, E, I, H, R))
sum(values(counts)) == n || error("final compartment counts do not sum to population size")

function json_string(value)
    value isa AbstractString && return "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
    value isa Bool && return value ? "true" : "false"
    string(value)
end

record = [
    "status" => "ok", "engine" => "Agents.jl", "engine_version" => arg["engine_version"],
    "n" => n, "days" => integer("days"), "replicate" => integer("replicate"),
    "seed" => integer("seed"), "network_sha256" => arg["network_sha256"],
    "network_edges" => edge_count, "mean_degree" => number("mean_degree"),
    "target_r0" => number("target_r0"),
    "transmission_multiplier" => number("transmission_multiplier"),
    "read_seconds" => read_seconds, "setup_seconds" => total_seconds - simulate_seconds,
    "simulate_seconds" => simulate_seconds, "total_seconds" => total_seconds,
    "final_susceptible" => counts[S], "final_exposed" => counts[E],
    "final_infected" => counts[I], "final_hospitalized" => counts[H],
    "final_recovered" => counts[R], "peak_hospitalized" => properties.peak_hospitalized,
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
