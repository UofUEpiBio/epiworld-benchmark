//! Single-replicate ixa runner for the network ABM benchmark.
//!
//! The model is the benchmark's synchronous daily SEIRH: one ixa plan per day
//! evaluates transmission along the shared contact network and the daily
//! progression probabilities, then applies every transition at once. Scenario
//! 01 adds an all-or-nothing vaccine given before the first day.

use std::cell::Cell;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Instant, SystemTime};

use clap::Parser;
use flate2::read::GzDecoder;
use ixa::prelude::*;
use serde_json::json;

define_entity!(Person);

define_property!(
    enum DiseaseStatus {
        S,
        E,
        I,
        H,
        R,
    },
    Person,
    default_const = DiseaseStatus::S
);

define_property!(
    enum VaccineStatus {
        Unvaccinated,
        Protected,
        Unprotected,
    },
    Person,
    default_const = VaccineStatus::Unvaccinated
);

define_edge_type!(struct Contact, Person);

define_rng!(SeedRng);
define_rng!(TransmissionRng);
define_rng!(ProgressionRng);
define_rng!(VaccineRng);

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    network: PathBuf,
    #[arg(long)]
    network_sha256: String,
    #[arg(long)]
    network_edges: usize,
    #[arg(long)]
    n: usize,
    #[arg(long)]
    days: u32,
    #[arg(long)]
    replicate: u32,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    mean_degree: f64,
    #[arg(long)]
    target_r0: f64,
    #[arg(long)]
    initial_infected: usize,
    #[arg(long)]
    latent_days: f64,
    #[arg(long)]
    infectious_days: f64,
    #[arg(long)]
    hospitalization_probability: f64,
    #[arg(long)]
    hospital_days: f64,
    #[arg(long)]
    vaccine_coverage: f64,
    #[arg(long)]
    vaccine_efficacy: f64,
    #[arg(long, default_value_t = 1.0)]
    transmission_multiplier: f64,
    #[arg(long)]
    fingerprint: String,
    #[arg(long)]
    engine_version: String,
    #[arg(long)]
    output: PathBuf,
}

/// Daily transition probabilities shared by every plan.
#[derive(Clone, Copy, Debug)]
struct Rates {
    transmission: f64,
    incubation: f64,
    hospitalization: f64,
    recovery: f64,
    hospital_recovery: f64,
}

impl Rates {
    /// The same analytic mapping used by the epiworldR and Python runners.
    fn from_args(args: &Args) -> Self {
        let recovery = 1.0 / args.infectious_days;
        let transmissibility = (args.target_r0 / (args.mean_degree - 1.0).max(1.0)).min(0.999);
        let transmission = transmissibility * recovery / (1.0 - transmissibility * (1.0 - recovery));
        let p = args.hospitalization_probability;
        Rates {
            transmission: transmission * args.transmission_multiplier,
            incubation: 1.0 / args.latent_days,
            hospitalization: p * recovery / (1.0 - p * (1.0 - recovery)),
            recovery,
            hospital_recovery: 1.0 / args.hospital_days,
        }
    }
}

#[derive(Debug, PartialEq)]
struct Outcome {
    susceptible: usize,
    exposed: usize,
    infected: usize,
    hospitalized: usize,
    recovered: usize,
    peak_hospitalized: usize,
    vaccinated: usize,
    vaccine_protected: usize,
}

fn read_edges(path: &Path) -> Vec<(usize, usize)> {
    let file = File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    let mut lines = BufReader::new(GzDecoder::new(file)).lines();
    lines.next().expect("edge list is empty").expect("cannot read header");
    lines
        .map(|line| {
            let line = line.expect("cannot read edge");
            let (source, target) = line.split_once('\t').expect("malformed edge");
            (source.parse().expect("bad source"), target.parse().expect("bad target"))
        })
        .collect()
}

/// All-or-nothing vaccine: a fixed share of agents is vaccinated, and each
/// vaccinated agent is protected with probability `efficacy`.
#[derive(Clone, Copy, Debug)]
struct Vaccine {
    coverage: f64,
    efficacy: f64,
}

/// Build the population, contact network, vaccination, and initial infections.
fn build_context(
    n: usize,
    edges: &[(usize, usize)],
    initial_infected: usize,
    vaccine: Vaccine,
    seed: u64,
) -> Context {
    let mut context = Context::new();
    context.init_random(seed);
    context.index_property::<Person, DiseaseStatus>();

    let people: Vec<PersonId> = (0..n)
        .map(|_| context.add_entity(Person).expect("cannot add person"))
        .collect();
    for &(source, target) in edges {
        context
            .add_edge_bidi::<Person, Contact>(people[source], people[target], 1.0, Contact)
            .expect("cannot add edge");
    }
    let vaccinated = (vaccine.coverage * n as f64).round() as usize;
    for person in context.sample_entities(VaccineRng, Person, vaccinated) {
        let status = if context.sample_bool(VaccineRng, vaccine.efficacy) {
            VaccineStatus::Protected
        } else {
            VaccineStatus::Unprotected
        };
        context.set_property(person, status);
    }
    for person in context.sample_entities(SeedRng, Person, initial_infected.min(n)) {
        context.set_property(person, DiseaseStatus::I);
    }
    context
}

fn with_status(context: &Context, status: DiseaseStatus) -> Vec<PersonId> {
    context.query_result_iterator(with!(Person, status)).collect()
}

/// epiworld's roulette: conditional on at most one competing event firing,
/// pick none or the index of the single event that fires.
fn roulette(context: &Context, probabilities: &[f64]) -> Option<usize> {
    let none: f64 = probabilities.iter().map(|p| 1.0 - p).product();
    let single: Vec<f64> = probabilities.iter().map(|p| p * none / (1.0 - p)).collect();
    let total = none + single.iter().sum::<f64>();
    let draw = context.sample_range(ProgressionRng, 0.0..1.0);
    let mut cumulative = none / total;
    if draw < cumulative {
        return None;
    }
    for (index, probability) in single.iter().enumerate() {
        cumulative += probability / total;
        if draw < cumulative {
            return Some(index);
        }
    }
    None
}

/// One synchronous day: every transition is sampled against the state at the
/// start of the day and applied afterwards.
fn step(context: &mut Context, rates: Rates, peak: &Cell<usize>) {
    let exposed = with_status(context, DiseaseStatus::E);
    let infected = with_status(context, DiseaseStatus::I);
    let hospitalized = with_status(context, DiseaseStatus::H);
    let mut transitions: Vec<(PersonId, DiseaseStatus)> = Vec::new();

    for &person in &infected {
        let contacts = context.get_matching_edges::<Person, Contact>(person, |context, edge| {
            let status: DiseaseStatus = context.get_property(edge.neighbor);
            let vaccine: VaccineStatus = context.get_property(edge.neighbor);
            status == DiseaseStatus::S && vaccine != VaccineStatus::Protected
        });
        for edge in contacts {
            if context.sample_bool(TransmissionRng, rates.transmission) {
                transitions.push((edge.neighbor, DiseaseStatus::E));
            }
        }
    }
    for &person in &exposed {
        if context.sample_bool(ProgressionRng, rates.incubation) {
            transitions.push((person, DiseaseStatus::I));
        }
    }
    let competing = [rates.hospitalization, rates.recovery];
    for &person in &infected {
        match roulette(context, &competing) {
            Some(0) => transitions.push((person, DiseaseStatus::H)),
            Some(_) => transitions.push((person, DiseaseStatus::R)),
            None => {}
        }
    }
    for &person in &hospitalized {
        if context.sample_bool(ProgressionRng, rates.hospital_recovery) {
            transitions.push((person, DiseaseStatus::R));
        }
    }

    for (person, status) in transitions {
        // A susceptible reached by several infectious contacts is exposed once.
        if status == DiseaseStatus::E {
            let current: DiseaseStatus = context.get_property(person);
            if current != DiseaseStatus::S {
                continue;
            }
        }
        context.set_property(person, status);
    }
    let now_hospitalized = context.query_entity_count(with!(Person, DiseaseStatus::H));
    peak.set(peak.get().max(now_hospitalized));
}

/// Schedule one plan per day and return a handle to the running peak.
fn schedule_days(context: &mut Context, days: u32, rates: Rates) -> Rc<Cell<usize>> {
    let peak = Rc::new(Cell::new(0));
    for day in 1..=days {
        let peak = Rc::clone(&peak);
        context.add_plan(f64::from(day), move |context| step(context, rates, &peak));
    }
    peak
}

fn outcome(context: &Context, peak: usize) -> Outcome {
    let count = |status| context.query_entity_count(with!(Person, status));
    let vaccine_count = |status| context.query_entity_count(with!(Person, status));
    let protected = vaccine_count(VaccineStatus::Protected);
    Outcome {
        susceptible: count(DiseaseStatus::S),
        exposed: count(DiseaseStatus::E),
        infected: count(DiseaseStatus::I),
        hospitalized: count(DiseaseStatus::H),
        recovered: count(DiseaseStatus::R),
        peak_hospitalized: peak,
        vaccinated: protected + vaccine_count(VaccineStatus::Unprotected),
        vaccine_protected: protected,
    }
}

fn atomic_write(path: &Path, contents: &str) {
    let directory = path.parent().expect("output has no parent directory");
    std::fs::create_dir_all(directory).expect("cannot create output directory");
    let name = path.file_name().unwrap().to_string_lossy();
    let temporary = directory.join(format!(".{name}.{}", std::process::id()));
    let mut stream = File::create(&temporary).expect("cannot create temporary output");
    stream.write_all(contents.as_bytes()).expect("cannot write output");
    stream.sync_all().expect("cannot flush output");
    std::fs::rename(&temporary, path).expect("cannot install output");
}

fn main() {
    let args = Args::parse();
    let total_started = Instant::now();

    let edges = read_edges(&args.network);
    assert_eq!(
        edges.len(),
        args.network_edges,
        "edge count mismatch: expected {}, read {}",
        args.network_edges,
        edges.len()
    );
    let rates = Rates::from_args(&args);
    let vaccine = Vaccine { coverage: args.vaccine_coverage, efficacy: args.vaccine_efficacy };
    let mut context = build_context(args.n, &edges, args.initial_infected, vaccine, args.seed);
    drop(edges);
    let peak = schedule_days(&mut context, args.days, rates);

    let simulate_started = Instant::now();
    context.execute();
    let simulate_seconds = simulate_started.elapsed().as_secs_f64();
    let total_seconds = total_started.elapsed().as_secs_f64();

    let result = outcome(&context, peak.get());
    let final_total =
        result.susceptible + result.exposed + result.infected + result.hospitalized + result.recovered;
    assert_eq!(final_total, args.n, "final compartment counts do not sum to population size");

    let record = json!({
        "status": "ok",
        "engine": "ixa",
        "engine_version": args.engine_version,
        "n": args.n,
        "days": args.days,
        "replicate": args.replicate,
        "seed": args.seed,
        "network_sha256": args.network_sha256,
        "network_edges": args.network_edges,
        "mean_degree": args.mean_degree,
        "target_r0": args.target_r0,
        "transmission_multiplier": args.transmission_multiplier,
        "setup_seconds": total_seconds - simulate_seconds,
        "simulate_seconds": simulate_seconds,
        "total_seconds": total_seconds,
        "final_susceptible": result.susceptible,
        "final_exposed": result.exposed,
        "final_infected": result.infected,
        "final_hospitalized": result.hospitalized,
        "final_recovered": result.recovered,
        "peak_hospitalized": result.peak_hospitalized,
        "vaccinated": result.vaccinated,
        "vaccine_protected": result.vaccine_protected,
        "fingerprint": args.fingerprint,
        "timestamp_utc": humantime::format_rfc3339_micros(SystemTime::now()).to_string(),
    });
    let mut contents = serde_json::to_string_pretty(&record).expect("cannot encode record");
    contents.push('\n');
    atomic_write(&args.output, &contents);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> Vec<(usize, usize)> {
        (0..n).flat_map(|i| [(i, (i + 1) % n), (i, (i + 2) % n)]).collect()
    }

    fn simulate_with(seed: u64, vaccine: Vaccine) -> Outcome {
        let rates = Rates {
            transmission: 0.2,
            incubation: 0.25,
            hospitalization: 0.05,
            recovery: 1.0 / 7.0,
            hospital_recovery: 1.0 / 7.0,
        };
        let mut context = build_context(200, &ring(200), 5, vaccine, seed);
        let peak = schedule_days(&mut context, 60, rates);
        context.execute();
        outcome(&context, peak.get())
    }

    fn simulate(seed: u64) -> Outcome {
        simulate_with(seed, Vaccine { coverage: 0.3, efficacy: 0.8 })
    }

    #[test]
    fn compartments_sum_to_population() {
        let result = simulate(1);
        let total = result.susceptible
            + result.exposed
            + result.infected
            + result.hospitalized
            + result.recovered;
        assert_eq!(total, 200);
        assert!(result.susceptible < 195, "the outbreak should spread: {result:?}");
    }

    #[test]
    fn simulation_is_deterministic_for_a_seed() {
        assert_eq!(simulate(7), simulate(7));
    }

    #[test]
    fn vaccine_protects_the_expected_share() {
        let result = simulate(3);
        assert_eq!(result.vaccinated, 60);
        assert!((36..=60).contains(&result.vaccine_protected), "{result:?}");
    }

    #[test]
    fn protected_agents_are_never_infected() {
        // Everyone vaccinated and protected: only the seeded cases progress.
        let result = simulate_with(5, Vaccine { coverage: 1.0, efficacy: 1.0 });
        assert_eq!(result.susceptible, 195, "{result:?}");
        assert_eq!(result.exposed, 0);
    }

    #[test]
    fn roulette_never_fires_with_zero_probabilities() {
        let mut context = Context::new();
        context.init_random(3);
        assert!((0..100).all(|_| roulette(&context, &[0.0, 0.0]).is_none()));
    }
}
