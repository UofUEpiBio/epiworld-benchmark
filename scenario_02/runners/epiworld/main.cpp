// Single-replicate runner for the epiworld C++ library.
//
// The model is the one epiworldR's runner builds, written against the C++ API
// that epiworldR wraps. Scenario 02 extracts the transmission tree, daily
// incidence, reproductive number, and transition matrix after the run. `make
// setup` compiles it the way epiworldR is compiled (-O2 -DNDEBUG
// -Depiworld_double=double), so the two differ only in the R layer.

#include <epiworld/epiworld.hpp>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace epiworld;
using Clock = std::chrono::steady_clock;

// --key value pairs, with dashes in keys turned into underscores.
std::map<std::string, std::string> parse_args(int argc, char ** argv) {
    if ((argc - 1) % 2 != 0)
        throw std::invalid_argument("Arguments must be --key value pairs");
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        std::string key(argv[i]);
        if (key.rfind("--", 0) != 0)
            throw std::invalid_argument("Unexpected argument: " + key);
        key = key.substr(2);
        std::replace(key.begin(), key.end(), '-', '_');
        args[key] = argv[i + 1];
    }
    return args;
}

void read_edges(const std::string & path, std::vector<int> & source, std::vector<int> & target) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (file == nullptr)
        throw std::runtime_error("Cannot open " + path);
    char line[256];
    gzgets(file, line, sizeof(line)); // header
    int from, to;
    while (gzgets(file, line, sizeof(line)) != nullptr) {
        if (std::sscanf(line, "%d\t%d", &from, &to) != 2)
            throw std::runtime_error("Malformed edge line in " + path);
        source.push_back(from);
        target.push_back(to);
    }
    gzclose(file);
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Resident memory in bytes from /proc/self/status: VmRSS now and VmHWM, its
// high-water mark. -1 (written as null) off Linux.
struct Memory {
    long long rss = -1;
    long long peak = -1;
};

Memory memory_status() {
    Memory memory;
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0)
            memory.rss = std::stoll(line.substr(6)) * 1024;
        else if (line.rfind("VmHWM:", 0) == 0)
            memory.peak = std::stoll(line.substr(6)) * 1024;
    }
    return memory;
}

// Writing 5 to clear_refs resets VmHWM to the current RSS (Linux 4.0 and later).
bool reset_peak() {
    std::ofstream clear("/proc/self/clear_refs");
    clear << "5" << std::flush;
    return static_cast<bool>(clear);
}

std::string json_bytes(long long bytes) {
    return bytes < 0 ? "null" : std::to_string(bytes);
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << epiworld_version() << std::endl;
        return 0;
    }

    auto args = parse_args(argc, argv);
    auto number = [&](const std::string & name) { return std::stod(args.at(name)); };
    auto integer = [&](const std::string & name) { return std::stoi(args.at(name)); };

    int n = integer("n");
    int days = integer("days");
    double mean_degree = number("mean_degree");
    double target_r0 = number("target_r0");
    double latent_days = number("latent_days");
    double infectious_days = number("infectious_days");
    double hospital_probability = number("hospitalization_probability");
    double hospital_days = number("hospital_days");
    double vaccine_coverage = number("vaccine_coverage");
    double vaccine_efficacy = number("vaccine_efficacy");
    double recovery = 1.0 / infectious_days;
    double transmissibility = std::min(0.999, target_r0 / std::max(1.0, mean_degree - 1.0));
    double beta = transmissibility * recovery / (1.0 - transmissibility * (1.0 - recovery));
    beta *= number("transmission_multiplier");
    double hospital_rate = hospital_probability * recovery /
        (1.0 - hospital_probability * (1.0 - recovery));

    Memory memory_baseline = memory_status();
    auto total_started = Clock::now();
    std::vector<int> source, target;
    read_edges(args.at("network"), source, target);
    if (static_cast<int>(source.size()) != integer("network_edges"))
        throw std::runtime_error("Network edge count mismatch");
    double read_seconds = seconds_since(total_started);
    Memory memory_after_read = memory_status();

    Model<> model;
    model.add_param(beta, "Transmission rate");
    model.add_param(1.0 / latent_days, "Incubation rate");
    model.add_param(hospital_rate, "Hospitalization rate");
    model.add_param(recovery, "Recovery rate");
    model.add_param(1.0 / hospital_days, "Hospital recovery rate");

    // Only infected agents transmit: exposed, hospitalized, and recovered
    // neighbours (states 1, 3, 4) are excluded.
    model.add_state("Susceptible", sampler::make_update_susceptible<>({1, 3, 4}));
    model.add_state("Exposed", new_state_update_transition<>({"Incubation rate"}, {2}));
    model.add_state("Infected", new_state_update_transition<>(
        {"Hospitalization rate", "Recovery rate"}, {3, 4}
    ));
    model.add_state("Hospitalized", new_state_update_transition<>({"Hospital recovery rate"}, {4}));
    model.add_state("Recovered");

    // New infections enter Exposed (state 1). The initial cases start there
    // too, as in the epiworldR runner.
    Virus<> pathogen("Benchmark pathogen", std::min(integer("initial_infected"), n), false);
    pathogen.set_state(1, 4, 4);
    pathogen.set_prob_infecting("Transmission rate");
    model.add_virus(pathogen);

    // All-or-nothing vaccine, drawn as in the epiworldR runner: unprotected
    // vaccinees behave exactly like unvaccinated agents, so only the protected
    // ones get a tool, which epiworld places on that many agents at random.
    std::mt19937 rng(integer("seed"));
    int vaccinated = static_cast<int>(std::nearbyint(vaccine_coverage * n));
    int protected_ = std::binomial_distribution<int>(vaccinated, vaccine_efficacy)(rng);
    Tool<> vaccine("Vaccine", protected_, false);
    vaccine.set_susceptibility_reduction(1.0);
    model.add_tool(vaccine);

    model.agents_from_edgelist(source, target, n, false);
    model.verbose_off();
    source = {};
    target = {};

    Memory memory_after_setup = memory_status();
    bool peak_reset = reset_peak();
    auto simulate_started = Clock::now();
    model.run(days, integer("seed"));
    double simulate_seconds = seconds_since(simulate_started);
    Memory memory_simulate = memory_status();
    // Extracting the outputs comes after the simulation and is timed apart.
    auto extract_started = Clock::now();

    // The four outputs, all of which epiworld records during the run. The
    // transmission tree includes the seed cases, whose source is -1.
    const auto & db = model.get_db();
    std::vector<int> tree_date, tree_source, tree_target, tree_virus, tree_source_date;
    db.get_transmissions(tree_date, tree_source, tree_target, tree_virus, tree_source_date);

    // The transition matrix: counts for every pair of states on every day.
    std::vector<std::string> transition_from, transition_to;
    std::vector<int> transition_date, transition_counts;
    db.get_hist_transition_matrix(
        transition_from, transition_to, transition_date, transition_counts, false
    );

    // Daily incidence is the matrix's Susceptible -> Exposed entry.
    std::vector<int> daily_incidence(days + 1, 0);
    for (size_t i = 0; i < transition_date.size(); ++i)
        if (transition_from[i] == "Susceptible" && transition_to[i] == "Exposed")
            daily_incidence[transition_date[i]] += transition_counts[i];

    // Each case's number of secondary infections, keyed by (virus, case, day
    // the case was infected), averaged by that day. Source -1 is the model.
    std::vector<double> rt_sum(days + 1, 0.0), rt_cases(days + 1, 0.0);
    for (const auto & [key, secondary] : db.get_reproductive_number()) {
        if (key[1] == -1)
            continue;
        rt_sum[key[2]] += secondary;
        rt_cases[key[2]] += 1.0;
    }
    std::vector<double> reproductive_number(days + 1, std::nan(""));
    for (int day = 0; day <= days; ++day)
        if (rt_cases[day] > 0)
            reproductive_number[day] = rt_sum[day] / rt_cases[day];

    double extract_seconds = seconds_since(extract_started);
    double total_seconds = seconds_since(total_started);

    std::vector<std::string> states;
    std::vector<int> counts;
    model.get_db().get_today_total(&states, &counts);
    std::map<std::string, int> today;
    for (size_t i = 0; i < states.size(); ++i)
        today[states[i]] = counts[i];

    std::vector<int> hist_dates, hist_counts;
    std::vector<std::string> hist_states;
    model.get_db().get_hist_total(&hist_dates, &hist_states, &hist_counts);
    int peak_hospitalized = 0;
    for (size_t i = 0; i < hist_states.size(); ++i)
        if (hist_states[i] == "Hospitalized")
            peak_hospitalized = std::max(peak_hospitalized, hist_counts[i]);

    // Summaries of the outputs for the result record.
    int transmissions = 0;
    for (int source : tree_source)
        transmissions += source != -1;
    std::map<std::string, int> transitions;
    for (size_t i = 0; i < transition_date.size(); ++i)
        if (transition_date[i] > 0 && transition_from[i] != transition_to[i])
            transitions[transition_from[i].substr(0, 1) + transition_to[i].substr(0, 1)] +=
                transition_counts[i];
    std::string incidence_json, rt_json;
    for (int day = 1; day <= days; ++day)
        incidence_json += (day > 1 ? ", " : "") + std::to_string(daily_incidence[day]);
    for (int day = 0; day <= days; ++day) {
        char value[32];
        if (std::isnan(reproductive_number[day]))
            std::snprintf(value, sizeof(value), "null");
        else
            std::snprintf(value, sizeof(value), "%.17g", reproductive_number[day]);
        rt_json += (day > 0 ? ", " : "") + std::string(value);
    }

    int total = 0;
    for (const auto & state : {"Susceptible", "Exposed", "Infected", "Hospitalized", "Recovered"})
        total += today.at(state);
    if (total != n)
        throw std::runtime_error("final compartment counts do not sum to population size");

    char timestamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));

    std::filesystem::path output(args.at("output"));
    std::filesystem::create_directories(output.parent_path());
    auto temporary = output.parent_path() / ("." + output.filename().string() + ".tmp");
    FILE * json = std::fopen(temporary.c_str(), "w");
    if (json == nullptr)
        throw std::runtime_error("Cannot write " + temporary.string());
    std::fprintf(json,
        "{\n"
        "  \"rss_baseline_bytes\": %s,\n"
        "  \"rss_after_read_bytes\": %s,\n"
        "  \"rss_after_setup_bytes\": %s,\n"
        "  \"peak_rss_setup_bytes\": %s,\n"
        "  \"peak_rss_simulate_bytes\": %s,\n"
        "  \"status\": \"ok\",\n"
        "  \"engine\": \"epiworld\",\n"
        "  \"engine_version\": \"%s\",\n"
        "  \"n\": %d,\n"
        "  \"days\": %d,\n"
        "  \"replicate\": %d,\n"
        "  \"seed\": %d,\n"
        "  \"network_sha256\": \"%s\",\n"
        "  \"network_edges\": %d,\n"
        "  \"mean_degree\": %.17g,\n"
        "  \"target_r0\": %.17g,\n"
        "  \"transmission_multiplier\": %.17g,\n"
        "  \"read_seconds\": %.17g,\n"
        "  \"setup_seconds\": %.17g,\n"
        "  \"simulate_seconds\": %.17g,\n"
        "  \"total_seconds\": %.17g,\n"
        "  \"final_susceptible\": %d,\n"
        "  \"final_exposed\": %d,\n"
        "  \"final_infected\": %d,\n"
        "  \"final_hospitalized\": %d,\n"
        "  \"final_recovered\": %d,\n"
        "  \"peak_hospitalized\": %d,\n"
        "  \"vaccinated\": %d,\n"
        "  \"vaccine_protected\": %d,\n"
        "  \"extract_seconds\": %.17g,\n"
        "  \"transmissions\": %d,\n"
        "  \"transitions_se\": %d,\n"
        "  \"transitions_ei\": %d,\n"
        "  \"transitions_ih\": %d,\n"
        "  \"transitions_ir\": %d,\n"
        "  \"transitions_hr\": %d,\n"
        "  \"daily_incidence\": [%s],\n"
        "  \"reproductive_number\": [%s],\n"
        "  \"fingerprint\": \"%s\",\n"
        "  \"timestamp_utc\": \"%s\"\n"
        "}\n",
        json_bytes(memory_baseline.rss).c_str(), json_bytes(memory_after_read.rss).c_str(),
        json_bytes(memory_after_setup.rss).c_str(), json_bytes(memory_after_setup.peak).c_str(),
        json_bytes(peak_reset ? memory_simulate.peak : -1).c_str(),
        args.at("engine_version").c_str(), n, days, integer("replicate"), integer("seed"),
        args.at("network_sha256").c_str(), integer("network_edges"), mean_degree, target_r0,
        number("transmission_multiplier"), read_seconds, total_seconds - simulate_seconds - extract_seconds, simulate_seconds,
        total_seconds, today.at("Susceptible"), today.at("Exposed"), today.at("Infected"),
        today.at("Hospitalized"), today.at("Recovered"), peak_hospitalized,
        vaccinated, protected_,
        extract_seconds, transmissions, transitions["SE"], transitions["EI"],
        transitions["IH"], transitions["IR"], transitions["HR"],
        incidence_json.c_str(), rt_json.c_str(),
        args.at("fingerprint").c_str(), timestamp
    );
    std::fclose(json);
    std::filesystem::rename(temporary, output);
    return 0;
}
