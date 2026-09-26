// Single-replicate runner for the epiworld C++ library.
//
// The model is the one epiworldR's runner builds, written against the C++ API
// that epiworldR wraps. `make setup` compiles it the way epiworldR is compiled
// (-O2 -DNDEBUG -Depiworld_double=double), so the two differ only in the R
// layer.

#include <epiworld/epiworld.hpp>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <map>
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
    double recovery = 1.0 / infectious_days;
    double transmissibility = std::min(0.999, target_r0 / std::max(1.0, mean_degree - 1.0));
    double beta = transmissibility * recovery / (1.0 - transmissibility * (1.0 - recovery));
    beta *= number("transmission_multiplier");
    double hospital_rate = hospital_probability * recovery /
        (1.0 - hospital_probability * (1.0 - recovery));

    auto total_started = Clock::now();
    std::vector<int> source, target;
    read_edges(args.at("network"), source, target);
    if (static_cast<int>(source.size()) != integer("network_edges"))
        throw std::runtime_error("Network edge count mismatch");

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

    model.agents_from_edgelist(source, target, n, false);
    model.verbose_off();
    source = {};
    target = {};

    auto simulate_started = Clock::now();
    model.run(days, integer("seed"));
    double simulate_seconds = seconds_since(simulate_started);
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
        "  \"setup_seconds\": %.17g,\n"
        "  \"simulate_seconds\": %.17g,\n"
        "  \"total_seconds\": %.17g,\n"
        "  \"final_susceptible\": %d,\n"
        "  \"final_exposed\": %d,\n"
        "  \"final_infected\": %d,\n"
        "  \"final_hospitalized\": %d,\n"
        "  \"final_recovered\": %d,\n"
        "  \"peak_hospitalized\": %d,\n"
        "  \"fingerprint\": \"%s\",\n"
        "  \"timestamp_utc\": \"%s\"\n"
        "}\n",
        args.at("engine_version").c_str(), n, days, integer("replicate"), integer("seed"),
        args.at("network_sha256").c_str(), integer("network_edges"), mean_degree, target_r0,
        number("transmission_multiplier"), total_seconds - simulate_seconds, simulate_seconds,
        total_seconds, today.at("Susceptible"), today.at("Exposed"), today.at("Infected"),
        today.at("Hospitalized"), today.at("Recovered"), peak_hospitalized,
        args.at("fingerprint").c_str(), timestamp
    );
    std::fclose(json);
    std::filesystem::rename(temporary, output);
    return 0;
}
