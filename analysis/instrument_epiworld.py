"""Add phase timers to a copy of epiworld's model-meat.hpp.

Usage: python instrument_epiworld.py <include directory>

The timers accumulate wall time for each phase of Model::run() and print the
totals to stderr when the process exits. The anchors are the exact source
lines of the epiworld commit pinned in .devcontainer/Dockerfile; the script
fails if one is missing, rather than timing the wrong code.
"""

from __future__ import annotations

from pathlib import Path
import sys

PHASES = [
    # (label, code to time) in Model::reset()
    ("reset: reset each agent", "    for (auto & p : population)\n        p.reset();\n"),
    ("reset: build the state index", "    state_index_build();\n\n    #ifdef EPI_DEBUG\n    for (auto & a: population)"),
    ("reset: reset the database", "    db.reset();\n"),
    ("reset: clear the queue", "    if (use_queuing)\n        queue.reset();\n"),
    ("reset: seed and place tools", "    dist_entities();\n    dist_virus();\n    dist_tools();\n"),
    # in Model::run()
    ("run: reset() in total", "    reset();\n\n    // Record the baseline"),
    ("day: update_state()", "        this->update_state();\n"),
    ("day: next() (database record)", "        this->next();\n"),
    # in Model::update_state()
    ("update: push transmission", "        transmission_push();\n"),
    ("update: other states' updates", "        transmission_update_others();\n"),
    ("update: apply the day's events", "    events_run();\n\n}\n\ntemplate<typename TSeq>\ninline void Model<TSeq>::mutate_virus"),
]

HEADER = """#include <chrono>
#include <cstdio>
struct EpiProfTotals {
    static constexpr const char * names[] = {%s};
    double seconds[%d] = {0};
    ~EpiProfTotals() {
        for (int i = 0; i < %d; ++i)
            std::fprintf(stderr, "%%-32s %%8.3f ms\\n", names[i], seconds[i] * 1e3);
    }
};
inline EpiProfTotals & epiprof_totals() { static EpiProfTotals totals; return totals; }
struct EpiProfTimer {
    int phase;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    explicit EpiProfTimer(int phase_) : phase(phase_) {}
    ~EpiProfTimer() {
        epiprof_totals().seconds[phase] += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    }
};
"""


def main() -> None:
    path = Path(sys.argv[1]) / "epiworld" / "model-meat.hpp"
    source = path.read_text()
    for index, (_, code) in enumerate(PHASES):
        if source.count(code) != 1:
            raise SystemExit(f"anchor found {source.count(code)} times: {code!r}")
        # Time only the first statement(s) of the anchor: the anchors carry
        # trailing context to make them unique, which stays outside the block.
        timed, _, rest = code.partition("\n\n") if "\n\n" in code else (code, "", "")
        replacement = f"{{ EpiProfTimer _epiprof({index});\n{timed}\n}}"
        replacement += f"\n\n{rest}" if rest else "\n"
        source = source.replace(code, replacement)
    names = ", ".join(f'"{label}"' for label, _ in PHASES)
    path.write_text(HEADER % (names, len(PHASES), len(PHASES)) + source)


if __name__ == "__main__":
    main()
