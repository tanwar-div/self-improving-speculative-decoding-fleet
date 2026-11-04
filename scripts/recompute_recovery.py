"""Recompute E2's recovery metric on a saved benchmark report.

The rolling throughput traces are stored in the report, so the metric can be
revised without re-running the hour of GPU work that produced them.

    PYTHONPATH=src .venv/bin/python scripts/recompute_recovery.py runs/latest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fleet.bench import recovery


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/latest.json")
    report = json.loads(path.read_text())
    rec = report.get("recovery")
    if not rec:
        print(f"{path} has no recovery section")
        raise SystemExit(1)

    setup = rec["setup"]
    analysis = recovery.analyse(
        rec["policies"],
        shift_index=setup["shift_index"],
        window=setup["rolling_window"],
        recovery_frac=setup["recovery_frac_of_oracle"],
        sustain=setup["sustain_requests"],
        oracle_tokens_per_s=rec.get("oracle_tokens_per_s"),
    )
    rec["recovery_vs_shared_plateau"] = analysis
    path.write_text(json.dumps(report, indent=2))

    print(
        f"shared target: {analysis['shared_target_tokens_per_s']} tok/s "
        f"({analysis['recovery_frac']:.0%} of the lower plateau); "
        f"oracle {analysis['oracle_tokens_per_s']} tok/s"
    )
    print(f"search starts at request {analysis['search_started_at_request']}\n")
    for name, p in analysis["policies"].items():
        r2r = p["requests_to_recover"]
        print(
            f"  {name:<18} plateau {p['post_shift_plateau_tokens_per_s']:7.2f} tok/s "
            f"({p['plateau_fraction_of_oracle']:.0%} of oracle)   "
            f"recovered@ {str(r2r) + ' req' if r2r is not None else 'never':>9}   "
            f"mean deficit {p['deficit_per_request_tokens_per_s']:6.2f} tok/s"
        )
    c = analysis["comparison"]
    print()
    if c["drift_faster_by_fraction"] is not None:
        print(
            f"  drift-triggered recovered {c['drift_faster_by_fraction']:.1%} faster "
            f"({c['drift_requests_to_recover']} vs {c['periodic_requests_to_recover']} requests)"
        )
    else:
        print(
            f"  requests-to-recover: periodic={c['periodic_requests_to_recover']}, "
            f"drift={c['drift_requests_to_recover']}"
        )
    if c["drift_deficit_lower_by_fraction"] is not None:
        print(
            f"  drift-triggered post-shift deficit {c['drift_deficit_lower_by_fraction']:.1%} "
            f"lower ({c['drift_deficit_per_request']} vs {c['periodic_deficit_per_request']} tok/s)"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
