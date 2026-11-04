"""Render a benchmark JSON report as readable text."""

from __future__ import annotations


def _fmt(x, nd: int = 2, dash: str = "-") -> str:
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render(report: dict) -> str:
    lines: list[str] = []
    cfg = report.get("config", {})
    lines.append("=" * 78)
    lines.append("SELF-IMPROVING SPECULATIVE DECODING FLEET -- BENCHMARK")
    lines.append("=" * 78)
    lines.append(
        f"model {cfg.get('model')}  device {cfg.get('device')}  dtype {cfg.get('dtype')}  "
        f"gamma {cfg.get('gamma')}  max_new_tokens {cfg.get('max_new_tokens')}"
    )
    pool = report.get("prompt_pool", {})
    lines.append(
        f"prompt pool {pool.get('actual')} prompts across "
        f"{len(pool.get('per_domain', {}))} domains  (all from real datasets: "
        f"{pool.get('fully_real')})"
    )
    lines.append(
        f"decoded per arm {cfg.get('decode_prompts_per_arm')}  "
        f"static warm-up {cfg.get('warmup_prompts')}  "
        f"audit 1 in {cfg.get('audit_every')}"
    )
    lines.append(f"total wall time {_fmt(report.get('duration_s'), 1)}s")

    # ---- E1 --------------------------------------------------------------
    for key, title in (
        ("throughput_nonstationary", "E1a  THROUGHPUT -- NON-STATIONARY WORKLOAD (the real case)"),
        ("throughput_stationary", "E1b  THROUGHPUT -- STATIONARY WORKLOAD (control)"),
    ):
        tp = report.get(key)
        if not tp:
            continue
        lines.append("")
        lines.append("-" * 78)
        lines.append(title)
        lines.append("-" * 78)
        lines.append(f"{tp['stream_requests']} requests per arm; {tp['label']}")
        if len(tp.get("regime_boundaries", [])) > 1:
            regimes = cfg.get("regimes", [])
            desc = "  ".join(
                f"@{b}:{'+'.join(r)}"
                for b, r in zip(tp["regime_boundaries"], regimes, strict=False)
            )
            lines.append(f"regime changes: {desc}")
        lines.append("")
        lines.append(
            f"{'arm':<17}{'tok/s':>8}{'steady':>8}{'p50 ms':>8}{'p95 ms':>8}"
            f"{'accept':>8}{'tok/fwd':>9}{'greedy':>8}{'vers':>5}"
        )
        for name in ("no_speculation", "static_initial", "static_broad", "self_improving"):
            a = tp["arms"].get(name)
            if not a:
                continue
            audit = a["greedy_audit"]
            greedy = f"{audit['matched']}/{audit['audited']}" if audit["audited"] else "n/a"
            steady = tp["steady_state_tokens_per_s"].get(name)
            lines.append(
                f"{name:<17}{_fmt(a['fleet_tokens_per_s'], 1):>8}{_fmt(steady, 1):>8}"
                f"{_fmt(a['p50_ms'], 1):>8}{_fmt(a['p95_ms'], 1):>8}"
                f"{_fmt(a['acceptance_rate'], 3):>8}"
                f"{_fmt(a['tokens_per_target_forward'], 3):>9}"
                f"{greedy:>8}{len(a['distinct_versions_served']):>5}"
            )
        c = tp["comparisons"]
        lines.append("")
        lines.append(
            f"  static (deploy-time data) vs no speculation : {_fmt(c['static_initial_vs_no_speculation'], 3)}x"
        )
        lines.append(
            f"  static (all domains) vs no speculation      : {_fmt(c['static_broad_vs_no_speculation'], 3)}x"
        )
        lines.append(
            f"  self-improving vs no speculation            : {_fmt(c['self_improving_vs_no_speculation'], 3)}x"
        )
        lines.append(
            f"  self-improving vs static (deploy-time)      : {_fmt(c['self_improving_vs_static_initial'], 3)}x"
        )
        marker = (
            "   <-- headline"
            if key == "throughput_nonstationary"
            else "   <-- expect ~1.0 (no regression)"
        )
        lines.append(
            f"  self-improving vs static (all domains)      : {_fmt(c['self_improving_vs_static_broad'], 3)}x{marker}"
        )
        lines.append(
            f"    ... excluding cold start                  : {_fmt(c['steady_state_self_improving_vs_static_broad'], 3)}x"
        )
        lines.append(
            f"  p95 latency, self-improving / static_broad   : {_fmt(c['p95_self_improving_vs_static_broad'], 3)}x"
        )
        si = tp["arms"].get("self_improving", {})
        lines.append(
            f"  self-improving: {si.get('retrains')} retrains, {si.get('promotions')} promotions, "
            f"{si.get('rollbacks')} rollbacks, {_fmt(si.get('train_wall_s'), 1)}s training "
            f"(off the serving path)"
        )

    # ---- E2 --------------------------------------------------------------
    rec = report.get("recovery")
    if rec:
        setup = rec["setup"]
        lines.append("")
        lines.append("-" * 78)
        lines.append("E2  WORKLOAD-SHIFT RECOVERY")
        lines.append("-" * 78)
        lines.append(
            f"{'+'.join(setup['pre_shift_domains'])} ({setup['pre_shift_requests']} requests)"
            f" -> {'+'.join(setup['post_shift_domains'])}"
            f" ({setup['post_shift_requests']} requests),"
            f" shift at request {setup['shift_index']}"
        )
        lines.append(
            f"both policies run the identical prompt stream; only the retrain "
            f"trigger differs (periodic every {setup['periodic_every_n_requests']} requests)"
        )

        lines.append("")
        lines.append(f"{'policy':<18}{'retrains':>10}{'tok/s':>9}{'p95 ms':>9}{'accept':>8}")
        for name in ("periodic", "drift_triggered"):
            pol = rec["policies"].get(name)
            if not pol:
                continue
            lines.append(
                f"{name:<18}{pol['retrains']:>10}"
                f"{_fmt(pol['fleet_tokens_per_s'], 1):>9}"
                f"{_fmt(pol['p95_ms'], 1):>9}"
                f"{_fmt(pol['acceptance_rate'], 3):>8}"
            )

        shared = rec.get("recovery_vs_shared_plateau")
        if shared and shared.get("policies"):
            lines.append("")
            lines.append(
                f"recovery target: {shared['recovery_frac']:.0%} of the lower policy "
                f"plateau = {_fmt(shared['shared_target_tokens_per_s'], 1)} tok/s "
                f"(sustained {shared['sustain_requests']} requests)"
            )
            lines.append(
                f"for context, a frozen drafter pre-trained on post-shift traffic "
                f"reaches {_fmt(shared['oracle_tokens_per_s'], 1)} tok/s -- unreachable "
                f"for a live adaptive system, which pays a cold start, a rollout "
                f"latency and a continuous canary tax"
            )
            lines.append("")
            lines.append(
                f"{'policy':<18}{'plateau':>9}{'%oracle':>9}{'recover@':>11}{'deficit':>10}"
            )
            for name in ("periodic", "drift_triggered"):
                sp = shared["policies"].get(name)
                if not sp:
                    continue
                r2r = sp["requests_to_recover"]
                pct = sp["plateau_fraction_of_oracle"] or 0.0
                recovered = f"{r2r} req" if r2r is not None else "never"
                lines.append(
                    f"{name:<18}"
                    f"{_fmt(sp['post_shift_plateau_tokens_per_s'], 1):>9}"
                    f"{format(pct, '.0%'):>9}"
                    f"{recovered:>11}"
                    f"{_fmt(sp['deficit_per_request_tokens_per_s'], 2):>10}"
                )
            sc = shared["comparison"]
            lines.append("")
            if sc["drift_faster_by_fraction"] is not None:
                lines.append(
                    f"  drift-triggered recovered {sc['drift_faster_by_fraction']:.1%} "
                    f"faster ({sc['drift_requests_to_recover']} vs "
                    f"{sc['periodic_requests_to_recover']} requests)   <-- headline"
                )
            if sc["drift_deficit_lower_by_fraction"] is not None:
                lines.append(
                    f"  drift-triggered post-shift deficit "
                    f"{sc['drift_deficit_lower_by_fraction']:.1%} lower "
                    f"({_fmt(sc['drift_deficit_per_request'], 2)} vs "
                    f"{_fmt(sc['periodic_deficit_per_request'], 2)} tok/s)"
                )
            lines.append(
                f"  retrains spent: periodic {rec['comparison']['periodic_retrains']}, "
                f"drift-triggered {rec['comparison']['drift_retrains']}"
            )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    print(render(json.loads(open(sys.argv[1]).read())))
