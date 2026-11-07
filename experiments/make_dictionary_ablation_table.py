"""Render the discovered-dictionary ablation as the paper's LaTeX table.

Reads results/discovered_dictionary_ablation/{results.csv,dictionaries.json}
and emits a table in the style of tab:prediction-benchmark (medians over seeds,
one multirow block per system).

    python experiments/make_dictionary_ablation_table.py [--outdir ...]
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(_ROOT, "results", "discovered_dictionary_ablation")

SYSTEM_LABEL = {"lorenz": "Lorenz", "ikeda": "Ikeda",
                "standard_map": "Standard map", "cat_map": "Cat map"}

METHOD_LABEL = {
    "ours": r"Ours (no re-fit)",
    "edmd_disc": r"EDMD $\mathcal{T}$",
    "edmd_poly": r"EDMD deg-$D$",
    "edmd_oracle": r"EDMD target",
    "edmdlin_disc": r"EDMD$_{\mathrm{lift}}$ $\mathcal{T}$",
    "edmdlin_poly": r"EDMD$_{\mathrm{lift}}$ deg-$D$",
    "edmdlin_oracle": r"EDMD$_{\mathrm{lift}}$ target",
    "sindy_disc": r"SINDy $\mathcal{T}$",
    "sindy_poly": r"SINDy deg-$D$",
    "sindy_oracle": r"SINDy target",
    "kandy_disc": r"KANDy $\mathcal{T}$",
    "kandy_state": r"KANDy (state)",
}

ORDER = ["ours", "edmd_disc", "edmd_poly", "edmd_oracle",
         "edmdlin_disc", "edmdlin_poly", "edmdlin_oracle",
         "sindy_disc", "sindy_poly", "sindy_oracle",
         "kandy_disc", "kandy_state"]


def fnum(x):
    """LaTeX scientific notation, or --- for missing."""
    if x is None or not np.isfinite(x):
        return "---"
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def fsteps(x):
    if x is None:
        return "---"
    if not np.isfinite(x):
        return r"$\infty$"
    return f"${x:.0f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=DEFAULT)
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    dict_path = os.path.join(args.outdir, "dictionaries.json")
    dicts = json.load(open(dict_path)) if os.path.exists(dict_path) else {}

    by = defaultdict(list)
    for r in rows:
        by[(r["system"], r["method"])].append(r)

    systems = []
    for r in rows:
        if r["system"] not in systems:
            systems.append(r["system"])

    def med(key, sysname, method, cast=float):
        vals = []
        for r in by[(sysname, method)]:
            v = r.get(key, "")
            if v in ("", None):
                continue
            try:
                vals.append(cast(v))
            except ValueError:
                vals.append(float("inf"))
        return float(np.median(vals)) if vals else None

    out = []
    out.append(r"\begin{table}[ht]")
    out.append(r"\centering")
    out.append(r"{\color{blue}")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\renewcommand{\arraystretch}{1.1}")
    out.append(r"\begin{tabular}{@{}llrrr@{}}")
    out.append(r"\toprule")
    out.append(r"System & Library / method & Atoms & 1-step nMSE $\downarrow$ "
               r"& $\mathrm{VPT}_{0.5}$ (steps) $\uparrow$ \\")
    out.append(r"\midrule")

    for si, s in enumerate(systems):
        methods = [m for m in ORDER if (s, m) in by]
        out.append(f"\\multirow{{{len(methods)}}}{{*}}{{{SYSTEM_LABEL.get(s, s)}}}")
        for m in methods:
            atoms = med("n_atoms", s, m)
            nmse = med("one_step_nmse", s, m)
            vpt = med("vpt_0.5_steps", s, m)
            out.append(f" & {METHOD_LABEL.get(m, m)} & "
                       f"${atoms:.0f}$ & {fnum(nmse)} & {fsteps(vpt)} \\\\")
        out.append(r"\midrule" if si < len(systems) - 1 else r"\bottomrule")

    out.append(r"\end{tabular}")
    out.append(r"\caption{\rev{Discovered-dictionary ablation. $\mathcal{T}$ is "
               r"the union of the per-coordinate inner supports read off the "
               r"trained encoder (Algorithm~\ref{alg:deep-koopman-kandy}, "
               r"Phase~2--3); deg-$D$ is the full monomial library of the same "
               r"degree; target is the known dictionary where one exists. "
               r"EDMD$_{\mathrm{lift}}$ propagates inside the lift without "
               r"re-lifting and is the arm that tests invariance. Medians over "
               r"three seeds. Reproducible via "
               r"\texttt{experiments/discovered\_dictionary\_ablation.py}.}}")
    out.append(r"\label{tab:dictionary-ablation}")
    out.append(r"}")
    out.append(r"\end{table}")

    tex = "\n".join(out)
    tex_path = os.path.join(args.outdir, "table.tex")
    with open(tex_path, "w") as f:
        f.write(tex + "\n")
    print(tex)
    print(f"\n% wrote {tex_path}")

    if dicts:
        print("\n% --- discovered dictionaries (seed 0) ---")
        for k, v in dicts.items():
            if not k.endswith("_s0"):
                continue
            line = (f"% {v['system']}: {v['n_union']}/{v['n_full_library']} atoms "
                    f"(scaled rule {v['n_union_scaled']}), "
                    f"R2(support)={v.get('r2_support_median_raw', float('nan')):.3f}")
            if "target_recall" in v:
                line += (f", recall={v['target_recall']:.2f}, "
                         f"Jaccard={v['target_jaccard']:.2f}")
            print(line)
            print(f"%   {', '.join(v['atoms'])}")


if __name__ == "__main__":
    main()
