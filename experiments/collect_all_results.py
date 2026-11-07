"""
Collect and print all ablation/baseline results + generate paper tables.

Usage:
    python experiments/collect_all_results.py
"""

import json
import csv
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent / 'results'

SEP = "=" * 75


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(x, decimals=4):
    if x is None or x == "?":
        return "—"
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def pm(mean, std, d=4):
    return f"{mean:.{d}f} ± {std:.{d}f}"


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 1
                      for i, h in enumerate(headers)]
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(w) for i, w in enumerate(col_widths)))


# ─────────────────────────────────────────────────────────────────────
# 1. Main Lorenz Ablation
# ─────────────────────────────────────────────────────────────────────

def section_lorenz_ablation():
    path = ROOT / 'ablation_lorenz' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    headers = ["Variant", "MSE", "Rollout 2τ", "Recall", "Precision", "F1",
               "FP(x²)", "FP(y²)", "FP(z²)", "FP(yz)"]
    rows = []
    for name, v in d.items():
        if not isinstance(v, dict) or 'seeds' not in v:
            continue
        seeds = v['seeds']
        mse_vals = [s['mse'] for s in seeds]
        roll_vals = [s['rollout_2tau'] for s in seeds]
        f1_vals = [s['support_f1'] for s in seeds if s.get('support_f1') is not None]
        det = v.get('per_term', {})
        recall = np.mean([sum(det.values()) / len(det)]) if det else "—"
        fp = v.get('per_fp', {})

        prec_vals = []
        for s in seeds:
            dr = s.get('det_rate')
            nfp = s.get('n_fp')
            if dr is not None and nfp is not None:
                tp = dr * 5
                prec_vals.append(tp / (tp + nfp) if (tp + nfp) > 0 else 0)

        rows.append([
            name[:25],
            pm(np.mean(mse_vals), np.std(mse_vals)),
            pm(np.mean(roll_vals), np.std(roll_vals)),
            fmt(recall),
            pm(np.mean(prec_vals), np.std(prec_vals)) if prec_vals else "—",
            pm(np.mean(f1_vals), np.std(f1_vals)) if f1_vals else "—",
            fmt(fp.get('x^2', 0)),
            fmt(fp.get('y^2', 0)),
            fmt(fp.get('z^2', 0)),
            fmt(fp.get('yz', 0)),
        ])
    print_table(headers, rows)

    print("\n  Per-seed detail:")
    for name, v in d.items():
        if not isinstance(v, dict) or 'seeds' not in v:
            continue
        print(f"\n  {name}:")
        for i, s in enumerate(v['seeds']):
            fp_str = ", ".join(s.get('false_positives', []))
            r2 = s.get('median_r2_hg')
            print(f"    s{i}: MSE={s['mse']:.6f}  roll={s['rollout_2tau']:.4f}  "
                  f"F1={fmt(s.get('support_f1'))}  R²(h∘g)={fmt(r2)}  FP=[{fp_str}]")


# ─────────────────────────────────────────────────────────────────────
# 2. Lorenz Pruned Ablation
# ─────────────────────────────────────────────────────────────────────

def section_lorenz_pruned():
    for label, fname in [("KANLinear", "results_pruned.json"),
                         ("PyKAN", "results_pruned_pykan.json")]:
        path = ROOT / 'ablation_lorenz' / fname
        if not path.exists():
            print(f"  [{label}: not found]")
            continue
        d = load_json(path)
        print(f"\n  {d.get('variant', label)}:")
        print(f"    MSE:      {pm(d['mse_mean'], d['mse_std'])}")
        print(f"    Rollout:  {pm(d['rollout_2tau_mean'], d['rollout_2tau_std'])}")
        print(f"    Jaccard:  {pm(d['jaccard_mean'], d['jaccard_std'])}")
        print(f"    Prec:     {fmt(d['precision_mean'])}")
        print(f"    Recall:   {fmt(d['recall_mean'])}")
        print(f"    Per-term: {d.get('per_term', {})}")
        print(f"    Per-FP:   {d.get('per_fp', {})}")
        for i, s in enumerate(d['seeds']):
            fp = ", ".join(s.get('false_positives', []))
            print(f"      s{i}: MSE={s['mse']:.6f}  roll={s['rollout_2tau']:.4f}  "
                  f"jac={s['jaccard']:.4f}  R²={fmt(s.get('median_r2_hg'))}  FP=[{fp}]")


# ─────────────────────────────────────────────────────────────────────
# 3. Lorenz Ablation with Score
# ─────────────────────────────────────────────────────────────────────

def section_lorenz_score():
    path = ROOT / 'ablation_lorenz_score' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    headers = ["Variant", "MSE", "Jaccard", "Prec", "Recall",
               "FP(x²)", "FP(y²)", "FP(z²)", "FP(yz)"]
    rows = []
    for name, v in d.items():
        fp = v.get('per_fp', {})
        rows.append([
            name[:25],
            fmt(v.get('mse_mean')),
            fmt(v.get('jaccard_mean')),
            fmt(v.get('precision_mean')),
            fmt(v.get('recall_mean')),
            fmt(fp.get('x^2', 0)), fmt(fp.get('y^2', 0)),
            fmt(fp.get('z^2', 0)), fmt(fp.get('yz', 0)),
        ])
    print_table(headers, rows)


# ─────────────────────────────────────────────────────────────────────
# 4. KS Equation Ablation
# ─────────────────────────────────────────────────────────────────────

def section_ks_ablation():
    path = ROOT / 'ablation_ks' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    headers = ["Variant", "Params", "MSE", "Horizon (steps)", "R²(h∘g)"]
    rows = []
    for name, v in d.items():
        r2_vals = [s.get('median_r2_hg') for s in v.get('seeds', [])
                   if s.get('median_r2_hg') is not None]
        rows.append([
            name[:25],
            v.get('n_params', '?'),
            pm(v.get('mse_mean', 0), v.get('mse_std', 0), 6),
            pm(v.get('horizon_mean', 0), v.get('horizon_std', 0), 1),
            fmt(np.mean(r2_vals)) if r2_vals else "—",
        ])
    print_table(headers, rows)

    print("\n  Per-seed:")
    for name, v in d.items():
        print(f"  {name}:")
        for i, s in enumerate(v.get('seeds', [])):
            print(f"    s{i}: MSE={s.get('mse',0):.6f}  horizon={s.get('horizon',0):.1f}  "
                  f"R²(h∘g)={fmt(s.get('median_r2_hg'))}")


# ─────────────────────────────────────────────────────────────────────
# 5. Baseline Comparison
# ─────────────────────────────────────────────────────────────────────

def section_baselines():
    path = ROOT / 'baseline_comparison' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    headers = ["Method", "MSE", "Horizon (τ_L)", "Det rate", "Avg FP"]
    rows = []
    for name, v in d.items():
        rows.append([
            name[:20],
            fmt(v.get('mse_mean')),
            pm(v.get('horizon_mean', 0), v.get('horizon_std', 0)),
            fmt(v.get('detection_rate')),
            fmt(v.get('avg_false_positives')),
        ])
    print_table(headers, rows)

    print("\n  Per-term detection rates:")
    for name, v in d.items():
        pt = v.get('per_term', {})
        pf = v.get('per_fp', {})
        print(f"  {name}: true={pt}  fp={pf}")


# ─────────────────────────────────────────────────────────────────────
# 6. SINDy-AE + PySR
# ─────────────────────────────────────────────────────────────────────

def section_sindy_ae():
    path = ROOT / 'sindy_ae_only' / 'results.csv'
    if not path.exists():
        print("  [not found]")
        return
    rows_raw = load_csv(path)

    print("  method | seed | MSE        | NRMSE_2tau | Jaccard | Recall | Detected")
    print("  " + "-" * 70)
    for r in rows_raw:
        print(f"  {r['method']:<16s} | {r['seed']:>4s} | "
              f"{float(r['one_step_mse']):10.6f} | {float(r['nrmse_2tau']):10.4f} | "
              f"{float(r['jaccard']):7.4f} | {float(r['recall']):6.2f} | {r['detected']}")


# ─────────────────────────────────────────────────────────────────────
# 7. EDMD-DL + PySR
# ─────────────────────────────────────────────────────────────────────

def section_edmd_dl():
    path = ROOT / 'edmd_dl_ablation' / 'results.csv'
    if not path.exists():
        print("  [not found]")
        return
    rows_raw = load_csv(path)

    print("  method | seed | MSE        | NRMSE_2tau | Jaccard | Recall | Detected")
    print("  " + "-" * 70)
    for r in rows_raw:
        print(f"  {r['method']:<16s} | {r['seed']:>4s} | "
              f"{float(r['one_step_mse']):10.6f} | {float(r['nrmse_2tau']):10.4f} | "
              f"{float(r['jaccard']):7.4f} | {float(r['recall']):6.2f} | {r['detected']}")


# ─────────────────────────────────────────────────────────────────────
# 8. External Ablation
# ─────────────────────────────────────────────────────────────────────

def section_external():
    path = ROOT / 'lorenz_external_ablation' / 'results.csv'
    if not path.exists():
        print("  [not found]")
        return
    rows_raw = load_csv(path)

    print("  method | seed | MSE        | NRMSE_2tau | Jaccard | Recall | Detected")
    print("  " + "-" * 80)
    for r in rows_raw:
        print(f"  {r['method']:<22s} | {r['seed']:>4s} | "
              f"{float(r['one_step_mse']):10.6f} | {float(r['nrmse_2tau']):10.4f} | "
              f"{float(r['jaccard']):7.4f} | {float(r['recall']):6.2f} | {r['detected']}")


# ─────────────────────────────────────────────────────────────────────
# 9. Spline Regularization Sweep
# ─────────────────────────────────────────────────────────────────────

def section_spline_reg():
    path = ROOT / 'spline_regularization_sweep' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    headers = ["Reg weight", "MSE", "Jaccard", "Recall", "FP(x²)", "FP(y²)"]
    rows = []
    for w, v in d.items():
        fp = v.get('per_fp', {})
        rows.append([
            w,
            pm(v.get('mse_mean', 0), v.get('mse_std', 0)),
            pm(v.get('jaccard_mean', 0), v.get('jaccard_std', 0)),
            fmt(v.get('recall_mean')),
            fmt(fp.get('x^2', 0)),
            fmt(fp.get('y^2', 0)),
        ])
    print_table(headers, rows)


# ─────────────────────────────────────────────────────────────────────
# 10. Pruning Stability
# ─────────────────────────────────────────────────────────────────────

def section_pruning_stability():
    path = ROOT / 'pruning_stability' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    print("  data_seed | model_seed | edges_pruned | total_edges")
    print("  " + "-" * 50)
    for r in d:
        print(f"  {r['data_seed']:>9d} | {r['model_seed']:>10d} | "
              f"{r['n_pruned']:>12d} | {r['n_total']:>11d}")

    from collections import defaultdict
    by_data = defaultdict(list)
    for r in d:
        by_data[r['data_seed']].append(r['n_pruned'])
    print("\n  Consistency check:")
    for ds, counts in sorted(by_data.items()):
        same = "YES" if len(set(counts)) <= 1 else f"NO (spread: {counts})"
        print(f"    data_seed={ds}: edges_pruned={counts}  identical={same}")


# ─────────────────────────────────────────────────────────────────────
# 11. PyKAN Pruned Sweep
# ─────────────────────────────────────────────────────────────────────

def section_pykan_sweep():
    path = ROOT / 'pykan_pruned_sweep' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    print(f"  Best alpha: {d.get('best_alpha')}")
    s = d.get('summary', {})
    for k, v in s.items():
        print(f"  {k}: {v}")


# ─────────────────────────────────────────────────────────────────────
# 12. Sparsity Decoder Analysis
# ─────────────────────────────────────────────────────────────────────

def section_sparsity_decoder():
    path = ROOT / 'sparsity_decoder_analysis' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    headers = ["Weight", "MSE", "Enc Prec", "Enc Recall", "Avg Sparsity",
               "Dec R²(x)", "Dec R²(y)", "Dec R²(z)"]
    rows = []
    for w, v in d.items():
        dec = v.get('dec_linear_r2', {})
        if isinstance(dec, str):
            dec = json.loads(dec)
        rows.append([
            w,
            fmt(v.get('one_step_mse'), 6),
            fmt(v.get('enc_prec')),
            fmt(v.get('enc_rec')),
            fmt(v.get('enc_avg_sparsity'), 1),
            fmt(dec.get('x')),
            fmt(dec.get('y')),
            fmt(dec.get('z')),
        ])
    print_table(headers, rows)
    print("  Note: sparsity does NOT change — encoder support is always "
          "{x,x²,xy,xz,y,y²,yz,z,z²}")


# ─────────────────────────────────────────────────────────────────────
# 13. Lorenz Benchmark
# ─────────────────────────────────────────────────────────────────────

def section_lorenz_benchmark():
    path = ROOT / 'lorenz_benchmark' / 'results.json'
    if not path.exists():
        print("  [not found]")
        return
    d = load_json(path)

    headers = ["Method", "MSE (seed avg)", "MSE std"]
    rows = []
    for name, v in d.items():
        seeds = v.get('seeds', v.get('per_seed', []))
        mses = []
        if isinstance(seeds, list):
            for s in seeds:
                m = s.get('mse', s.get('one_step_mse'))
                if m is not None and m > 0:
                    mses.append(m)
        elif isinstance(seeds, dict):
            for s in seeds.values():
                m = s.get('mse', s.get('one_step_mse'))
                if m is not None and m > 0:
                    mses.append(m)
        if not mses and 'mse_mean' in v:
            mses = [v['mse_mean']]
        if mses:
            rows.append([name[:20], fmt(np.mean(mses), 6), fmt(np.std(mses), 6)])
        else:
            rows.append([name[:20], "—", "—"])
    print_table(headers, rows)


# ─────────────────────────────────────────────────────────────────────
# 14. Paper Table 1: Main ablation (LaTeX)
# ─────────────────────────────────────────────────────────────────────

def section_paper_table_1():
    """Table 1 (\\label{tab:ablation-main}): Main ablation + baselines on Lorenz.

    7 columns: Variant, Params, Wall, MSE, NRMSE_2tau, Jaccard, Recall.
    External baselines (EDMD-DL, SINDy-AE), architectural ablations (5 variants),
    and our method (l1-reg + pruned).
    """

    # EDMD-DL
    rows_csv = load_csv(ROOT / 'edmd_dl_ablation' / 'results.csv')
    edmd_mse = np.mean([float(r['one_step_mse']) for r in rows_csv])
    edmd_nrmse = [float(r['nrmse_2tau']) for r in rows_csv]
    edmd_jac = [float(r['jaccard']) for r in rows_csv]
    edmd_rec = np.mean([float(r['recall']) for r in rows_csv])
    edmd_params = int(rows_csv[0]['n_params'])
    edmd_wall = 3.9  # hours, from run log

    # SINDy-AE
    rows_csv = load_csv(ROOT / 'sindy_ae_only' / 'results.csv')
    sindy_mse = np.mean([float(r['one_step_mse']) for r in rows_csv])
    sindy_nrmse = [float(r['nrmse_2tau']) for r in rows_csv]
    sindy_jac = [float(r['jaccard']) for r in rows_csv]
    sindy_rec = np.mean([float(r['recall']) for r in rows_csv])
    sindy_params = int(rows_csv[0]['n_params'])
    sindy_wall = 11.1  # hours, from run log

    # Architecture ablations
    score = load_json(ROOT / 'ablation_lorenz_score' / 'results.json')
    full = load_json(ROOT / 'ablation_lorenz' / 'results.json')

    def ablation_row(label, key):
        vs = score[key]
        vf = full[key]
        seeds = vf['seeds']
        rolls = [s['rollout_2tau'] for s in seeds]
        times = [s['wall_time'] for s in seeds if s.get('wall_time') is not None]
        return {
            'label': label, 'mse': vs['mse_mean'],
            'nrmse_mean': np.mean(rolls), 'nrmse_std': np.std(rolls),
            'jac_mean': vs['jaccard_mean'], 'jac_std': vs['jaccard_std'],
            'rec': vs['recall_mean'], 'params': vf.get('n_params', '?'),
            'wall_hrs': np.sum(times) / 3600 if times else None,
        }

    abl = {
        '1L':   ablation_row('1L-KAN + stable',              'B: 1L-KAN + stable'),
        'Denc': ablation_row('KAN-enc + MLP-dec',             'D: KAN-enc + MLP-dec'),
        'Edec': ablation_row('MLP-enc + KAN-dec',             'E: MLP-enc + KAN-dec'),
        'MLP':  ablation_row('MLP + stable',                  'C: MLP + stable'),
        'unc':  ablation_row('2L-KAN + unconstrained',        'F: 2L-KAN + unconstrained'),
        'l1':   ablation_row('Deep-Koopman-KANDy ($\\ell_1$-reg)', 'A: 2L-KAN + stable'),
    }

    # Pruned (pykan_pruned_sweep)
    sweep = load_json(ROOT / 'pykan_pruned_sweep' / 'results.json')
    pr_mses   = [s['mse']        for s in sweep['stage2']]
    pr_nrmses = [s['nrmse_2tau'] for s in sweep['stage2']]
    pr_jacs   = [s['jaccard']    for s in sweep['stage2']]
    pr_recs   = [s['recall']     for s in sweep['stage2']]
    pr_times  = [s['wall_time']  for s in sweep['stage2']]
    pr_params = 800
    pr_wall   = np.sum(pr_times) / 3600

    # ── Plain text ──
    print("\n  Verified numbers for Table 1:\n")
    hdr = (f"  {'Row':<32s} {'Params':>6s} {'Wall(h)':>7s} {'MSE':>10s} "
           f"{'NRMSE_2tau':>16s} {'Jaccard':>14s} {'Rec':>6s}")
    print(hdr)
    print("  " + "-" * len(hdr))

    def row(label, params, wall, mse, nm, ns, jm, js, r):
        wall_s = f"{wall:.1f}" if wall is not None else "—"
        return (f"  {label:<32s} {params:>6s} {wall_s:>7s} {mse:>10.2e} "
                f"{nm:>6.2f} ± {ns:<5.2f}   "
                f"{jm:>5.2f} ± {js:<5.2f}  {r:>5.2f}")

    print(row("EDMD-DL", str(edmd_params), edmd_wall, edmd_mse,
              np.mean(edmd_nrmse), np.std(edmd_nrmse),
              np.mean(edmd_jac), np.std(edmd_jac), edmd_rec))
    print(row("SINDy-AE", str(sindy_params), sindy_wall, sindy_mse,
              np.mean(sindy_nrmse), np.std(sindy_nrmse),
              np.mean(sindy_jac), np.std(sindy_jac), sindy_rec))
    print()
    for key in ['1L', 'Denc', 'Edec', 'MLP', 'unc']:
        a = abl[key]
        print(row(a['label'], str(a['params']), a['wall_hrs'], a['mse'],
                  a['nrmse_mean'], a['nrmse_std'],
                  a['jac_mean'], a['jac_std'], a['rec']))
    print()
    a = abl['l1']
    print(row(a['label'], str(a['params']), a['wall_hrs'], a['mse'],
              a['nrmse_mean'], a['nrmse_std'],
              a['jac_mean'], a['jac_std'], a['rec']))
    print(row("Deep-Koopman-KANDy (pruned)", str(pr_params), pr_wall,
              np.mean(pr_mses),
              np.mean(pr_nrmses), np.std(pr_nrmses),
              np.mean(pr_jacs), np.std(pr_jacs), np.mean(pr_recs)))

    # ── LaTeX ──
    def texmse(v):
        exp = int(np.floor(np.log10(abs(v))))
        mantissa = v / 10**exp
        return f"${mantissa:.2f}{{\\times}}10^{{{exp}}}$"

    def texpm(m, s):
        return f"${m:.2f} \\pm {s:.2f}$"

    def texwall(h):
        if h is None:
            return "—"
        return f"${h*60:.0f}$m" if h < 1 else f"${h:.1f}$h"

    ncols = 7
    print(f"\n\n  % ── LaTeX for Table 1 ──")
    print(f"  % \\begin{{tabular}}{{@{{}}lrccccc@{{}}}}\n")

    lines = []
    lines.append(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{External baselines}}}} \\\\")

    lines.append(f"EDMD-DL & ${edmd_params:,}$ & {texwall(edmd_wall)}"
                 f" & {texmse(edmd_mse)}"
                 f" & $\\mathbf{{{np.mean(edmd_nrmse):.2f} \\pm {np.std(edmd_nrmse):.2f}}}$"
                 f" & {texpm(np.mean(edmd_jac), np.std(edmd_jac))}"
                 f" & ${edmd_rec:.2f}$ \\\\")

    lines.append(f"SINDy-AE & ${sindy_params:,}$ & {texwall(sindy_wall)}"
                 f" & $\\mathbf{{{texmse(sindy_mse)[1:-1]}}}$"
                 f" & {texpm(np.mean(sindy_nrmse), np.std(sindy_nrmse))}"
                 f" & {texpm(np.mean(sindy_jac), np.std(sindy_jac))}"
                 f" & ${sindy_rec:.2f}$ \\\\")

    lines.append("")
    lines.append("\\addlinespace[2pt]")
    lines.append(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{Architectural ablations}}}} \\\\")

    for key in ['1L', 'Denc', 'Edec', 'MLP', 'unc']:
        a = abl[key]
        rec_str = f"\\mathbf{{{a['rec']:.2f}}}" if a['rec'] >= 0.99 else f"{a['rec']:.2f}"
        lines.append(f"{a['label']} & ${a['params']:,}$ & {texwall(a['wall_hrs'])}"
                     f" & {texmse(a['mse'])}"
                     f" & {texpm(a['nrmse_mean'], a['nrmse_std'])}"
                     f" & {texpm(a['jac_mean'], a['jac_std'])}"
                     f" & ${rec_str}$ \\\\")

    lines.append("")
    lines.append("\\addlinespace[2pt]")
    lines.append(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{Ours}}}} \\\\")

    a = abl['l1']
    lines.append(f"Deep-Koopman-KANDy ($\\ell_1$-reg)"
                 f" & ${a['params']:,}$ & {texwall(a['wall_hrs'])}"
                 f" & {texmse(a['mse'])}"
                 f" & {texpm(a['nrmse_mean'], a['nrmse_std'])}"
                 f" & \\underline{{{texpm(a['jac_mean'], a['jac_std'])}}}"
                 f" & $\\mathbf{{{a['rec']:.2f}}}$ \\\\")

    lines.append(f"\\textbf{{Deep-Koopman-KANDy (pruned)}}"
                 f" & ${pr_params:,}$ & {texwall(pr_wall)}"
                 f" & {texmse(np.mean(pr_mses))}"
                 f" & {texpm(np.mean(pr_nrmses), np.std(pr_nrmses))}"
                 f" & $\\mathbf{{{np.mean(pr_jacs):.2f} \\pm {np.std(pr_jacs):.2f}}}$"
                 f" & $\\mathbf{{{np.mean(pr_recs):.2f}}}$ \\\\")

    for line in lines:
        print(f"  {line}")


# ─────────────────────────────────────────────────────────────────────
# 15. Paper Table: Level-set decomposition (Appendix)
# ─────────────────────────────────────────────────────────────────────

def section_paper_table_levelset():
    """Table 2 (\\label{tab:levelset-decomposition}): Per-dim h∘g decomposition.

    Appendix table. Single-seed pruned model (lorenz_5d_third_test/model_pruned_0.03.pt).
    Lasso alpha=1e-5, degree-3 polynomial basis, top-5 coefficients per dim.
    Reproduces by loading the checkpoint and running manifold_analysis.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from models import PyKANAutoEncoder
    from lorenz_stable import DeepKoopmanCTStable
    from data import simulate_lorenz
    from training import set_seed
    from manifold_analysis import (
        compute_network_gradient_nd, recover_inner_function_nd,
        eval_polynomial_nd, grad_polynomial_nd,
        recover_outer_function_from_raw, eval_h,
    )

    ckpt_path = ROOT / 'lorenz_5d_third_test' / 'model_pruned_0.03.pt'
    if not ckpt_path.exists():
        print("  [checkpoint not found]")
        return

    ae = PyKANAutoEncoder(3, 5, 5, grid=5, k=3)
    model = DeepKoopmanCTStable(ae, 5)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    set_seed(42)
    traj = simulate_lorenz(n_traj=60, n_steps=6000, dt=0.01, burn_in=500, device="cpu")
    x_flat = traj[:48].reshape(-1, 3)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(x_flat), 10000, replace=False)
    x_data = torch.tensor(x_flat[idx].numpy(), dtype=torch.float32)
    x_np = x_data.numpy()

    ALPHA = 1e-5
    target = {'x', 'y', 'z', 'xy', 'xz'}

    print(f"\n  Model: lorenz_5d_third_test/model_pruned_0.03.pt")
    print(f"  Lasso alpha={ALPHA}, degree=3, top-5 coefficients per dim\n")

    # ── Plain text ──
    print(f"  {'Dim':>4s} | {'R²(h∘g)':>8s} | {'x':>8s} | {'y':>8s} | "
          f"{'z':>8s} | {'xy':>8s} | {'xz':>8s} | FP")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}")

    results = []
    for k in range(5):
        grad_f, f_vals = compute_network_gradient_nd(model, x_data, k, 'latent')
        inner = recover_inner_function_nd(x_np, ['x', 'y', 'z'], f_vals,
                                           max_degree=3, alpha=ALPHA)
        g_vals = eval_polynomial_nd(x_np, inner['coeffs'], inner['powers'])
        grad_g = grad_polynomial_nd(x_np, inner['coeffs'], inner['powers'])
        dot = np.sum(grad_f * grad_g, axis=1)
        norm_sq = np.sum(grad_g**2, axis=1)
        outer = recover_outer_function_from_raw(dot, norm_sq, g_vals, f_vals,
                                                 poly_degree=4)
        f_hat = eval_h(g_vals, outer)
        r2_hg = 1.0 - np.sum((f_vals - f_hat)**2) / max(
            np.sum((f_vals - f_vals.mean())**2), 1e-12)

        # Top 5 by magnitude (excluding constant)
        pairs = [(nm, float(cf)) for nm, cf in
                 zip(inner['active_names'], inner['active_coeffs']) if nm != '1']
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top5 = pairs[:5]
        coeff_map = {nm: cf for nm, cf in top5}
        fp_in_top5 = [nm for nm, _ in top5 if nm not in target]

        def f(name):
            v = coeff_map.get(name, 0)
            return '---' if abs(v) < 1e-3 else f'{v:+.3f}'

        fp_str = ', '.join(fp_in_top5) if fp_in_top5 else '---'

        print(f"  z_{k} | {r2_hg:8.3f} | {f('x'):>8s} | {f('y'):>8s} | "
              f"{f('z'):>8s} | {f('xy'):>8s} | {f('xz'):>8s} | {fp_str}")

        results.append({
            'dim': k, 'r2_hg': r2_hg, 'coeff_map': coeff_map,
            'fp': fp_in_top5,
        })

    # Union check
    all_top5 = set()
    for r in results:
        all_top5.update(r['coeff_map'].keys())
    found = target & all_top5
    fp_union = all_top5 - target
    print(f"\n  Union of top-5: {sorted(all_top5)}")
    print(f"  Target found: {sorted(found)} ({len(found)}/5)")
    print(f"  FP in union: {sorted(fp_union)}")

    # ── LaTeX ──
    print(f"\n\n  % ── LaTeX for Appendix level-set table ──\n")
    for r in results:
        k = r['dim']
        cm = r['coeff_map']
        fp = r['fp']

        def lf(name):
            v = cm.get(name, 0)
            return '---' if abs(v) < 1e-3 else f'${v:+.3f}$'

        fp_tex = ', '.join(f'${f}$' for f in fp) if fp else '---'

        print(f"  $z_{k}$ & ${r['r2_hg']:.3f}$ & "
              f"{lf('x')} & {lf('y')} & {lf('z')} & "
              f"{lf('xy')} & {lf('xz')} & {fp_tex} \\\\")


# ─────────────────────────────────────────────────────────────────────
# 16. Paper Table: Lorenz single-seed summary (Appendix)
# ─────────────────────────────────────────────────────────────────────

def section_paper_table_lorenz_summary():
    """Table 3 (\\label{tab:lorenz_results}): Performance summary of the best Lorenz model.

    Appendix table. PyKAN [3,5,5] + stable propagator, after prune@0.03 + retrain.
    Eigenvalues, prediction horizon, dictionary recovery (Lasso alpha=5e-3).
    Sources: lorenz_5d_best_model/summary.json, lorenz_5d_third_test/alpha_sweep.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    sm_path = ROOT / 'lorenz_5d_best_model' / 'summary.json'
    asw_path = ROOT / 'lorenz_5d_third_test' / 'alpha_sweep_all_models.json'

    if not sm_path.exists() or not asw_path.exists():
        print("  [not found]")
        return

    sm = load_json(sm_path)
    asw = load_json(asw_path)

    # Post-prune eigenvalues
    mags = [ev['abs'] for ev in sm['eigenvalues_discrete']]
    re_mus = [ev['real'] for ev in sm['eigenvalues_generator']]
    gen_freqs = [ev for ev in sm['eigenvalues_generator'] if abs(ev['freq_hz']) > 0.01]

    # Dictionary at alpha=5e-3
    p = asw['prune@0.03']['0.005']

    # Pre-prune MSE
    d3 = load_json(ROOT / 'lorenz_5d_third_test' / 'results.json')

    print(f"\n  Model: PyKAN [3,5,5] + stable, prune@0.03 + retrain")
    print(f"  Source: lorenz_5d_best_model/summary.json\n")

    rows = [
        ("Architecture",             f"PyKAN [3, 5, 5] + K = -L^T L"),
        ("Parameters",               f"{sm.get('n_params', 1593)}"),
        ("Pruning",                   f"threshold 0.03, retrain 100 epochs"),
        ("---", "---"),
        ("Pre-prune MSE",            f"{d3['pre_prune_mse']:.4e}"),
        ("Prediction horizon",       f"{sm['prediction_horizon_tau']:.3f} tau_L"),
        ("---", "---"),
        ("|lambda| range",           f"[{min(mags):.3f}, {max(mags):.3f}]"),
        ("max Re(mu)",               f"{max(re_mus):.2f}"),
        ("Eigenmodes",               f"±{abs(gen_freqs[0]['freq_hz']):.2f} Hz (spiral), "
                                      f"{sum(1 for ev in sm['eigenvalues_generator'] if abs(ev['imag'])<0.01)} real damping"),
        ("---", "---"),
        ("Dict (Lasso a=5e-3, deg 3)", f"{{{', '.join(p['true_found'])}}}"),
        ("Union Jaccard",            f"{p['jaccard']}"),
        ("Recall",                   f"{p['recall']*100:.0f}%"),
        ("Precision",                f"{p['precision']*100:.0f}%"),
        ("Mean FP/dim",              f"{p['mean_fp']}"),
    ]

    for label, val in rows:
        if label == "---":
            print(f"  {'─'*50}")
        else:
            print(f"  {label:<35s}  {val}")

    # Regression R² from post-prune
    print(f"\n  Eigenmode regression R²:")
    for term, val in sm['regression_r2'].items():
        print(f"    {term}: {val:.4f}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

SECTIONS = [
    ("1. LORENZ ABLATION (7 variants × 5 seeds)",     section_lorenz_ablation),
    ("2. LORENZ PRUNED ABLATION",                      section_lorenz_pruned),
    ("3. LORENZ ABLATION WITH SCORE",                  section_lorenz_score),
    ("4. KS EQUATION ABLATION (7 variants × 3 seeds)", section_ks_ablation),
    ("5. BASELINE COMPARISON (5 seeds)",               section_baselines),
    ("6. SINDy-AE + PySR (5 seeds)",                   section_sindy_ae),
    ("7. EDMD-DL + PySR (5 seeds)",                    section_edmd_dl),
    ("8. EXTERNAL ABLATION (KANDy vs MLP vs SINDy)",   section_external),
    ("9. SPLINE REGULARIZATION SWEEP",                 section_spline_reg),
    ("10. PRUNING STABILITY (3 data × 4 model seeds)", section_pruning_stability),
    ("11. PYKAN PRUNED SWEEP",                         section_pykan_sweep),
    ("12. SPARSITY DECODER ANALYSIS",                  section_sparsity_decoder),
    ("13. LORENZ BENCHMARK",                           section_lorenz_benchmark),
    ("14. TABLE 1: tab:ablation-main (main ablation)",  section_paper_table_1),
    ("15. TABLE 2: tab:levelset-decomposition (h∘g)",  section_paper_table_levelset),
    ("16. TABLE 3: tab:lorenz_results (model summary)", section_paper_table_lorenz_summary),
]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', type=int, default=None,
                        help='Run only this section (1-indexed)')
    args = parser.parse_args()

    print(SEP)
    print("  ALL ABLATION & BASELINE RESULTS")
    print(f"  Results directory: {ROOT}")
    print(SEP)

    for i, (title, fn) in enumerate(SECTIONS, 1):
        if args.section is not None and i != args.section:
            continue
        print(f"\n{SEP}")
        print(f"  {title}")
        print(SEP)
        try:
            fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)
