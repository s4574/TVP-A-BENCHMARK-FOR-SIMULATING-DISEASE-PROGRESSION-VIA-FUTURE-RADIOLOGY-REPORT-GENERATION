"""Paper figures 3-8 (matplotlib, unified palette per dataviz reference; direct labels compensate
the yellow/teal contrast WARN). Outputs paper/iclr2027/figs/fig{3..8}.pdf(+png)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "${VP_ROOT}"
OUT = f"{ROOT}/paper/iclr2027/figs"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("->", name)

# ---------------- Fig 3: budget/memory sweep flatness (dumbbell) ----------------
cells = [("12 steps", .556, .4952), ("24 steps", .556, .4927), ("48 steps", .557, .4915),
         ("mem 0", .560, .4949), ("mem 12", .557, .5019), ("mem all", .557, .4931),
         ("+progress", .557, .4931)]
fig, ax = plt.subplots(figsize=(4.6, 2.2))
x = np.arange(len(cells))
for i, (n, v, s) in enumerate(cells):
    ax.plot([i, i], [s, v], color="#d9d8d3", lw=1, zorder=1)
ax.scatter(x, [c[1] for c in cells], s=22, facecolor=SURF, edgecolor=YELLOW, lw=1.4, zorder=3, label="internal val (searched)")
ax.scatter(x, [c[2] for c in cells], s=22, color=BLUE, zorder=3, label="sealed (scored once)")
ax.axhspan(.491, .502, color=BLUE, alpha=.08, zorder=0)
ax.text(len(cells)-.4, .4955, "sealed band\n0.491–0.502", fontsize=6.5, color=BLUE, ha="right", va="center")
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cells], rotation=20, ha="right")
ax.set_ylabel("state-direct micro-F1"); ax.set_ylim(.47, .58)
ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
save(fig, "fig3_sweep")

# ---------------- Fig 4: honesty scatter (val vs sealed) ----------------
pts = [  # (val, sealed, label)
    (.529, .499, "logreg (light search)"), (.554, .5012, "v1 mini"), (.557, .4992, "v1 astra"),
    (.554, .4914, "v2 mini"), (.555, .5023, "v2 astra"),
    (.556, .4952, ""), (.556, .4927, ""), (.557, .4915, ""), (.560, .4949, ""),
    (.557, .5019, ""), (.557, .4931, ""), (.557, .4931, ""),
]
fig, ax = plt.subplots(figsize=(2.9, 2.7))
lims = [.47, .58]
ax.plot(lims, lims, color="#d9d8d3", lw=1, ls="--")
ax.text(.545, .549, "y = x", fontsize=6.5, color=INK2, rotation=38)
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26, color=BLUE, alpha=.85)
ax.annotate("light search:\nsmall gap", xy=(.529, .499), xytext=(.482, .513), fontsize=6.5,
            color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
ax.annotate("intense search: val inflates,\nsealed does not move", xy=(.557, .496), xytext=(.492, .476),
            fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("best internal-val micro-F1 (search feedback)")
ax.set_ylabel("sealed micro-F1 (one-shot)")
save(fig, "fig4_honesty")

# ---------------- Fig 5: flicker (heatmap + delta lines) ----------------
M = np.array([[.366, .063, .026, .545], [.070, .327, .016, .587],
              [.173, .107, .087, .633], [.043, .090, .006, .861]])
lab = ["present", "absent", "uncertain", "not\nmentioned"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.2), gridspec_kw={"width_ratios": [1.05, 1]})
im = a1.imshow(M, cmap="Blues", vmin=0, vmax=.9)
for i in range(4):
    for j in range(4):
        a1.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if M[i, j] > .5 else INK)
a1.set_xticks(range(4)); a1.set_xticklabels(lab, fontsize=6.5)
a1.set_yticks(range(4)); a1.set_yticklabels(lab, fontsize=6.5)
a1.set_xlabel("label in target report"); a1.set_ylabel("state in prior reports")
a1.grid(False)
bins = ["≤7d", "7–30d", "30–120d", ">120d"]
a2.plot(bins, [.509, .520, .568, .572], color=BLUE, marker="o", ms=4, lw=1.6, label="prior present")
a2.plot(bins, [.562, .598, .594, .583], color=TEAL, marker="s", ms=4, lw=1.6, label="prior absent")
a2.text(0.02, .53, "51% already at ≤7d:\nreporting convention,\nnot resolution", fontsize=6.5, color=INK2,
        transform=a2.get_yaxis_transform())
a2.set_ylim(.40, .70); a2.set_ylabel("P(not_mentioned | prior state)")
a2.set_xlabel("time gap Δ to target exam")
a2.legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "fig5_flicker")

# ---------------- Fig 6: render spectrum (dot plot; truncated-axis bars are an anti-pattern) ----------------
names = ["end-to-end\n(matched 4k ctx)", "state-only\nrender", "+ctx\nanchored", "+ctx\nadvisory",
         "+mention-sel.\ninstruction", "advisory\n(full ~30k ctx)", "end-to-end\n(full ~30k ctx)"]
vals = [.5031, .5041, .5138, .5233, .5174, .5270, .5550]
cols = [ORANGE, BLUE, BLUE, BLUE, BLUE, BLUE, ORANGE]
fig, ax = plt.subplots(figsize=(4.6, 2.3))
for i, (v, c) in enumerate(zip(vals, cols)):
    ax.plot([i, i], [.495, v], color="#d9d8d3", lw=1, zorder=1)
    ax.scatter([i], [v], s=42, color=c, zorder=3, alpha=.55 if i >= 5 else 1.0)
    ax.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold" if i == 3 else "normal")
ax.axhline(.5031, color=ORANGE, lw=.8, ls=":")
ax.annotate("+0.020 from structured state\nat matched context", xy=(3, .5233), xytext=(0.8, .548),
            fontsize=6.8, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=.7))
ax.text(5.5, .565, "full-context pair", ha="center", fontsize=6.2, color=INK2)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6.3)
ax.set_ylabel("report micro-F1"); ax.set_ylim(.495, .575)
save(fig, "fig6_render")

# ---------------- Fig 7: trajectory draft ----------------
import csv
rows = list(csv.DictReader(open(f"{ROOT}/paper/figures_handoff/fig7_trajectory/trajectory.csv")))
fig, ax = plt.subplots(figsize=(4.8, 2.3))
best = [float(r["best_so_far"]) for r in rows]
steps = [int(r["step"]) for r in rows]
ax.step(steps, best, where="post", color=BLUE, lw=1.8, label="best so far", zorder=3)
sym = {"train": "o", "train_ensemble": "D", "train_transition": "^"}
seen = set()
for r in rows:
    v = r["val_micro"]
    if v:
        m = sym.get(r["action"], "o")
        lbl = r["action"] if r["action"] not in seen else None
        seen.add(r["action"])
        ax.scatter(int(r["step"]), float(v), marker=m, s=18, facecolor=SURF, edgecolor=INK2, lw=1, zorder=2, label=lbl)
    else:
        ax.plot([int(r["step"])], [.472], marker="|", color=INK2, ms=6)
for s, txt, tx, ty in [(0, "“cheap strong baseline first”", 2.5, .462),
                       (14, "“state_concepts …\navailable but risky”", 10.5, .500),
                       (21, "“only 3 iters left” → safe ensemble", 12.5, .578)]:
    row = rows[s]
    y = float(row["val_micro"]) if row["val_micro"] else .472
    ax.annotate(txt, xy=(s, y), xytext=(tx, ty), fontsize=6.2, color=INK2,
                bbox=dict(boxstyle="round,pad=0.25", fc="#fdf3d7", ec=YELLOW, lw=.8),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
ax.set_xlabel("search step"); ax.set_ylabel("internal-val micro-F1")
ax.set_ylim(.44, .60)
ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=4)
save(fig, "fig7_trajectory")

# ---------------- Fig 8: confidence faithfulness ----------------
order = ["Almost certain", "Highly likely", "Very good chance", "Likely",
         "Better than even", "Less than even", "Unlikely", "Chances are slight"]
short = ["almost\ncertain", "highly\nlikely", "very good\nchance", "likely",
         "better\nthan even", "less than\neven", "unlikely", "chances\nare slight"]
def f1s(by):
    out = []
    for c in order:
        d = by.get(c)
        if d and d["n"] >= 15:
            P = d["tp"] / max(d["tp"] + d["fp"], 1); R = d["tp"] / max(d["tp"] + d["fn"], 1)
            out.append(2 * P * R / max(P + R, 1e-9))
        else:
            out.append(np.nan)
    return out
raw = json.load(open(f"{ROOT}/analysis/confidence_faithfulness_raw.json"))
glm = {"Highly likely": .642, "Very good chance": .476, "Likely": .567, "Better than even": .537,
       "Less than even": .496, "Chances are slight": .438, "Unlikely": .350}
series = [("gpt-6-astra", f1s(raw["gpt-6-astra"]), ORANGE),
          ("gpt-5.4-mini", f1s(raw["gpt-5.4-mini"]), BLUE),
          ("claude-opus-4-7", f1s(raw["claude-opus-4-7"]), TEAL),
          ("glm-5.3-flash", [glm.get(c, np.nan) for c in order], YELLOW)]
fig, ax = plt.subplots(figsize=(4.8, 2.4))
xs = np.arange(len(order))
for name, ys, col in series:
    ax.plot(xs, ys, color=col, marker="o", ms=3.5, lw=1.5)
    ok = [i for i in range(len(ys)) if not np.isnan(ys[i])]
    ax.text(ok[-1] + .12, ys[ok[-1]], name, fontsize=6.5, color=col, va="center")
ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=6.2)
ax.set_xlabel("self-reported confidence (high → low)")
ax.set_ylabel("realized micro-F1"); ax.set_ylim(0, .75)
ax.text(6.0, .68, "faithful = downward slope\n(classes with n<15 omitted)", fontsize=6.3, color=INK2, ha="center")
save(fig, "fig8_confidence")
print("all figures done ->", OUT)
