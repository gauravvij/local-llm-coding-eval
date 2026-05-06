"""
generate_visuals.py
Generates 4 dark-minimalist comparison charts for the local LLM evaluation blog post.
Output: /root/local_coding_eval/images/*.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
GRID    = "#30363d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"

# One accent per model — cohesive GitHub-dark palette
COLORS = {
    "qwen3.6:27b":        "#58a6ff",   # blue
    "qwen3.6:35b-a3b":    "#d2a8ff",   # violet
    "qwen3-coder:30b":    "#3fb950",   # green
    "deepseek-coder:33b": "#f78166",   # coral
}

MODELS      = list(COLORS.keys())
MODEL_SHORT = ["qwen3.6\n27b", "qwen3.6\n35b-a3b", "qwen3-coder\n30b", "deepseek-coder\n33b"]
PALETTE     = list(COLORS.values())

# ── Evaluation data (exact from blog.md summary table) ───────────────────────
data = {
    "Code Generation (%)":  [80,    70,    80,    90   ],
    "Tool Selection (%)":   [84.62, 84.62, 76.92, 84.62],
    "Param Accuracy (%)":   [84.62, 84.62, 69.23, 69.23],
    "Agent Accuracy (%)":   [100,   100,   80,    10   ],
    "Reasoning Score (/3)": [0.3,   1.8,   2.8,   1.8  ],
}

# ── Shared style helpers ──────────────────────────────────────────────────────
def base_fig(w=12, h=7):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

def style_ax(ax, ylim=None):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=11)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(ylim)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {name}  ({size_kb:.1f} KB)")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Grouped bar chart: all 4 metrics side by side
# ═══════════════════════════════════════════════════════════════════════════════
def chart_grouped_bars():
    metrics = ["Code Gen", "Tool Select", "Param Acc", "Agent Acc"]
    metric_keys = [
        "Code Generation (%)",
        "Tool Selection (%)",
        "Param Accuracy (%)",
        "Agent Accuracy (%)",
    ]

    n_metrics = len(metrics)
    n_models  = len(MODELS)
    x         = np.arange(n_metrics)
    bar_w     = 0.18
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    fig = base_fig(13, 7)
    ax  = fig.add_axes([0.08, 0.12, 0.88, 0.74])
    style_ax(ax, ylim=(0, 118))

    for i, (model, color, offset) in enumerate(zip(MODELS, PALETTE, offsets)):
        vals = [data[k][i] for k in metric_keys]
        bars = ax.bar(x + offset, vals, bar_w * 0.88, color=color, alpha=0.90,
                      label=model, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.0f}%", ha='center', va='bottom',
                    color=color, fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=TEXT, fontsize=12)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=SUBTEXT, fontsize=10)
    ax.set_ylabel("Score (%)", color=SUBTEXT, fontsize=11, labelpad=8)

    ax.set_title("Model Performance Across All Benchmarks",
                 color=TEXT, fontsize=15, fontweight='bold', pad=16,
                 loc='left')

    legend = ax.legend(
        loc='upper right', framealpha=0, labelcolor=TEXT,
        fontsize=10, handlelength=1.2, handleheight=0.9,
        borderpad=0.6, labelspacing=0.4
    )

    fig.text(0.5, 0.02,
             "Evaluated on CPU-only hardware  ·  Ollama inference backend",
             ha='center', color=SUBTEXT, fontsize=9)

    save(fig, "chart_all_metrics.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Radar / spider chart
# ═══════════════════════════════════════════════════════════════════════════════
def chart_radar():
    categories = ["Code Gen", "Tool Select", "Param Acc", "Agent Acc", "Reasoning"]
    # Normalise all to 0-100 (reasoning /3 → ×33.33)
    norm_data = []
    for i in range(len(MODELS)):
        norm_data.append([
            data["Code Generation (%)"][i],
            data["Tool Selection (%)"][i],
            data["Param Accuracy (%)"][i],
            data["Agent Accuracy (%)"][i],
            data["Reasoning Score (/3)"][i] / 3 * 100,
        ])

    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 9), facecolor=BG)
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(PANEL)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"],
                       color=SUBTEXT, fontsize=8)
    ax.yaxis.set_tick_params(pad=26)
    ax.set_rlabel_position(15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT, fontsize=12, fontweight='bold')

    ax.grid(color=GRID, linewidth=0.7, linestyle='--', alpha=0.6)
    ax.spines['polar'].set_color(GRID)

    for i, (model, color, vals) in enumerate(zip(MODELS, PALETTE, norm_data)):
        v = vals + vals[:1]
        ax.plot(angles, v, color=color, linewidth=2.2, zorder=3)
        ax.fill(angles, v, color=color, alpha=0.10, zorder=2)
        ax.scatter(angles[:-1], vals, color=color, s=50, zorder=4)

    ax.set_title("Capability Profile by Model",
                 color=TEXT, fontsize=15, fontweight='bold', pad=30)

    handles = [mpatches.Patch(color=c, label=m) for m, c in zip(MODELS, PALETTE)]
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.16), ncol=2,
              framealpha=0, labelcolor=TEXT, fontsize=10,
              handlelength=1.2, borderpad=0.6, labelspacing=0.5)

    fig.tight_layout()
    save(fig, "chart_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Agent accuracy spotlight
# ═══════════════════════════════════════════════════════════════════════════════
def chart_agent_spotlight():
    agent_scores = data["Agent Accuracy (%)"]   # [100, 100, 80, 10]

    fig = base_fig(11, 6.5)
    ax  = fig.add_axes([0.10, 0.12, 0.86, 0.72])
    style_ax(ax, ylim=(0, 125))

    x    = np.arange(len(MODELS))
    bars = ax.bar(x, agent_scores, width=0.50, color=PALETTE, alpha=0.92, zorder=3)

    for bar, val, color in zip(bars, agent_scores, PALETTE):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.5,
                f"{val}%",
                ha='center', va='bottom',
                color=color, fontsize=24, fontweight='bold')

    # Callout for the 10% bar
    ax.annotate(
        "Code-focused model\nstruggles with\nmulti-step reasoning",
        xy=(3, 12), xytext=(2.28, 72),
        color=PALETTE[3], fontsize=10,
        arrowprops=dict(arrowstyle='->', color=PALETTE[3], lw=1.5),
        ha='center'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT, color=TEXT, fontsize=11)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=SUBTEXT, fontsize=10)
    ax.set_ylabel("Tasks Completed Correctly (%)", color=SUBTEXT, fontsize=11, labelpad=8)
    ax.set_title("Agent Task Accuracy  ·  10 Multi-Step Tasks",
                 color=TEXT, fontsize=15, fontweight='bold', pad=16, loc='left')

    ax.axhline(100, color=GRID, linewidth=0.8, linestyle='--', zorder=1)

    fig.text(0.5, 0.02,
             "deepseek-coder:33b scored 90% on code generation yet only 10% on agentic tasks",
             ha='center', color=SUBTEXT, fontsize=9, style='italic')

    save(fig, "chart_agent_spotlight.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Recommendation grid (use-case heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_recommendation_grid():
    use_cases = [
        "Coding Agent\n(tools + planning)",
        "Pure Code\nGeneration",
        "Reasoning\nTransparency",
        "Balanced\nAll-Round",
        "CPU Speed\n(MoE efficiency)",
    ]

    # 0=poor  1=ok  2=good  3=best
    scores = np.array([
        # qwen3.6:27b  qwen3.6:35b-a3b  qwen3-coder:30b  deepseek-coder:33b
        [3,            3,                2,                0],   # Coding Agent
        [2,            1,                2,                3],   # Pure Code Gen
        [1,            2,                3,                2],   # Reasoning Transparency
        [3,            2,                3,                1],   # Balanced All-Round
        [1,            3,                2,                1],   # CPU Speed
    ])

    labels = np.array([
        ["Best",  "Best", "Good", "Poor"],
        ["Good",  "OK",   "Good", "Best"],
        ["OK",    "Good", "Best", "Good"],
        ["Best",  "Good", "Best", "OK"  ],
        ["OK",    "Best", "Good", "OK"  ],
    ])

    fig = plt.figure(figsize=(12, 6.5), facecolor=BG)
    ax  = fig.add_axes([0.18, 0.06, 0.78, 0.82])
    ax.set_facecolor(BG)

    n_rows, n_cols = scores.shape

    for r in range(n_rows):
        for c in range(n_cols):
            s     = scores[r, c]
            color = PALETTE[c]
            alpha = [0.07, 0.22, 0.52, 0.88][s]
            rect  = mpatches.FancyBboxPatch(
                (c + 0.05, n_rows - r - 1 + 0.05),
                0.90, 0.90,
                boxstyle="round,pad=0.05",
                facecolor=color, alpha=alpha,
                edgecolor=GRID, linewidth=0.7
            )
            ax.add_patch(rect)
            lbl_color = TEXT if s >= 2 else SUBTEXT
            fw = 'bold' if s == 3 else 'normal'
            ax.text(c + 0.5, n_rows - r - 0.5, labels[r, c],
                    ha='center', va='center',
                    color=lbl_color, fontsize=11, fontweight=fw)

    # Column headers
    col_headers = ["qwen3.6\n27b", "qwen3.6\n35b-a3b", "qwen3-coder\n30b", "deepseek-coder\n33b"]
    for c, (hdr, color) in enumerate(zip(col_headers, PALETTE)):
        ax.text(c + 0.5, n_rows + 0.30, hdr,
                ha='center', va='bottom',
                color=color, fontsize=11, fontweight='bold')

    # Row labels
    for r, uc in enumerate(use_cases):
        ax.text(-0.10, n_rows - r - 0.5, uc,
                ha='right', va='center',
                color=TEXT, fontsize=10.5)

    ax.set_xlim(-1.4, n_cols + 0.1)
    ax.set_ylim(-0.1, n_rows + 0.85)
    ax.axis('off')

    fig.text(0.18, 0.93, "Which Model for Which Use Case?",
             color=TEXT, fontsize=15, fontweight='bold', va='top')

    # Legend
    legend_items = [
        mpatches.Patch(facecolor='#8b949e', alpha=0.07, edgecolor=GRID, label='Poor'),
        mpatches.Patch(facecolor='#8b949e', alpha=0.22, edgecolor=GRID, label='OK'),
        mpatches.Patch(facecolor='#8b949e', alpha=0.52, edgecolor=GRID, label='Good'),
        mpatches.Patch(facecolor='#8b949e', alpha=0.88, edgecolor=GRID, label='Best'),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              bbox_to_anchor=(1.0, -0.04), ncol=4,
              framealpha=0, labelcolor=TEXT, fontsize=9,
              handlelength=1.2, borderpad=0.4, labelspacing=0.4)

    save(fig, "chart_recommendation_grid.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating charts...")
    chart_grouped_bars()
    chart_radar()
    chart_agent_spotlight()
    chart_recommendation_grid()
    print("\nDone. All charts saved to:", OUT_DIR)
