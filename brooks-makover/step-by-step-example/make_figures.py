import matplotlib.pyplot as plt
import networkx as nx

from collections import defaultdict
from matplotlib.lines import Line2D


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUT_DIR = "."

RIBBON_OUT = f"{OUT_DIR}/ribbon_graph.png"
CONVERGENCE_OUT = f"{OUT_DIR}/convergence.png"


# ============================================================
# 1. RIBBON GRAPH
# ============================================================

edges = [
    (0, 9),
    (0, 2),
    (0, 1),

    (1, 5),
    (1, 7),

    (2, 5),
    (2, 7),

    (3, 4),
    (3, 9),
    (3, 11),

    # short 2-cycle / double edge
    (4, 13),
    (4, 13),

    (5, 11),

    # short 2-cycle / double edge
    (6, 10),
    (6, 10),

    (6, 12),

    (7, 12),

    (8, 10),
    (8, 12),
    (8, 11),

    (9, 13),
]


G = nx.MultiGraph()

G.add_nodes_from(
    range(14)
)

G.add_edges_from(
    edges
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

pos = nx.circular_layout(
    G,
    scale=1.0
)


# Rotate the circular layout slightly for a cleaner orientation.
angle = 0.10

c = __import__("math").cos(angle)
s = __import__("math").sin(angle)

for node in pos:

    x, y = pos[node]

    pos[node] = (
        c * x - s * y,
        s * x + c * y
    )


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 8)
)


# Nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=720,
    node_color="#2b83ba",
    edgecolors="black",
    linewidths=1.2,
    ax=ax
)


# Labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=11,
    font_color="black",
    ax=ax
)


# ------------------------------------------------------------
# Group parallel edges
# ------------------------------------------------------------

pairs = defaultdict(list)

for u, v, key in G.edges(keys=True):

    pair = tuple(
        sorted((u, v))
    )

    pairs[pair].append(
        (u, v, key)
    )


# Highlighted short 2-cycles
highlight = {
    (4, 13): "#d62728",   # red
    (6, 10): "#2ca02c",   # green
}


# ------------------------------------------------------------
# Draw edges
# ------------------------------------------------------------

for pair, edge_list in pairs.items():

    # --------------------------
    # Ordinary single edge
    # --------------------------

    if len(edge_list) == 1:

        u, v, key = edge_list[0]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=1.25,
            edge_color="#222222",
            alpha=0.90,
            ax=ax
        )

    # --------------------------
    # Multiple / double edge
    # --------------------------

    else:

        # for two parallel edges:
        if len(edge_list) == 2:

            rads = [
                -0.17,
                0.17
            ]

        else:

            # generic fallback for >2 edges
            rads = [
                -0.20
                + 0.40 * i / max(1, len(edge_list) - 1)
                for i in range(len(edge_list))
            ]

        color = highlight.get(
            pair,
            "#222222"
        )

        width = (
            3.2
            if pair in highlight
            else 1.4
        )

        for (u, v, key), rad in zip(
            edge_list,
            rads
        ):

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=width,
                edge_color=color,
                alpha=0.96,
                connectionstyle=f"arc3,rad={rad}",
                ax=ax
            )


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

legend_handles = [

    Line2D(
        [0],
        [0],
        color="#d62728",
        lw=3.2,
        label=r"2-cycle $(4,13)$"
    ),

    Line2D(
        [0],
        [0],
        color="#2ca02c",
        lw=3.2,
        label=r"2-cycle $(6,10)$"
    )
]


ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    frameon=True,
    fontsize=10,
    ncol=2
)


# ------------------------------------------------------------
# Title + formatting
# ------------------------------------------------------------

ax.set_title(
    "Underlying trivalent multigraph: n=7, seed=12",
    fontsize=15,
    pad=18
)

ax.axis("off")

fig.tight_layout()

fig.savefig(
    RIBBON_OUT,
    dpi=260,
    bbox_inches="tight"
)

plt.close(fig)

print(
    f"Saved: {RIBBON_OUT}"
)


# ============================================================
# 2. CONVERGENCE PLOT
# ============================================================

h = [
    0.12,
    0.10,
    0.08
]

cusped = [
    0.2324693214,
    0.2322749276,
    0.2320742785
]

compact = [
    0.469456,
    0.468826,
    0.4680844126
]


fig, ax = plt.subplots(
    figsize=(7.2, 4.6)
)


ax.plot(
    h,
    cusped,
    marker="o",
    linewidth=1.8,
    markersize=6,
    label=r"cusped $\lambda_1(S)$"
)


ax.plot(
    h,
    compact,
    marker="o",
    linewidth=1.8,
    markersize=6,
    label=r"compact $\lambda_1(\bar S)$"
)


# finer meshes should visually lie to the right
ax.invert_xaxis()


ax.set_xlabel(
    "mesh width $h$ (finer to the right)"
)

ax.set_ylabel(
    r"$\lambda_1$"
)


ax.set_title(
    "Mesh refinement: two different spectral problems"
)


ax.grid(
    True,
    alpha=0.25
)


ax.legend(
    frameon=True
)


fig.tight_layout()


fig.savefig(
    CONVERGENCE_OUT,
    dpi=260,
    bbox_inches="tight"
)


plt.close(fig)

print(
    f"Saved: {CONVERGENCE_OUT}"
)


print()
print("Done.")