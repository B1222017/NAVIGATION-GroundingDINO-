"""Generate a floor-plan PNG showing all 13 zones from the topological map,
with each zone circled and labeled so the user can see which area is which."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

# ── Zone data from navigation_map.html ──────────────────────────────
zones = {
    0:  {"name": "Zone 0 — Sofa Area",        "photos": 6,  "range": "1432-1437", "key_objects": "sofa, plant, cabinet, door",            "role": "start"},
    1:  {"name": "Zone 1 — Fire Ext. Area",    "photos": 2,  "range": "1438-1439", "key_objects": "cabinet, fire ext., fridge",            "role": "both"},
    2:  {"name": "Zone 2 — Plant Area",        "photos": 2,  "range": "1440-1441", "key_objects": "sofa, plant, chair, door",              "role": "normal"},
    3:  {"name": "Zone 3 — Chair Area",        "photos": 2,  "range": "1442-1443", "key_objects": "sofa, door, chair, sign",               "role": "normal"},
    4:  {"name": "Zone 4 — Door Area",         "photos": 2,  "range": "1444-1445", "key_objects": "door, sign, fire ext.",                 "role": "fire"},
    5:  {"name": "Zone 5 — Refrigerator Area", "photos": 3,  "range": "1446-1448", "key_objects": "sofa, fridge, plant, chair",            "role": "fridge"},
    6:  {"name": "Zone 6 — (Open Area)",       "photos": 5,  "range": "1449-1453", "key_objects": "sofa, plant, chair, fridge, door",      "role": "fridge"},
    7:  {"name": "Zone 7 — Sign Area",         "photos": 7,  "range": "1454-1460", "key_objects": "fire ext., sign, chair, door",          "role": "fire"},
    8:  {"name": "Zone 8 — Chair Sofa Area",   "photos": 6,  "range": "1461-1466", "key_objects": "door, sign, chair sofa, cabinet",       "role": "normal"},
    9:  {"name": "Zone 9 — Cabinet Area",      "photos": 7,  "range": "1467-1473", "key_objects": "fridge(0.85!), fire ext., cabinet",     "role": "both"},
    10: {"name": "Zone 10 — Table Desk Area",  "photos": 10, "range": "1474-1483", "key_objects": "fire ext., cabinet, fridge, door",      "role": "both"},
    11: {"name": "Zone 11 — (Corridor)",       "photos": 2,  "range": "1484-1485", "key_objects": "door, sign, fire ext.",                 "role": "fire"},
    12: {"name": "Zone 12 — Printer Area",     "photos": 2,  "range": "1486-1487", "key_objects": "plant, chair, printer, sofa",           "role": "normal"},
}

# ── Edges (connections between zones) ───────────────────────────────
edges = [
    (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9),
    (9,10), (10,11), (11,12),
    (0,5), (0,6), (0,12),
    (2,5), (2,6), (2,12),
    (3,6), (3,7),
    (5,7), (5,12),
    (6,12), (7,12),
]

# ── Build graph and compute layout ──────────────────────────────────
G = nx.Graph()
for zid in zones:
    G.add_node(zid)
for a, b in edges:
    G.add_edge(a, b)

# Use a spring layout then refine for a floor-plan feel
# Seed for reproducibility
pos = nx.spring_layout(G, k=2.8, iterations=200, seed=42)

# Scale positions to a nice range
xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)
for zid in pos:
    x, y = pos[zid]
    pos[zid] = (2 + 14 * (x - x_min) / (x_max - x_min + 1e-9),
                2 + 10 * (y - y_min) / (y_max - y_min + 1e-9))

# ── Color scheme ────────────────────────────────────────────────────
role_colors = {
    "start":  ("#ff4444", "#ff444430", "#ffcccc"),   # red
    "fridge": ("#2277ee", "#2277ee30", "#cce0ff"),   # blue
    "fire":   ("#22aa44", "#22aa4430", "#ccf5cc"),   # green
    "both":   ("#ee8822", "#ee882230", "#ffe8cc"),   # orange
    "normal": ("#888888", "#88888820", "#e8e8e8"),   # grey
}

# ── Draw ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 16))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

# Draw edges first
for a, b in edges:
    xa, ya = pos[a]
    xb, yb = pos[b]
    # Navigation-path edges (sequential) are bolder
    is_sequential = abs(a - b) == 1
    lw = 2.5 if is_sequential else 0.8
    color = '#4488cc' if is_sequential else '#bbbbbb'
    style = '-' if is_sequential else '--'
    ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, linestyle=style,
            zorder=1, alpha=0.7 if is_sequential else 0.4)

# Draw zone circles and labels
for zid, info in zones.items():
    x, y = pos[zid]
    role = info["role"]
    edge_color, fill_color, text_bg = role_colors[role]

    # Circle size based on photo count
    radius = 0.45 + 0.04 * info["photos"]

    # Draw filled circle
    circle = plt.Circle((x, y), radius, facecolor=fill_color,
                         edgecolor=edge_color, linewidth=3, zorder=3)
    ax.add_patch(circle)

    # Zone number in center (large)
    ax.text(x, y + 0.12, str(zid), fontsize=22, fontweight='bold',
            ha='center', va='center', color=edge_color, zorder=5)

    # Zone name below circle
    short_name = info["name"].split("—")[1].strip() if "—" in info["name"] else f"Zone {zid}"
    ax.text(x, y - radius - 0.18, short_name, fontsize=10, fontweight='bold',
            ha='center', va='top', color='#333333', zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=text_bg, edgecolor=edge_color, alpha=0.9))

    # Photo range below name
    ax.text(x, y - radius - 0.55, f"IMG_{info['range']}\n({info['photos']} photos)",
            fontsize=7.5, ha='center', va='top', color='#666666', zorder=5)

    # Key objects inside circle (small)
    objs = info["key_objects"]
    if len(objs) > 30:
        objs = objs[:28] + "..."
    ax.text(x, y - 0.18, objs, fontsize=6, ha='center', va='center',
            color='#555555', zorder=5, style='italic')

    # Role label
    role_labels = {"start": "START", "fridge": "FRIDGE", "fire": "FIRE EXT.",
                   "both": "FRIDGE+FIRE", "normal": ""}
    rl = role_labels[role]
    if rl:
        ax.text(x, y + radius + 0.12, rl, fontsize=8, fontweight='bold',
                ha='center', va='bottom', color='white', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', facecolor=edge_color, alpha=0.85))

# ── Legend ───────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor='#ff444430', edgecolor='#ff4444', linewidth=2, label='START (You Are Here)'),
    mpatches.Patch(facecolor='#2277ee30', edgecolor='#2277ee', linewidth=2, label='Has Refrigerator'),
    mpatches.Patch(facecolor='#22aa4430', edgecolor='#22aa44', linewidth=2, label='Has Fire Extinguisher'),
    mpatches.Patch(facecolor='#ee882230', edgecolor='#ee8822', linewidth=2, label='Has Both Goals'),
    mpatches.Patch(facecolor='#88888820', edgecolor='#888888', linewidth=2, label='Other Zone'),
    plt.Line2D([0], [0], color='#4488cc', linewidth=2.5, label='Sequential path'),
    plt.Line2D([0], [0], color='#bbbbbb', linewidth=1, linestyle='--', label='Shared-object link'),
]
ax.legend(handles=legend_items, loc='upper left', fontsize=10,
          framealpha=0.95, facecolor='white', edgecolor='#cccccc',
          title='Zone Legend', title_fontsize=11)

# ── Title ───────────────────────────────────────────────────────────
ax.set_title('Floor Map — Zone Layout with Object Detections\n'
             'CSIE Department Office | 56 Photos → 13 Zones | GroundingDINO SwinT OGC',
             fontsize=16, fontweight='bold', color='#222222', pad=20)

# ── Axes ────────────────────────────────────────────────────────────
ax.set_xlim(0, 18)
ax.set_ylim(-0.5, 13.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
out_path = r'C:\Users\user\Downloads\Navigation-main\Navigation-main\floor_zones_map.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")
