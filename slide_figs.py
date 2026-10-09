"""
[study] Slide-sized figures for the team deck (sizes match the deck's figure boxes, Korean labels).

    python slide_figs.py --runs results/runs --out results/figures/slides --font <Pretendard dir>
"""
import argparse, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from analyze import load_runs, C_DEPTH, C_KIND

INK, INK2, GRID = '#000000', '#615d59', '#e6e3df'

p = argparse.ArgumentParser()
p.add_argument('--runs', default='results/runs')
p.add_argument('--out', default='results/figures/slides')
p.add_argument('--font', default='')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
for f in glob.glob(os.path.join(a.font, '**', 'Pretendard-*.otf'), recursive=True):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Pretendard', 'font.size': 13, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.unicode_minus': False})
runs = load_runs(a.runs)

# ---- slide 9: learning curves, box 6.90 x 4.48 in --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.7, 4.3), sharey=True)
for ax, kind, title in [(axes[0], 'plain', 'Plain'), (axes[1], 'residual', 'ResNet')]:
    sel = sorted([r for r in runs if r['kind'] == kind], key=lambda r: r['depth'])
    for r in sel:
        lg = r['log'][r['log'].epoch > 0]
        c = C_DEPTH[r['depth']]
        ax.plot(lg['epoch'], lg['train_err'], color=c, lw=1.2, ls='--')
        ax.plot(lg['epoch'], lg['test_err'], color=c, lw=2.2)
    for e in (82, 123):
        ax.axvline(e, color=GRID, lw=1, zorder=0)
    # end labels on the test curves, kept apart
    ends = sorted((min(r['log']['test_err'].iloc[-1], 19.3), r['depth']) for r in sel)
    last = -1e9
    for y, d in ends:
        y = max(y, last + 1.3); last = y
        ax.text(166, y, f"{d}층", va='center', fontsize=12, color=C_DEPTH[d], fontweight='bold')
    ax.set_title(title, loc='left', fontsize=15, fontweight='bold', color=INK)
    ax.set_xlim(0, 164); ax.set_ylim(0, 20); ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_xlabel('에폭'); ax.grid(axis='y', color=GRID, lw=0.8)
axes[0].set_ylabel('오차 (%)')
fig.legend([Line2D([], [], color=INK2, ls='--', lw=1.2), Line2D([], [], color=INK2, lw=2.2)],
           ['학습 오차', '테스트 오차'], frameon=False, loc='upper right', bbox_to_anchor=(0.99, 1.0),
           ncol=2, fontsize=12, handlelength=1.8, columnspacing=1.2)
fig.tight_layout(w_pad=2.2, rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(a.out, 'slide9_curves.png'), dpi=220)
plt.close(fig)

# ---- slide 11: per-layer gradient norm, box 12.00 x 2.17 in -----------------------------------
sel = {r['kind']: r for r in runs if r['depth'] == 56}
fig, axes = plt.subplots(1, 2, figsize=(11.7, 2.05), sharey=True)
for ax, e, title in [(axes[0], 40, '학습 중 · 40 에폭 (lr 0.1)'), (axes[1], 164, '학습 종료 · 164 에폭')]:
    for kind, name in [('plain', 'plain-56'), ('residual', 'ResNet-56')]:
        g = sel[kind]['grads']; g = g[g.epoch == e].sort_values('layer_idx')
        ax.plot(g['layer_idx'] + 1, g['grad_norm'], color=C_KIND[kind], lw=2, label=name)
    ax.set_yscale('log'); ax.set_ylim(1.5e-3, 2)
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', color=INK, pad=4)
    ax.set_xlabel('conv 층 위치 (입력 → 출력)', fontsize=12, labelpad=1)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', color=GRID, lw=0.8)
axes[0].set_ylabel('기울기 크기', fontsize=12)
axes[0].legend(frameon=False, loc='lower right', fontsize=12, ncol=2, borderaxespad=0.1)
fig.tight_layout(w_pad=3)
fig.savefig(os.path.join(a.out, 'slide11_grad.png'), dpi=220)
plt.close(fig)
print('saved', a.out)
