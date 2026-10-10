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

# ---- slide 7: ResNet-20 (CIFAR) whole-network strip, paper Fig.3 style, box ~5.5 x 2.95 in ---------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
W, Hh = 5.5, 2.95
fig = plt.figure(figsize=(W, Hh)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis('off')
STAGE = [('#e8eef9', '#2a78d6', '16ch · 32×32'), ('#e6f4ee', '#1b8a5a', '32ch · 16×16'), ('#fbeae3', '#c4532a', '64ch · 8×8')]
bw, gap, y0, bh = 0.15, 0.058, 1.28, 0.62
x = 0.12
def box(x, face, edge, label=None, w=bw):
    ax.add_patch(FancyBboxPatch((x, y0), w, bh, boxstyle='round,pad=0,rounding_size=0.03', fc=face, ec=edge, lw=1))
    if label:
        ax.text(x + w / 2, y0 + bh / 2, label, rotation=90, ha='center', va='center', fontsize=10.5, color=INK)
ax.text(x, y0 + bh / 2, '입력', ha='left', va='center', fontsize=11, color=INK2); x += 0.5
ax.annotate('', (x - 0.03, y0 + bh / 2), (x - 0.15, y0 + bh / 2), arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
box(x, '#f3f1ee', INK2, '3×3', w=0.24); x += 0.24 + gap
starts = []
for si, (face, edge, lab) in enumerate(STAGE):
    sx = x
    for b in range(3):                       # n = 3 blocks -> 2 convs each
        bx = x
        for c in range(2):
            box(x, face, edge); x += bw + gap
        # shortcut arc over the block
        ex = x - gap
        dashed = (si > 0 and b == 0)
        ax.add_patch(FancyArrowPatch((bx - gap / 2, y0 + bh), (ex + gap / 2, y0 + bh), connectionstyle='arc3,rad=-0.55',
                                     arrowstyle='-|>', mutation_scale=7, lw=1.3, color=INK,
                                     linestyle=(0, (2, 1.5)) if dashed else '-'))
    starts.append((sx, x - gap, edge, lab))
    x += 0.04
box(x, '#f3f1ee', INK2, 'GAP', w=0.24); x += 0.24 + gap
box(x, '#f3f1ee', INK2, 'FC 10', w=0.24); x += 0.24
for sx, ex, edge, lab in starts:            # stage brackets + labels
    yb = y0 - 0.1
    ax.plot([sx, sx, ex, ex], [yb + 0.05, yb, yb, yb + 0.05], color=edge, lw=1.2)
    ax.text((sx + ex) / 2, yb - 0.08, lab, ha='center', va='top', fontsize=11, color=edge, fontweight='bold')
ax.text(0.12, 0.42, '첫 3×3 층 + 블록 9개(conv 2층씩) 18층 + FC = 20층', fontsize=11.5, color=INK, va='center')
ax.text(0.12, 0.14, 'n = 9로 늘리면 56층 (0.85M) · plain 망은 화살표만 제거', fontsize=11.5, color=INK2, va='center')
ax.plot([3.3, 3.62], [2.62, 2.62], color=INK, lw=1.3); ax.text(3.67, 2.62, '항등 shortcut', fontsize=11, va='center', color=INK)
ax.plot([3.3, 3.62], [2.36, 2.36], color=INK, lw=1.3, linestyle=(0, (2, 1.5))); ax.text(3.67, 2.36, '크기 변경 (zero-pad)', fontsize=11, va='center', color=INK)
ax.text(0.12, 2.62, 'ResNet-20 (n = 3, 0.27M)', fontsize=14, fontweight='bold', va='center', color=INK)
fig.savefig(os.path.join(a.out, 'slide7_resnet20.png'), dpi=250)
plt.close(fig)
print('saved slide7')

# ---- appendix: per-layer gradient norm at initialization, box ~11.7 x 3.3 in ----------------------
sel = {r['kind']: r for r in runs if r['depth'] == 56}
fig, ax = plt.subplots(figsize=(11.6, 3.3))
for kind, name in [('plain', 'plain-56'), ('residual', 'ResNet-56')]:
    g = sel[kind]['grads']; g = g[g.epoch == 0].sort_values('layer_idx')
    ax.plot(g['layer_idx'] + 1, g['grad_norm'], color=C_KIND[kind], lw=2.4, label=name)
ax.set_yscale('log')
ax.set_title('초기화 직후 (학습 전, 미니배치 10개 평균)', loc='left', fontsize=14, fontweight='bold', color=INK)
ax.set_xlabel('conv 층 위치 (입력 → 출력)', fontsize=13); ax.set_ylabel('기울기 크기 (로그)', fontsize=13)
ax.grid(axis='y', color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=13, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(a.out, 'appendix_grad_init.png'), dpi=220)
plt.close(fig)
print('saved appendix')
