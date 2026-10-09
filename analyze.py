"""
[study] Analysis + plots from the run logs (log.csv / grads.csv / config.json per run).

    python analyze.py --runs /content/drive/MyDrive/resnet_cifar/runs --out figures

Outputs (in --out):
  fig6_ours.png / .pdf     paper Fig. 6 style: left plain, right ResNet; dashed = train error, solid = test error
  curves.html              same data, interactive (hover) - for checking, not for slides
  grad_by_layer.png        weight-gradient norm of every conv layer, plain-56 vs ResNet-56 (init / early / final)
  results_table.csv / .md  slide 8 table (our final & best test error, akamaster README, paper)
"""
import argparse, glob, json, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# validated categorical slots (dataviz reference palette, light surface)
C_DEPTH = {20: '#2a78d6', 32: '#1baf7a', 44: '#eda100', 56: '#eb6834', 110: '#4a3aa7'}
C_KIND = {'residual': '#2a78d6', 'plain': '#eb6834'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
README_ERR = {20: 8.27, 32: 7.37, 44: 6.90, 56: 6.61, 110: 6.32}
PAPER_ERR = {20: 8.75, 32: 7.51, 44: 7.17, 56: 6.97, 110: '6.43 (best of 5)'}

plt.rcParams.update({'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def load_runs(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.exists(os.path.join(d, 'log.csv')):
            continue
        cfg = json.load(open(os.path.join(d, 'config.json')))
        # after a disconnect an epoch can be logged twice (logged, then killed before its checkpoint) -> keep last
        log = pd.read_csv(os.path.join(d, 'log.csv')).drop_duplicates('epoch', keep='last').sort_values('epoch')
        grads = pd.read_csv(os.path.join(d, 'grads.csv')).drop_duplicates(['epoch', 'layer_idx'], keep='last')
        depth = int(cfg['arch'].replace('resnet', ''))
        runs.append(dict(name=cfg['run'], depth=depth, kind='plain' if cfg['plain'] else 'residual',
                         seed=cfg['seed'], cfg=cfg, log=log, grads=grads,
                         done=os.path.exists(os.path.join(d, 'DONE'))))
    return runs


def fig6(runs, out, ymax=20):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kind, title in [(axes[0], 'plain', 'Plain'), (axes[1], 'residual', 'ResNet')]:
        sel = sorted([r for r in runs if r['kind'] == kind], key=lambda r: r['depth'])
        # end-of-line label positions, pushed apart so they never overlap
        ends = sorted(((min(r['log'][r['log'].epoch > 0]['test_err'].iloc[-1], ymax - 0.5), r['depth'])
                       for r in sel if (r['log'].epoch > 0).any()))
        label_y, last = {}, -1e9
        for y, dpt in ends:
            y = max(y, last + 0.9)
            label_y[dpt], last = y, y
        for r in sel:
            lg = r['log'][r['log'].epoch > 0]
            x = lg['iter'] / 1e4
            c = C_DEPTH.get(r['depth'], INK2)
            ax.plot(x, lg['train_err'], color=c, lw=1.2, ls='--', alpha=0.9)
            ax.plot(x, lg['test_err'], color=c, lw=2.2)
            # direct label at the right end of the test curve
            ax.annotate(f"{title.lower()}-{r['depth']}", (x.iloc[-1], label_y[r['depth']]),
                        xytext=(6, 0), textcoords='offset points', va='center', fontsize=10, color=INK)
        for m in sel[:1]:
            for it in m['cfg']['milestones_iter']:
                ax.axvline(it / 1e4, color=GRID, lw=1, zorder=0)
        ax.set_title(title, loc='left', color=INK, fontsize=12, fontweight='bold')
        ax.set_xlabel('iter. (1e4)')
        ax.set_ylim(0, ymax)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.margins(x=0.02)
    axes[0].set_ylabel('error (%)')
    from matplotlib.lines import Line2D
    axes[1].legend([Line2D([], [], color=INK2, ls='--', lw=1.2), Line2D([], [], color=INK2, lw=2.2)],
                   ['train error', 'test error'], frameon=False, loc='upper right')
    fig.suptitle('CIFAR-10 — our runs (dashed: training error, bold: test error)', x=0.01, ha='left',
                 color=INK, fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig6_ours.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def curves_html(runs, out):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(1, 2, shared_yaxes=True, subplot_titles=('Plain', 'ResNet'))
    for col, kind in [(1, 'plain'), (2, 'residual')]:
        for r in sorted([r for r in runs if r['kind'] == kind], key=lambda r: r['depth']):
            lg = r['log'][r['log'].epoch > 0]
            c = C_DEPTH.get(r['depth'], INK2)
            for key, dash, w in [('train_err', 'dash', 1.2), ('test_err', 'solid', 2.2)]:
                fig.add_trace(go.Scatter(x=lg['iter'], y=lg[key], name=f"{r['name']} {key}",
                                         line=dict(color=c, dash=dash, width=w),
                                         hovertemplate='iter %{x}<br>%{y:.2f}%<extra>' + r['name'] + '</extra>'),
                              1, col)
    fig.update_yaxes(range=[0, 20], title_text='error (%)', col=1)
    fig.update_xaxes(title_text='iteration')
    fig.update_layout(height=450, width=1100, hovermode='x unified', template='simple_white')
    fig.write_html(os.path.join(out, 'curves.html'), include_plotlyjs='cdn')


def grad_by_layer(runs, out, depth=56):
    sel = {r['kind']: r for r in runs if r['depth'] == depth}
    if len(sel) < 2:
        print(f'grad_by_layer: need plain-{depth} and resnet-{depth}, skip')
        return
    last = min(int(sel['plain']['grads'].epoch.max()), int(sel['residual']['grads'].epoch.max()))
    epochs = [e for e in [0, 1, last] if e <= last]
    epochs = sorted(set(epochs))
    titles = {0: 'at initialization', 1: 'epoch 1'}
    fig, axes = plt.subplots(1, len(epochs), figsize=(4.2 * len(epochs), 3.8), sharey=True)
    axes = axes if len(epochs) > 1 else [axes]
    for ax, e in zip(axes, epochs):
        for kind in ['plain', 'residual']:
            g = sel[kind]['grads']
            g = g[g.epoch == e].sort_values('layer_idx')
            ax.plot(g['layer_idx'] + 1, g['grad_norm'], color=C_KIND[kind], lw=2,
                    label=f"{'plain' if kind == 'plain' else 'resnet'}-{depth}")
        ax.set_yscale('log')
        ax.set_title(titles.get(e, f'epoch {e} (final)'), loc='left', color=INK, fontsize=11)
        ax.set_xlabel('conv layer (input → output)')
        ax.grid(axis='y', color=GRID, lw=0.8, which='major')
    axes[0].set_ylabel('‖∂L/∂W‖ (log scale)')
    axes[-1].legend(frameon=False)
    fig.suptitle(f'Weight-gradient norm per conv layer — plain-{depth} vs ResNet-{depth}', x=0.01, ha='left',
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'grad_by_layer.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def results_table(runs, out):
    rows = []
    for r in sorted(runs, key=lambda r: (r['kind'] != 'plain', r['depth'])):
        lg = r['log'][r['log'].epoch > 0]
        model = f"{'plain' if r['kind'] == 'plain' else 'ResNet'}-{r['depth']}"
        rows.append({
            'model': model, 'layers': r['depth'], 'params': f"{r['cfg']['params'] / 1e6:.2f}M",
            'iters': int(lg['iter'].iloc[-1]) if len(lg) else 0, 'finished': r['done'],
            'final train err (%)': round(lg['train_err'].iloc[-1], 2) if len(lg) else None,
            'final test err (%)': round(lg['test_err'].iloc[-1], 2) if len(lg) else None,
            'best test err (%)': round(lg['test_err'].min(), 2) if len(lg) else None,
            'akamaster README (%)': README_ERR.get(r['depth']) if r['kind'] == 'residual' else '수치 없음',
            'paper Table 6 (%)': PAPER_ERR.get(r['depth']) if r['kind'] == 'residual' else '수치 없음(그래프만)',
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'results_table.csv'), index=False)
    with open(os.path.join(out, 'results_table.md'), 'w') as f:
        f.write(df.to_markdown(index=False) if hasattr(df, 'to_markdown') else df.to_string(index=False))
    print(df.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='figures')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs)
    print(f'{len(runs)} runs:', ', '.join(f"{r['name']}{'' if r['done'] else ' (running)'}" for r in runs))
    fig6(runs, a.out)
    curves_html(runs, a.out)
    grad_by_layer(runs, a.out)
    results_table(runs, a.out)
    print('saved to', a.out)
