"""
Visualization script (converted from visualizations.ipynb)

Changes from the original notebook:
  1. Font family switched to sans-serif everywhere.
  2. No multi-panel/subplot figures. Every figure that used to be a panel
     inside a subplot grid (e.g. axes[0], axes[1], gs[0,0], etc.) is now
     its own single-axes figure, saved as its own file.

Everything else (data loading, computations, colors, labels, titles,
save paths / naming convention as closely as possible) is unchanged.
"""

import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ROOT = os.path.abspath(os.path.join('..', '..'))
RESULTS = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

print(f'Results directory: {RESULTS}')
print(f'Figures will be saved to: {FIG_DIR}')


def load_json_results(task_dir):
    """Load all JSON result files from a task directory into a DataFrame."""
    records = []
    path = os.path.join(RESULTS, task_dir)
    if not os.path.exists(path):
        return pd.DataFrame()
    for model_dir in os.listdir(path):
        model_path = os.path.join(path, model_dir)
        if not os.path.isdir(model_path) or model_dir.startswith('hf_'):
            continue
        for f in glob.glob(os.path.join(model_path, '*.json')):
            with open(f, encoding='utf-8') as fp:
                data = json.load(fp)
                records.append({
                    'task': data.get('task', task_dir),
                    'model': data.get('model', model_dir),
                    'source': data['source'],
                    'target': data['target'],
                    'type': data['type'],
                    'macro_f1': data['macro_f1'],
                    'accuracy': data.get('accuracy', 0),
                    'k_shot': data.get('k_shot', None),
                })
    return pd.DataFrame(records)


TASK_DIRS = {
    'T1_Nastaliq_SA':       'Nastaliq Sentiment',
    'T1_Nastaliq_HS':       'Nastaliq Hate Speech',
    'T1_Nastaliq_FND':      'Nastaliq Fake News',
    'T1_Nastaliq_QA':       'Nastaliq QA',
    'T5_Roman_CyberAbuse':  'Roman Cyber Abuse',
    'T6_Roman_ProductReviews': 'Roman Product Reviews',
    'T7_Roman_Sentiments':  'Roman Sentiments',
    'T8_Roman_TwitterOpinions': 'Roman Twitter Opinions',
}

all_results = {}
for task_dir, task_name in TASK_DIRS.items():
    df = load_json_results(task_dir)
    if len(df) > 0:
        all_results[task_name] = df
        print(f'{task_name}: {len(df)} results loaded')
    else:
        print(f'{task_name}: NO RESULTS YET')

print(f'\nTotal tasks with results: {len(all_results)}')

tasks_with_results = [name for name in all_results.keys()]
n_tasks = len(tasks_with_results)


# ============================================================
# Figure 1: Cross-Domain Performance Heatmaps (one file per task)
# ============================================================
def plot_heatmap(df, task_name, model_filter='XLM-R_Base'):
    subset = df[df['model'] == model_filter].copy()
    if len(subset) == 0:
        return

    domains = sorted(subset['source'].unique())
    matrix = pd.DataFrame(index=domains, columns=domains, dtype=float)
    for _, row in subset.iterrows():
        matrix.loc[row['source'], row['target']] = row['macro_f1']

    short_names = [d.split('_')[0] + '_' + '_'.join(d.split('_')[1:3]) for d in domains]

    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = LinearSegmentedColormap.from_list('custom', ['#fee0d2', '#fc9272', '#de2d26'])
    sns.heatmap(matrix.astype(float), annot=True, fmt='.3f', cmap=cmap,
                vmin=0.3, vmax=1.0, ax=ax, cbar_kws={'shrink': 0.8},
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.5, linecolor='white')
    ax.set_title(f'Cross-Domain F1 Performance Matrix: {task_name} (XLM-R Base)')
    ax.set_xlabel('Target Domain')
    ax.set_ylabel('Source Domain')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    safe_name = task_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(FIG_DIR, f'fig1_heatmap_{safe_name}.pdf'))
    plt.savefig(os.path.join(FIG_DIR, f'fig1_heatmap_{safe_name}.png'))
    plt.show()
    print(f'Saved: fig1_heatmap_{safe_name}.pdf/png')


if n_tasks > 0:
    for task_name in tasks_with_results:
        plot_heatmap(all_results[task_name], task_name)
else:
    print('No results available yet for heatmaps.')


# ============================================================
# Figure 2: Domain Robustness Radar Charts (one file per task)
# ============================================================
def plot_radar(df, task_name):
    models = ['XLM-R_Base', 'mBERT']
    colors = ['#2171b5', '#cb181d']

    domains = sorted(df['target'].unique())
    n_domains = len(domains)
    angles = np.linspace(0, 2 * np.pi, n_domains, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for model, color in zip(models, colors):
        subset = df[(df['model'] == model) & (df['type'] == 'CROSS-DOMAIN')]
        if len(subset) == 0:
            continue
        avg_per_target = subset.groupby('target')['macro_f1'].mean()
        values = [avg_per_target.get(d, 0) for d in domains]
        values += values[:1]
        ax.plot(angles, values, 'o-', color=color, linewidth=2, label=model, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)

    short_labels = [d.split('_')[0] for d in domains]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f'Cross-Domain Transfer Performance: {task_name}', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    safe_name = task_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(FIG_DIR, f'fig2_radar_{safe_name}.pdf'))
    plt.savefig(os.path.join(FIG_DIR, f'fig2_radar_{safe_name}.png'))
    plt.show()
    print(f'Saved: fig2_radar_{safe_name}.pdf/png')


if n_tasks > 0:
    for task_name in tasks_with_results:
        plot_radar(all_results[task_name], task_name)


# ============================================================
# Figure 3: In-Domain vs Cross-Domain Performance Gap
# (split into two single-plot figures)
# ============================================================
gap_df = pd.DataFrame()
if n_tasks > 0:
    gap_records = []
    for task_name, df in all_results.items():
        for model in df['model'].unique():
            if model not in ['XLM-R_Base', 'mBERT']:
                continue
            model_df = df[df['model'] == model]
            in_domain = model_df[model_df['type'] == 'IN-DOMAIN']
            cross_domain = model_df[model_df['type'] == 'CROSS-DOMAIN']

            for _, cd_row in cross_domain.iterrows():
                id_row = in_domain[in_domain['source'] == cd_row['source']]
                if len(id_row) > 0:
                    drop = id_row.iloc[0]['macro_f1'] - cd_row['macro_f1']
                    gap_records.append({
                        'task': task_name,
                        'model': model,
                        'source': cd_row['source'],
                        'target': cd_row['target'],
                        'drop': drop,
                        'script': 'Nastaliq' if 'Nastaliq' in task_name else 'Roman',
                    })

    gap_df = pd.DataFrame(gap_records)

    if len(gap_df) > 0:
        # --- Figure 3a: by task ---
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.violinplot(data=gap_df, x='task', y='drop', hue='model',
                       split=True, inner='quart', palette=['#2171b5', '#cb181d'],
                       ax=ax, cut=0)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Performance Drop Distribution by Task')
        ax.set_xlabel('')
        ax.set_ylabel('F1 Drop (In-Domain - Cross-Domain)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig3a_domain_drop_violin_by_task.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig3a_domain_drop_violin_by_task.png'))
        plt.show()
        print('Saved: fig3a_domain_drop_violin_by_task.pdf/png')

        # --- Figure 3b: by script type ---
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(data=gap_df, x='script', y='drop', hue='model',
                    palette=['#2171b5', '#cb181d'], ax=ax, width=0.6)
        sns.stripplot(data=gap_df, x='script', y='drop', hue='model',
                      palette=['#2171b5', '#cb181d'], ax=ax,
                      dodge=True, alpha=0.3, size=3, legend=False)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Performance Drop: Nastaliq vs Roman Script')
        ax.set_xlabel('Script Type')
        ax.set_ylabel('F1 Drop (In-Domain - Cross-Domain)')
        ax.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig3b_domain_drop_by_script.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig3b_domain_drop_by_script.png'))
        plt.show()
        print('Saved: fig3b_domain_drop_by_script.pdf/png')


# ============================================================
# Figure 4: Table 4 Summary
# (split into two single-plot figures)
# ============================================================
table4_files = glob.glob(os.path.join(RESULTS, 'Table4_*.csv'))
print(f'Found {len(table4_files)} Table 4 CSV files.')

table4_df = pd.DataFrame()
if table4_files:
    table4_all = []
    for f in sorted(table4_files):
        df = pd.read_csv(f)
        table4_all.append(df)

    table4_df = pd.concat(table4_all, ignore_index=True)
    print(table4_df.to_string(index=False))

    plot_df = table4_df.copy()
    task_short = plot_df['task'].str.replace('T1_Nastaliq_', 'N:').str.replace('T5_Roman_', 'R:').str.replace('T6_Roman_', 'R:').str.replace('T7_Roman_', 'R:').str.replace('T8_Roman_', 'R:')
    plot_df['task_short'] = task_short

    x = np.arange(len(plot_df))
    width = 0.35

    # --- Figure 4a: In-Domain vs Cross-Domain F1 ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, plot_df['avg_SS'], width, label='In-Domain (avg_SS)',
           color='#2171b5', edgecolor='white')
    ax.bar(x + width/2, plot_df['avg_ST'], width, label='Cross-Domain (avg_ST)',
           color='#fc9272', edgecolor='white')
    ax.set_xlabel('')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('In-Domain vs Cross-Domain Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['task_short']}\n({row['model']})" for _, row in plot_df.iterrows()],
                        rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4a_indomain_vs_crossdomain.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig4a_indomain_vs_crossdomain.png'))
    plt.show()
    print('Saved: fig4a_indomain_vs_crossdomain.pdf/png')

    # --- Figure 4b: Source Drop vs Target Drop ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, plot_df['avg_SD'], width, label='Avg Source Drop',
           color='#6baed6', edgecolor='white')
    ax.bar(x + width/2, plot_df['avg_TD'], width, label='Avg Target Drop',
           color='#fcbba1', edgecolor='white')

    ax.scatter(x - width/2, plot_df['WSD'], marker='v', color='#08519c', s=50, zorder=5, label='Worst SD')
    ax.scatter(x + width/2, plot_df['WTD'], marker='v', color='#a50f15', s=50, zorder=5, label='Worst TD')

    ax.set_xlabel('')
    ax.set_ylabel('F1 Drop')
    ax.set_title('Domain Shift Impact (Drops)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['task_short']}\n({row['model']})" for _, row in plot_df.iterrows()],
                        rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4b_domain_shift_drops.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig4b_domain_shift_drops.png'))
    plt.show()
    print('Saved: fig4b_domain_shift_drops.pdf/png')


# ============================================================
# Figure 5: Table 5 — Fine-Tuned vs Few-Shot LLM Comparison
# (one file per task instead of one subplot per task)
# ============================================================
table5_files = glob.glob(os.path.join(RESULTS, 'Table5_*.csv'))
print(f'Found {len(table5_files)} Table 5 CSV files.')

if table5_files:
    table5_all = []
    for f in sorted(table5_files):
        df = pd.read_csv(f)
        task_name = os.path.basename(f).replace('Table5_', '').replace('.csv', '')
        df['task'] = task_name
        table5_all.append(df)

    table5_df = pd.concat(table5_all, ignore_index=True)
    print(table5_df.to_string(index=False))

    tasks = table5_df['task'].unique()

    for task in sorted(tasks):
        task_df = table5_df[table5_df['task'] == task].copy()

        colors = []
        for _, row in task_df.iterrows():
            if row['type'] == 'fine-tuned':
                colors.append('#2171b5')
            else:
                colors.append('#e6550d')

        y_pos = np.arange(len(task_df))

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.barh(y_pos, task_df['avg_ST'], color=colors, edgecolor='white', height=0.7)
        ax.scatter(task_df['avg_SS'], y_pos, marker='D', color='black', s=40, zorder=5, label='In-Domain (SS)')

        for i, (ss, st) in enumerate(zip(task_df['avg_SS'], task_df['avg_ST'])):
            ax.plot([st, ss], [i, i], 'k--', alpha=0.3, linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(task_df['model'], fontsize=9)
        ax.set_xlabel('Macro F1')
        ax.set_title(f'Fine-Tuned vs Few-Shot LLM: {task.replace("_", " ")}')
        ax.set_xlim(0, 1.0)
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)

        ft_patch = mpatches.Patch(color='#2171b5', label='Fine-tuned')
        fs_patch = mpatches.Patch(color='#e6550d', label='Few-shot')
        ss_marker = plt.Line2D([0], [0], marker='D', color='black', linestyle='', markersize=5, label='In-Domain')
        ax.legend(handles=[ft_patch, fs_patch, ss_marker], loc='lower right', fontsize=8)

        plt.tight_layout()
        safe_task = task.replace(' ', '_').lower()
        plt.savefig(os.path.join(FIG_DIR, f'fig5_finetuned_vs_fewshot_{safe_task}.pdf'))
        plt.savefig(os.path.join(FIG_DIR, f'fig5_finetuned_vs_fewshot_{safe_task}.png'))
        plt.show()
        print(f'Saved: fig5_finetuned_vs_fewshot_{safe_task}.pdf/png')


# ============================================================
# Figure 6: Script Comparison — Nastaliq vs Roman Urdu
# (split into three single-plot figures, one per metric)
# ============================================================
if table4_files:
    nastaliq_tasks = table4_df[table4_df['task'].str.contains('Nastaliq')].copy()
    roman_tasks = table4_df[table4_df['task'].str.contains('Roman')].copy()

    if len(nastaliq_tasks) > 0 and len(roman_tasks) > 0:
        metrics = ['avg_SS', 'avg_ST', 'avg_SD']
        titles = ['In-Domain Performance', 'Cross-Domain Performance', 'Avg Source Drop']
        file_tags = ['indomain', 'crossdomain', 'sourcedrop']

        for metric, title, tag in zip(metrics, titles, file_tags):
            nastaliq_vals = nastaliq_tasks.groupby('model')[metric].mean()
            roman_vals = roman_tasks.groupby('model')[metric].mean()

            models = sorted(set(nastaliq_vals.index) & set(roman_vals.index))
            x = np.arange(len(models))
            width = 0.35

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(x - width/2, [nastaliq_vals.get(m, 0) for m in models], width,
                   label='Nastaliq', color='#4a1486', edgecolor='white')
            ax.bar(x + width/2, [roman_vals.get(m, 0) for m in models], width,
                   label='Roman', color='#41ab5d', edgecolor='white')

            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_title(f'Script Comparison: {title}')
            ax.set_ylabel('Score')
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f'fig6_script_comparison_{tag}.pdf'))
            plt.savefig(os.path.join(FIG_DIR, f'fig6_script_comparison_{tag}.png'))
            plt.show()
            print(f'Saved: fig6_script_comparison_{tag}.pdf/png')
    else:
        print('Need both Nastaliq and Roman results for script comparison.')


# ============================================================
# Figure 7: Model Robustness Scatter Plot (already a single plot)
# ============================================================
if n_tasks > 0:
    scatter_records = []
    for task_name, df in all_results.items():
        for model in df['model'].unique():
            if model not in ['XLM-R_Base', 'mBERT']:
                continue
            model_df = df[df['model'] == model]
            in_domain = model_df[model_df['type'] == 'IN-DOMAIN']['macro_f1'].mean()
            cross_domain = model_df[model_df['type'] == 'CROSS-DOMAIN']['macro_f1'].mean()
            scatter_records.append({
                'task': task_name,
                'model': model,
                'in_domain': in_domain,
                'cross_domain': cross_domain,
                'script': 'Nastaliq' if 'Nastaliq' in task_name else 'Roman',
            })

    scatter_df = pd.DataFrame(scatter_records)

    if len(scatter_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 7))

        markers = {'XLM-R_Base': 'o', 'mBERT': 's'}
        colors = {'Nastaliq': '#4a1486', 'Roman': '#41ab5d'}

        for _, row in scatter_df.iterrows():
            ax.scatter(row['in_domain'], row['cross_domain'],
                      marker=markers[row['model']], color=colors[row['script']],
                      s=100, edgecolors='white', linewidth=0.5, zorder=5)

        ax.plot([0.3, 1.0], [0.3, 1.0], 'k--', alpha=0.3, label='Perfect Robustness')

        for _, row in scatter_df.iterrows():
            short_task = row['task'].split()[-1][:8]
            ax.annotate(short_task, (row['in_domain'], row['cross_domain']),
                       textcoords='offset points', xytext=(5, 5), fontsize=7, alpha=0.7)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='XLM-R Base'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='mBERT'),
            mpatches.Patch(color='#4a1486', label='Nastaliq'),
            mpatches.Patch(color='#41ab5d', label='Roman'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        ax.set_xlabel('In-Domain F1 (avg_SS)')
        ax.set_ylabel('Cross-Domain F1 (avg_ST)')
        ax.set_title('Model Robustness: In-Domain vs Cross-Domain')
        ax.set_xlim(0.3, 1.02)
        ax.set_ylim(0.3, 1.02)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig7_robustness_scatter.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig7_robustness_scatter.png'))
        plt.show()
        print('Saved: fig7_robustness_scatter.pdf/png')


# ============================================================
# Figure 8: Domain Difficulty Ranking (already a single plot)
# ============================================================
if n_tasks > 0:
    difficulty_records = []
    for task_name, df in all_results.items():
        cross_df = df[(df['type'] == 'CROSS-DOMAIN') & (df['model'] == 'XLM-R_Base')]
        for target in cross_df['target'].unique():
            avg_f1 = cross_df[cross_df['target'] == target]['macro_f1'].mean()
            difficulty_records.append({
                'task': task_name,
                'domain': target,
                'avg_cross_f1': avg_f1,
            })

    diff_df = pd.DataFrame(difficulty_records)

    if len(diff_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))

        diff_df_sorted = diff_df.sort_values('avg_cross_f1', ascending=True)

        colors_map = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(diff_df_sorted)))

        y_pos = np.arange(len(diff_df_sorted))
        ax.barh(y_pos, diff_df_sorted['avg_cross_f1'], color=colors_map, edgecolor='white')

        labels = [f"{row['task']}: {row['domain'][:25]}" for _, row in diff_df_sorted.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Avg Cross-Domain F1 (lower = harder target)')
        ax.set_title('Domain Difficulty Ranking (as Transfer Target)')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='F1 = 0.5')
        ax.legend()
        ax.set_xlim(0, 1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig8_domain_difficulty.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig8_domain_difficulty.png'))
        plt.show()
        print('Saved: fig8_domain_difficulty.pdf/png')


# ============================================================
# Figure 9: Training Size vs Performance (already a single plot)
# ============================================================
size_df = pd.DataFrame()
if n_tasks > 0:
    size_records = []
    for task_name, df in all_results.items():
        in_domain = df[df['type'] == 'IN-DOMAIN']
        for _, row in in_domain.iterrows():
            size_records.append({
                'task': task_name,
                'model': row['model'],
                'domain': row['source'],
                'macro_f1': row['macro_f1'],
                'script': 'Nastaliq' if 'Nastaliq' in task_name else 'Roman',
            })

    size_df = pd.DataFrame(size_records)

    if len(size_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        model_palette = {
            'XLM-R_Base': '#2171b5',
            'mBERT': '#cb181d',
            'Qwen2.5-7B': '#e6550d',
            'Llama3.1-8B': '#31a354',
            'Mistral-7B': '#0170b8',
        }
        available_models = [m for m in size_df['model'].unique() if m in model_palette]
        # fall back to a generated color for any model not in the fixed palette
        extra_models = [m for m in size_df['model'].unique() if m not in model_palette]
        if extra_models:
            extra_colors = plt.cm.tab10(np.linspace(0, 1, len(extra_models)))
            for m, c in zip(extra_models, extra_colors):
                model_palette[m] = c

        sns.boxplot(data=size_df, x='task', y='macro_f1', hue='model',
                   palette=model_palette,
                   ax=ax, width=0.6)

        ax.set_xlabel('')
        ax.set_ylabel('In-Domain Macro F1')
        ax.set_title('In-Domain Performance Distribution Across Tasks')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0.4, 1.02)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.legend(title='Model')

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig9_indomain_distribution.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig9_indomain_distribution.png'))
        plt.show()
        print('Saved: fig9_indomain_distribution.pdf/png')


# ============================================================
# Figure 10: Comprehensive Task-Level Summary Dashboard
# (split into six single-plot figures, one per former panel)
# ============================================================
if table4_files and n_tasks > 0:
    # --- Panel A: Overall avg_SS/avg_ST by model across all tasks ---
    fig, ax1 = plt.subplots(figsize=(6, 5))
    model_summary = table4_df.groupby('model')[['avg_SS', 'avg_ST']].mean()
    x = np.arange(len(model_summary))
    ax1.bar(x - 0.15, model_summary['avg_SS'], 0.3, label='In-Domain', color='#2171b5')
    ax1.bar(x + 0.15, model_summary['avg_ST'], 0.3, label='Cross-Domain', color='#fc9272')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_summary.index)
    ax1.set_title('Avg Performance by Model')
    ax1.set_ylabel('Macro F1')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig10a_avg_performance_by_model.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig10a_avg_performance_by_model.png'))
    plt.show()
    print('Saved: fig10a_avg_performance_by_model.pdf/png')

    # --- Panel B: Worst-case drops ---
    fig, ax2 = plt.subplots(figsize=(6, 5))
    task_worst = table4_df.groupby('task')[['WSD', 'WTD']].max()
    task_worst.plot(kind='bar', ax=ax2, color=['#6baed6', '#fcbba1'], edgecolor='white')
    ax2.set_title('Worst-Case Drops by Task')
    ax2.set_ylabel('Worst F1 Drop')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(['Source Drop', 'Target Drop'], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig10b_worst_case_drops.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig10b_worst_case_drops.png'))
    plt.show()
    print('Saved: fig10b_worst_case_drops.pdf/png')

    # --- Panel C: Number of domains per task ---
    domain_counts = {}
    for task_name, df in all_results.items():
        domain_counts[task_name] = df['source'].nunique()
    if domain_counts:
        fig, ax3 = plt.subplots(figsize=(6, 5))
        tasks_sorted = sorted(domain_counts.keys())
        counts = [domain_counts[t] for t in tasks_sorted]
        colors_bar = ['#4a1486' if 'Nastaliq' in t else '#41ab5d' for t in tasks_sorted]
        ax3.barh(range(len(tasks_sorted)), counts, color=colors_bar, edgecolor='white')
        ax3.set_yticks(range(len(tasks_sorted)))
        ax3.set_yticklabels([t.split()[-1][:12] for t in tasks_sorted], fontsize=8)
        ax3.set_xlabel('Number of Domains')
        ax3.set_title('Domains per Task')
        n_patch = mpatches.Patch(color='#4a1486', label='Nastaliq')
        r_patch = mpatches.Patch(color='#41ab5d', label='Roman')
        ax3.legend(handles=[n_patch, r_patch], fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig10c_domains_per_task.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig10c_domains_per_task.png'))
        plt.show()
        print('Saved: fig10c_domains_per_task.pdf/png')

    # --- Panel D: XLM-R vs mBERT head-to-head ---
    head2head = []
    for task_name, df in all_results.items():
        xlmr_avg = df[(df['model'] == 'XLM-R_Base') & (df['type'] == 'CROSS-DOMAIN')]['macro_f1'].mean()
        mbert_avg = df[(df['model'] == 'mBERT') & (df['type'] == 'CROSS-DOMAIN')]['macro_f1'].mean()
        if not np.isnan(xlmr_avg) and not np.isnan(mbert_avg):
            head2head.append({'task': task_name.split()[-1][:10], 'XLM-R': xlmr_avg, 'mBERT': mbert_avg})
    if head2head:
        fig, ax4 = plt.subplots(figsize=(6, 5))
        h2h_df = pd.DataFrame(head2head)
        x = np.arange(len(h2h_df))
        ax4.plot(x, h2h_df['XLM-R'], 'o-', color='#2171b5', label='XLM-R Base', linewidth=2)
        ax4.plot(x, h2h_df['mBERT'], 's-', color='#cb181d', label='mBERT', linewidth=2)
        ax4.set_xticks(x)
        ax4.set_xticklabels(h2h_df['task'], rotation=45, fontsize=8)
        ax4.set_title('XLM-R vs mBERT (Cross-Domain)')
        ax4.set_ylabel('Avg Cross-Domain F1')
        ax4.legend(fontsize=8)
        ax4.set_ylim(0.3, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig10d_xlmr_vs_mbert.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig10d_xlmr_vs_mbert.png'))
        plt.show()
        print('Saved: fig10d_xlmr_vs_mbert.pdf/png')

    # --- Panel E: Source/Cross drop distribution ---
    if len(gap_df) > 0:
        fig, ax5 = plt.subplots(figsize=(6, 5))
        sns.kdeplot(data=gap_df, x='drop', hue='model', ax=ax5,
                   palette=['#2171b5', '#cb181d'], fill=True, alpha=0.3)
        ax5.set_title('Distribution of Performance Drops')
        ax5.set_xlabel('F1 Drop')
        ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig10e_drop_distribution.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig10e_drop_distribution.png'))
        plt.show()
        print('Saved: fig10e_drop_distribution.pdf/png')

    # --- Panel F: Script robustness comparison ---
    if len(gap_df) > 0:
        fig, ax6 = plt.subplots(figsize=(6, 5))
        script_summary = gap_df.groupby(['script', 'model'])['drop'].agg(['mean', 'std']).reset_index()
        x = np.arange(2)
        width = 0.35
        for i, model in enumerate(['XLM-R_Base', 'mBERT']):
            model_data = script_summary[script_summary['model'] == model]
            means = [model_data[model_data['script'] == s]['mean'].values[0] if len(model_data[model_data['script'] == s]) > 0 else 0 for s in ['Nastaliq', 'Roman']]
            stds = [model_data[model_data['script'] == s]['std'].values[0] if len(model_data[model_data['script'] == s]) > 0 else 0 for s in ['Nastaliq', 'Roman']]
            color = '#2171b5' if model == 'XLM-R_Base' else '#cb181d'
            ax6.bar(x + i*width - width/2, means, width, yerr=stds,
                   label=model, color=color, edgecolor='white', capsize=3)
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Nastaliq', 'Roman'])
        ax6.set_title('Avg Drop by Script')
        ax6.set_ylabel('Mean F1 Drop')
        ax6.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig10f_script_robustness.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig10f_script_robustness.png'))
        plt.show()
        print('Saved: fig10f_script_robustness.pdf/png')


# ============================================================
# Figure 11: Per-Domain Transfer Matrix (Clustermap)
# (already one file per task - no change needed)
# ============================================================
if n_tasks > 0:
    for task_name, df in all_results.items():
        subset = df[df['model'] == 'XLM-R_Base']
        if len(subset) == 0:
            continue

        domains_list = sorted(subset['source'].unique())
        if len(domains_list) < 3:
            continue

        matrix = pd.DataFrame(index=domains_list, columns=domains_list, dtype=float)
        for _, row in subset.iterrows():
            matrix.loc[row['source'], row['target']] = row['macro_f1']

        short_names = {d: d.split('_')[0] + '_' + d.split('_')[1][:6] for d in domains_list}
        matrix.rename(index=short_names, columns=short_names, inplace=True)

        g = sns.clustermap(matrix.astype(float), annot=True, fmt='.2f',
                          cmap='YlOrRd', figsize=(7, 6),
                          linewidths=0.5, vmin=0.3, vmax=1.0,
                          dendrogram_ratio=0.15)
        g.fig.suptitle(f'Domain Clustering: {task_name}', y=1.02)

        safe_name = task_name.replace(' ', '_').lower()
        plt.savefig(os.path.join(FIG_DIR, f'fig11_clustermap_{safe_name}.pdf'))
        plt.savefig(os.path.join(FIG_DIR, f'fig11_clustermap_{safe_name}.png'))
        plt.show()
        print(f'Saved: fig11_clustermap_{safe_name}.pdf/png')


# ============================================================
# Figure 12: LLM Few-Shot Performance Across Tasks (already a single plot)
# ============================================================
if table5_files:
    llm_perf = {}
    for task_name, df in all_results.items():
        llm_df = df[~df['model'].isin(['XLM-R_Base', 'mBERT'])]
        if len(llm_df) == 0:
            continue
        for model in llm_df['model'].unique():
            if model not in llm_perf:
                llm_perf[model] = {}
            avg_f1 = llm_df[llm_df['model'] == model]['macro_f1'].mean()
            llm_perf[model][task_name] = avg_f1

    if llm_perf:
        perf_df = pd.DataFrame(llm_perf).T

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(perf_df)))

        for idx, (model_name, row) in enumerate(perf_df.iterrows()):
            values = row.values
            x = np.arange(len(values))
            ax.plot(x, values, 'o-', color=colors[idx], linewidth=2, markersize=8, label=model_name)

        ax.set_xticks(np.arange(len(perf_df.columns)))
        ax.set_xticklabels([c.split()[-1][:12] for c in perf_df.columns], rotation=30, fontsize=9)
        ax.set_ylabel('Average Macro F1')
        ax.set_title('LLM Few-Shot Performance Across Tasks')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig12_llm_parallel_coords.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig12_llm_parallel_coords.png'))
        plt.show()
        print('Saved: fig12_llm_parallel_coords.pdf/png')
    else:
        print('No LLM results available yet.')


# ============================================================
# Figure 13: LLM Performance — Nastaliq vs Roman Script
# (split into two single-plot figures)
# ============================================================
if n_tasks > 0:
    llm_script_records = []
    for task_name, df in all_results.items():
        llm_df = df[~df['model'].isin(['XLM-R_Base', 'mBERT'])]
        if len(llm_df) == 0:
            continue
        script = 'Nastaliq' if 'Nastaliq' in task_name else 'Roman'
        for model in llm_df['model'].unique():
            model_df = llm_df[llm_df['model'] == model]
            in_domain = model_df[model_df['type'] == 'IN-DOMAIN']['macro_f1'].mean()
            cross_domain = model_df[model_df['type'] == 'CROSS-DOMAIN']['macro_f1'].mean()
            overall = model_df['macro_f1'].mean()
            llm_script_records.append({
                'model': model, 'script': script, 'task': task_name,
                'in_domain': in_domain, 'cross_domain': cross_domain, 'overall': overall,
            })

    llm_script_df = pd.DataFrame(llm_script_records)

    if len(llm_script_df) > 0:
        avg_by_model_script = llm_script_df.groupby(['model', 'script'])['overall'].mean().reset_index()
        models_order = avg_by_model_script.groupby('model')['overall'].mean().sort_values(ascending=False).index.tolist()

        x = np.arange(len(models_order))
        width = 0.35
        nastaliq_vals = [avg_by_model_script[(avg_by_model_script['model'] == m) & (avg_by_model_script['script'] == 'Nastaliq')]['overall'].values for m in models_order]
        roman_vals = [avg_by_model_script[(avg_by_model_script['model'] == m) & (avg_by_model_script['script'] == 'Roman')]['overall'].values for m in models_order]
        nastaliq_vals = [v[0] if len(v) > 0 else 0 for v in nastaliq_vals]
        roman_vals = [v[0] if len(v) > 0 else 0 for v in roman_vals]

        # --- Figure 13a: Overall average by LLM and script ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - width/2, nastaliq_vals, width, label='Nastaliq', color='#4a1486', edgecolor='white')
        ax.bar(x + width/2, roman_vals, width, label='Roman', color='#41ab5d', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(models_order, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Average Macro F1')
        ax.set_title('LLM Overall Performance by Script')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig13a_llm_overall_by_script.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig13a_llm_overall_by_script.png'))
        plt.show()
        print('Saved: fig13a_llm_overall_by_script.pdf/png')

        # --- Figure 13b: Cross-domain robustness gap by LLM and script ---
        llm_script_df['gap'] = llm_script_df['in_domain'] - llm_script_df['cross_domain']
        gap_by_model_script = llm_script_df.groupby(['model', 'script'])['gap'].mean().reset_index()

        nastaliq_gaps = [gap_by_model_script[(gap_by_model_script['model'] == m) & (gap_by_model_script['script'] == 'Nastaliq')]['gap'].values for m in models_order]
        roman_gaps = [gap_by_model_script[(gap_by_model_script['model'] == m) & (gap_by_model_script['script'] == 'Roman')]['gap'].values for m in models_order]
        nastaliq_gaps = [v[0] if len(v) > 0 else 0 for v in nastaliq_gaps]
        roman_gaps = [v[0] if len(v) > 0 else 0 for v in roman_gaps]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - width/2, nastaliq_gaps, width, label='Nastaliq', color='#4a1486', edgecolor='white')
        ax.bar(x + width/2, roman_gaps, width, label='Roman', color='#41ab5d', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(models_order, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('F1 Drop (In-Domain − Cross-Domain)')
        ax.set_title('LLM Robustness Gap by Script')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig13b_llm_robustness_gap.pdf'))
        plt.savefig(os.path.join(FIG_DIR, 'fig13b_llm_robustness_gap.png'))
        plt.show()
        print('Saved: fig13b_llm_robustness_gap.pdf/png')
    else:
        print('No LLM results available for script comparison.')


# ============================================================
# Figure 15: Fine-Tuned vs Best LLM Radar
# (split into one file per script instead of one subplot per script)
# ============================================================
if n_tasks > 0:
    method_records = []
    for task_name, df in all_results.items():
        script = 'Nastaliq' if 'Nastaliq' in task_name else 'Roman'
        ft_df = df[df['model'].isin(['XLM-R_Base', 'mBERT'])]
        if len(ft_df) > 0:
            ft_cross = ft_df[ft_df['type'] == 'CROSS-DOMAIN']['macro_f1'].mean()
            ft_in = ft_df[ft_df['type'] == 'IN-DOMAIN']['macro_f1'].mean()
            method_records.append({
                'task': task_name, 'script': script, 'method': 'Fine-tuned',
                'cross_domain': ft_cross, 'in_domain': ft_in,
            })
        llm_df = df[~df['model'].isin(['XLM-R_Base', 'mBERT'])]
        if len(llm_df) > 0:
            best_llm = llm_df.groupby('model')['macro_f1'].mean().idxmax()
            best_df = llm_df[llm_df['model'] == best_llm]
            llm_cross = best_df[best_df['type'] == 'CROSS-DOMAIN']['macro_f1'].mean()
            llm_in = best_df[best_df['type'] == 'IN-DOMAIN']['macro_f1'].mean()
            method_records.append({
                'task': task_name, 'script': script, 'method': f'Best LLM ({best_llm})',
                'cross_domain': llm_cross, 'in_domain': llm_in,
            })

    method_df = pd.DataFrame(method_records)

    if len(method_df) > 0 and method_df['method'].nunique() >= 2:
        for script in ['Nastaliq', 'Roman']:
            script_df = method_df[method_df['script'] == script]
            tasks_in_script = script_df['task'].unique()
            n_dims = len(tasks_in_script)

            fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))

            if n_dims == 0:
                ax.set_title(f'{script}\n(No data)')
            else:
                angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
                angles += angles[:1]

                ft_vals = []
                for t in tasks_in_script:
                    v = script_df[(script_df['task'] == t) & (script_df['method'] == 'Fine-tuned')]['cross_domain'].values
                    ft_vals.append(v[0] if len(v) > 0 else 0)
                ft_vals += ft_vals[:1]

                llm_vals = []
                for t in tasks_in_script:
                    v = script_df[(script_df['task'] == t) & (~script_df['method'].str.startswith('Fine'))]['cross_domain'].values
                    llm_vals.append(v[0] if len(v) > 0 else 0)
                llm_vals += llm_vals[:1]

                ax.plot(angles, ft_vals, 'o-', color='#2171b5', linewidth=2, label='Fine-tuned (avg)')
                ax.fill(angles, ft_vals, alpha=0.1, color='#2171b5')
                ax.plot(angles, llm_vals, 's-', color='#e6550d', linewidth=2, label='Best LLM (5-shot)')
                ax.fill(angles, llm_vals, alpha=0.1, color='#e6550d')

                short_labels = [t.split()[-1][:10] for t in tasks_in_script]
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(short_labels, fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_title(f'Fine-Tuned vs Best LLM: {script} Script', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

            plt.tight_layout()
            safe_script = script.lower()
            plt.savefig(os.path.join(FIG_DIR, f'fig15_ft_vs_llm_radar_{safe_script}.pdf'))
            plt.savefig(os.path.join(FIG_DIR, f'fig15_ft_vs_llm_radar_{safe_script}.png'))
            plt.show()
            print(f'Saved: fig15_ft_vs_llm_radar_{safe_script}.pdf/png')
    else:
        print('Need both fine-tuned and LLM results for gap analysis.')


# ============================================================
# Figure 9 (recomputed with expanded model palette; kept as a
# separate step exactly as it appeared in the original notebook,
# still a single plot — same file name, so it overwrites the
# earlier fig9 output, matching the original notebook's behavior)
# ============================================================
if len(size_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    model_palette = {
        'XLM-R_Base': '#2171b5',
        'mBERT': '#cb181d',
        'Qwen2.5-7B': '#e6550d',
        'Llama3.1-8B': '#31a354',
        'Mistral-7B': '#0170b8',
    }

    available_models = [m for m in model_palette.keys() if m in size_df['model'].unique()]

    sns.boxplot(data=size_df, x='task', y='macro_f1', hue='model',
               palette={m: model_palette[m] for m in available_models},
               ax=ax, width=0.6)

    ax.set_xlabel('')
    ax.set_ylabel('In-Domain Macro F1')
    ax.set_title('In-Domain Performance Distribution Across Tasks')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.legend(title='Model')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig9_indomain_distribution.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig9_indomain_distribution.png'))
    plt.show()
    print('Saved: fig9_indomain_distribution.pdf/png')


# ============================================================
# Summary
# ============================================================
print('\n=== Generated Figures ===')
fig_files = sorted(glob.glob(os.path.join(FIG_DIR, '*')))
for f in fig_files:
    print(f'  {os.path.basename(f)}')
print(f'\nTotal: {len(fig_files)} files in {FIG_DIR}')