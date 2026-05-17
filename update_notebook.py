import json
import os

notebook_path = 'scripts/ESCan_ffdtf_envelpes.ipynb'
marker = '# TD_ASD_V2'

def create_cell1():
    code = f"""{marker}
from IPython.display import Markdown, display
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

display(Markdown('### TD vs ASD: Split-Violin Distributions'))

# Helper for labels
def get_group_label(g):
    g = str(g).lower()
    if 'td' in g or 'typ' in g: return 'TD'
    if 'asd' in g or 'aut' in g: return 'ASD'
    return None

df = ffdtf_env_df[(ffdtf_env_df['pair_type'] == 'real') & (ffdtf_env_df['edge_type'] == 'cross-brain')].copy()
df['Group'] = df['group'].apply(get_group_label)
df = df[df['Group'].notnull()]

events = df['event'].unique()
edges = df['edge'].unique()

for event in events:
    plt.figure(figsize=(15, 5))
    sns.violinplot(data=df[df['event']==event], x='edge', y='ffdtf_env', hue='Group', split=True, inner='quart')
    plt.title(f'Event: {{event}}')
    plt.xticks(rotation=45)
    plt.show()
"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": code.splitlines(True)
    }

def create_cell2():
    code = f"""{marker}
from IPython.display import Markdown, display
import numpy as np
import pandas as pd
from scipy import stats

display(Markdown('### TD vs ASD: Mean Difference (Permutation Test)'))

def _perm_pvalue_two_sided(x1, x2, n_perm=1000):
    obs_diff = np.abs(np.mean(x1) - np.mean(x2))
    combined = np.concatenate([x1, x2])
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        diff = np.abs(np.mean(combined[:len(x1)]) - np.mean(combined[len(x1):]))
        if diff >= obs_diff: count += 1
    return count / n_perm

def _bh_fdr(ps):
    from statsmodels.stats.multitest import multipletests
    return multipletests(ps, method='fdr_bh')[1]

df = ffdtf_env_df[(ffdtf_env_df['pair_type'] == 'real') & (ffdtf_env_df['edge_type'] == 'cross-brain')].copy()
df['Group'] = df['group'].apply(lambda g: 'TD' if 'td' in str(g).lower() or 'typ' in str(g).lower() else ('ASD' if 'asd' in str(g).lower() or 'aut' in str(g).lower() else None))
df = df[df['Group'].notnull()]

dyad_means = df.groupby(['dyad_id', 'Group', 'edge', 'event'])['ffdtf_env'].mean().reset_index()

results = []
for event in dyad_means['event'].unique():
    ev_df = dyad_means[dyad_means['event'] == event]
    for edge in ev_df['edge'].unique():
        td_vals = ev_df[(ev_df['edge'] == edge) & (ev_df['Group'] == 'TD')]['ffdtf_env'].values
        asd_vals = ev_df[(ev_df['edge'] == edge) & (ev_df['Group'] == 'ASD')]['ffdtf_env'].values
        if len(td_vals) > 0 and len(asd_vals) > 0:
            p = _perm_pvalue_two_sided(td_vals, asd_vals)
            results.append({{'event': event, 'edge': edge, 'p': p}})

res_df = pd.DataFrame(results)
if not res_df.empty:
    res_df['p_adj'] = _bh_fdr(res_df['p'])
    display(res_df[res_df['p_adj'] < 0.05])
else:
    print("No results to display.")
"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": code.splitlines(True)
    }

def create_cell3():
    code = f"""{marker}
from IPython.display import Markdown, display
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

display(Markdown('### TD vs ASD: Distribution Shape (KS + Permutation)'))

def _perm_pvalue_ks(x1, x2, n_perm=1000):
    obs_ks = ks_2samp(x1, x2).statistic
    combined = np.concatenate([x1, x2])
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        ks = ks_2samp(combined[:len(x1)], combined[len(x1):]).statistic
        if ks >= obs_ks: count += 1
    return count / n_perm

df = ffdtf_env_df[(ffdtf_env_df['pair_type'] == 'real') & (ffdtf_env_df['edge_type'] == 'cross-brain')].copy()
df['Group'] = df['group'].apply(lambda g: 'TD' if 'td' in str(g).lower() or 'typ' in str(g).lower() else ('ASD' if 'asd' in str(g).lower() or 'aut' in str(g).lower() else None))
df = df[df['Group'].notnull()]

results = []
for event in df['event'].unique():
    ev_df = df[df['event'] == event]
    for edge in ev_df['edge'].unique():
        td_vals = ev_df[(ev_df['edge'] == edge) & (ev_df['Group'] == 'TD')]['ffdtf_env'].values
        asd_vals = ev_df[(ev_df['edge'] == edge) & (ev_df['Group'] == 'ASD')]['ffdtf_env'].values
        if len(td_vals) > 0 and len(asd_vals) > 0:
            p = _perm_pvalue_ks(td_vals, asd_vals)
            results.append({{'event': event, 'edge': edge, 'p': p}})

res_df = pd.DataFrame(results)
if not res_df.empty:
    from statsmodels.stats.multitest import multipletests
    res_df['p_adj'] = multipletests(res_df['p'], method='fdr_bh')[1]
    display(res_df[res_df['p_adj'] < 0.05])
else:
    print("No results to display.")
"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": code.splitlines(True)
    }

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Idempotency: remove old cells with marker
nb['cells'] = [c for c in nb['cells'] if not (c['cell_type'] == 'code' and any(marker in line for line in c['source']))]

# Append new cells
nb['cells'].extend([create_cell1(), create_cell2(), create_cell3()])

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Total cells: {len(nb['cells'])}")
for i in range(3, 0, -1):
    print(f"Cell -{i} first line: {nb['cells'][-i]['source'][0].strip()}")
