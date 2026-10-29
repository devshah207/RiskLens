"""Read-only Optuna study audit. No model training or writes to the study DB."""
import argparse
import ast
import json
from pathlib import Path
import pandas as pd


def run(root: Path):
    output = {'warning': 'Development tuning audit only; no independent test conclusions.', 'models': {}}
    for name in ('lstm', 'xgboost'):
        filename = root / f'{name}_trials.csv'
        if not filename.exists():
            output['models'][name] = {'error': f'Missing {filename.name}; run scripts/export_optuna_evidence.py first'}
            continue
        frame = pd.read_csv(filename)
        if 'state' not in frame or 'value' not in frame:
            raise ValueError(f'{filename} missing state/value columns')
        completed = frame[(frame.state == 'COMPLETE') & (frame.value.notna())].copy()
        completed['value'] = pd.to_numeric(completed['value'], errors='coerce')
        completed = completed.dropna(subset=['value']).sort_values('value').reset_index(drop=True)
        if completed.empty:
            output['models'][name] = {'completed': 0}
            continue
        k = max(1, int(len(completed) * .10))
        elite = completed.head(k)
        params = [c for c in frame.columns if c.startswith('params_')]
        sections = {}
        for col in params:
            vals = elite[col].dropna()
            if not len(vals): continue
            if pd.api.types.is_numeric_dtype(vals):
                sections[col.removeprefix('params_')] = {'top_10pct_median': float(vals.median()), 'top_10pct_min': float(vals.min()), 'top_10pct_max': float(vals.max())}
            else:
                sections[col.removeprefix('params_')] = {'top_10pct_most_common': str(vals.mode().iloc[0]), 'top_10pct_distinct': int(vals.nunique())}
        entry = {'completed': len(completed), 'total_trials': len(frame), 'best_trial': int(completed.iloc[0]['number']), 'best_brier': float(completed.iloc[0]['value']), 'median_brier': float(completed.value.median()), 'top_10pct_count': k, 'best_params': {p.removeprefix('params_'): (None if pd.isna(completed.iloc[0][p]) else completed.iloc[0][p].item() if hasattr(completed.iloc[0][p], 'item') else completed.iloc[0][p]) for p in params}, 'top_parameter_ranges': sections}
        year_col = 'user_attrs_tuning_per_year'
        if year_col in completed:
            per_year = []
            for _, trial in elite.iterrows():
                try:
                    yearly = ast.literal_eval(trial[year_col])
                    if isinstance(yearly, list) and yearly:
                        per_year.append({'trial': int(trial.number), 'brier_by_year': {str(x['year']): float(x['brier']) for x in yearly}, 'range_across_years': float(max(x['brier'] for x in yearly) - min(x['brier'] for x in yearly))})
                except (ValueError, SyntaxError, TypeError, KeyError):
                    continue
            entry['top_trial_yearly_brier'] = per_year[:20]
        output['models'][name] = entry
    path = root / 'optuna_audit.json'
    path.write_text(json.dumps(output, indent=2, default=str) + '\n')
    print('Saved:', path)
    for name, item in output['models'].items():
        print(name, 'completed', item.get('completed'), 'best tuning Brier', item.get('best_brier'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    run(ap.parse_args().directory)
