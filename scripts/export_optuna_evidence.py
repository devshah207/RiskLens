"""Export incomplete/resumable Optuna studies without starting any new trials."""
import json
import argparse
from pathlib import Path
import optuna

def main():
    p=argparse.ArgumentParser();p.add_argument('output_dir');a=p.parse_args()
    out=Path(a.output_dir);storage='sqlite:///'+str((out/'optuna_studies.sqlite3').resolve())
    report={'important':'Tuning results are developmental, not holdout predictions. 2025+ untouched.','studies':{}}
    for name in ('xgboost','lstm'):
        try:study=optuna.load_study(study_name='risklens_'+name,storage=storage)
        except KeyError:continue
        study.trials_dataframe().to_csv(out/f'{name}_trials.csv',index=False)
        complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
        report['studies'][name]={'completed_trials':len(complete),'total_trials_including_interrupted':len(study.trials),
           'best_tuning_brier':study.best_value if complete else None,
           'best_params':study.best_params if complete else None}
    (out/'optuna_search_progress.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
