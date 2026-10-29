"""CLI: point-in-time historical VaR backtest for development risk analysis."""
import argparse
from risklens.models.market_risk_monitor import save_var_backtest

def main():
    p=argparse.ArgumentParser(description='RiskLens 95% and 99% VaR exception backtest (development only)')
    p.add_argument('nifty_csv');p.add_argument('xgboost_predictions_csv');p.add_argument('--output-dir',required=True);p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    report=save_var_backtest(args.nifty_csv,args.xgboost_predictions_csv,args.output_dir,overwrite=args.overwrite)
    for k,v in report['results'].items():
        print(k,'exceptions:',v['exception_count'],'/',v['n_observations'],'observed:',f"{v['observed_exception_rate']:.2%}",'expected:',f"{v['expected_exception_rate']:.2%}")

if __name__=='__main__':main()
