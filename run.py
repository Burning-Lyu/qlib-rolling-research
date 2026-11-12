"""Alpha158 + LightGBM: fixed model, monthly rolling retraining and turnover.

python run.py --help
Each execution creates fresh outputs. See README.md for the fixed protocol.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile

# Set before importing Qlib. Artifacts and caches have separate locations.
ROOT = Path(__file__).resolve().parent
runtime = Path(tempfile.gettempdir()) / "qlib_rolling_research_runtime"
for folder in ["appdata", "localappdata", "mlflow"]:
    (runtime / folder).mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(runtime / "appdata")
os.environ["LOCALAPPDATA"] = str(runtime / "localappdata")
os.environ["MLFLOW_TRACKING_URI"] = (runtime / "mlflow").as_uri()
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
import requests
import scipy
import sklearn
from qlib.backtest import backtest
from qlib.config import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import RobustZScoreNorm, Fillna, DropnaLabel, CSRankNorm

LOCK = json.loads((ROOT / "data_snapshot.json").read_text(encoding="utf-8"))


SEGMENTS = {'train': ('2008-01-02', '2014-12-29'), 'valid': ('2015-01-05', '2016-12-28'), 'development': ('2017-01-03', '2018-12-26')}


INFER_PROCESSORS = [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}]


LEARN_PROCESSORS = [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}]


MODEL_PARAMS = {'objective': 'regression', 'metric': 'l2', 'verbosity': -1, 'colsample_bytree': 0.8879, 'learning_rate': 0.2, 'subsample': 0.8789, 'lambda_l1': 205.6999, 'lambda_l2': 580.9768, 'max_depth': 8, 'num_leaves': 210, 'num_threads': 8, 'seed': 20261027, 'feature_fraction_seed': 20261027, 'bagging_seed': 20261027, 'data_random_seed': 20261027, 'deterministic': True, 'force_col_wise': True}


NUM_BOOST_ROUND = 1000


EARLY_STOPPING_ROUNDS = 50


INITIAL_CASH = 100_000_000


TOPK = 50


FROZEN_N_DROP = 5


N_DROPS = [5, 1, 3]


EXCHANGE = {
    "freq": "day", "deal_price": "close", "limit_threshold": 0.095,
    "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
    "trade_unit": 100,
}


def save_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_hashes(folder):
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.name != "sha256.txt")
    (folder / "sha256.txt").write_text(
        "\n".join(f"{p.name}  {sha256(p)}" for p in paths) + "\n", encoding="utf-8"
    )


def prepare_data(archive, data, download):
    """校验整个固定归档；核对研究涉及的解压文件，避免读取无关股票。"""
    if not archive.is_file():
        if not download:
            raise FileNotFoundError(f"{archive}; use --download or --archive PATH")
        archive.parent.mkdir(parents=True, exist_ok=True)
        # 临时文件和最终文件位于同一磁盘，校验完成才移动到正式路径。
        with tempfile.TemporaryDirectory(prefix="download_", dir=archive.parent) as tmp:
            part = Path(tmp) / "archive.part"
            with requests.get(LOCK["download_url"], stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                with part.open("wb") as stream:
                    for block in response.iter_content(2**20):
                        stream.write(block)
            if part.stat().st_size != LOCK["archive_size_bytes"] or sha256(part) != LOCK["archive_sha256"]:
                raise ValueError("Downloaded archive differs from the pinned snapshot.")
            part.replace(archive)
    print("Checking complete snapshot archive...", flush=True)
    if archive.stat().st_size != LOCK["archive_size_bytes"] or sha256(archive) != LOCK["archive_sha256"]:
        raise ValueError("Snapshot size or SHA256 mismatch.")
    if not data.exists():
        data.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="extract_", dir=data.parent) as tmp:
            with tarfile.open(archive) as bundle:
                bundle.extractall(tmp, filter="data")
            shutil.move(str(Path(tmp) / "qlib_bin"), str(data))
    # 根据历史成员选出本研究实际读取的文件。股票池本身也须通过归档核对。
    members = pd.read_csv(data / "instruments/csi300.txt", sep="\t",
                          names=["symbol", "start", "end"], parse_dates=["start", "end"])
    symbols = set(members.loc[(members.end >= "2008-01-01") &
                             (members.start <= "2020-07-31"), "symbol"].str.lower())
    symbols.add("sh000300")
    fields = ["open", "high", "low", "close", "adjclose", "factor",
              "volume", "amount", "vwap", "change"]
    required = {f"features/{s}/{f}.day.bin" for s in symbols - {"sh000300"} for f in fields}
    required.add("features/sh000300/close.day.bin")
    required.update(["calendars/day.txt", "instruments/csi300.txt"])
    checked = set()
    print(f"Checking {len(required)} required data files and calendar/instrument metadata...", flush=True)
    # 校验已有目录，避免归档正确、实际行情被修改的情况。
    count = 0
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            relative = Path(member.name).relative_to("qlib_bin")
            name = relative.as_posix()
            if name not in required and relative.parts[0] not in {"calendars", "instruments"}:
                continue
            path = data / relative
            if not path.is_file() or path.stat().st_size != member.size:
                raise ValueError(f"Missing or changed snapshot file: {relative}")
            with bundle.extractfile(member) as stream:
                expected = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha256(path) != expected:
                raise ValueError(f"Changed snapshot file: {relative}")
            checked.add(name)
            count += 1
            if count % 1000 == 0:
                print(f"Verified {count} data files.", flush=True)
    if not required.issubset(checked):
        raise ValueError(f"Required fields missing from archive: {sorted(required - checked)[:5]}")
    print(f"Verified complete archive and {count} study/metadata files.", flush=True)


def require(condition, message):
    """关键数据或账户检查失败时中止运行，保留已有诊断。"""
    if not condition:
        raise ValueError(message)


def check_reference(main_results, cost_summary):
    """与随仓库保存的历史数值比较；原始历史结果保持冻结。"""
    reference = json.loads((ROOT / "reference_results.json").read_text(encoding="utf-8"))
    actual = pd.concat([
        main_results[main_results.experiment.isin(["B0", "A"])].assign(
            kind="main", key=lambda f: f.scope + "/" + f.experiment,
        ),
        cost_summary.query("period == 'all'").assign(
            kind="cost", key=lambda f: f.model + "/" + f.cost_multiplier.map(lambda x: f"{x:g}"),
        ),
    ], ignore_index=True).set_index(["kind", "key"])
    differences = []
    for record in reference["metrics"]:
        row = actual.loc[(record["kind"], record["key"])]
        for key, expected in record["values"].items():
            value = float(row[key])
            if not np.isclose(value, expected, rtol=1e-8, atol=1e-10):
                differences.append({"kind": record["kind"], "key": record["key"],
                                    "metric": key, "expected": expected, "actual": value})
    result = {"matched": not differences, "rtol": 1e-8, "atol": 1e-10,
              "differences": differences, "reference_sha256": sha256(ROOT / "reference_results.json")}
    save_json(RUN / "reproduction_check.json", result)
    print(f"Historical metric comparison: {'PASS' if not differences else 'DIFFERENCES FOUND'}", flush=True)
    return result


def read_field(symbol, field):
    raw = np.fromfile(DATA / "features" / symbol.lower() / f"{field}.day.bin", dtype="<f4")
    require(len(raw) > 1 and np.isfinite(raw[0]) and raw[0] == int(raw[0]),
            f"Invalid binary header: {symbol}/{field}")
    start = int(raw[0])
    require(start >= 0 and start + len(raw) - 1 <= len(calendar), "Binary exceeds calendar.")
    return pd.Series(raw[1:], index=calendar[start:start + len(raw) - 1], name=field)


def read_stock(symbol):
    return pd.concat([read_field(symbol, field) for field in fields], axis=1)


def member_dates_for(symbol):
    mask = pd.Series(False, index=study_dates)
    for row in study_members[study_members.symbol == symbol].itertuples():
        mask.loc[(mask.index >= row.start) & (mask.index <= row.end)] = True
    return mask.index[mask]


def fit_model(train, valid):
    for name, frame in [("train", train), ("valid", valid)]:
        require(not frame.empty and frame.index.is_unique, f"Invalid {name} keys.")
        require(frame["feature"].shape[1] == 158, "Alpha158 dimension mismatch.")
        require(np.isfinite(frame.to_numpy()).all(), f"Non-finite {name} values.")
    train_set = lgb.Dataset(train["feature"], label=train["label"].iloc[:, 0])
    valid_set = lgb.Dataset(valid["feature"], label=valid["label"].iloc[:, 0], reference=train_set)
    history = {}
    model = lgb.train(
        MODEL_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, valid_set], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                   lgb.log_evaluation(period=20), lgb.record_evaluation(history)],
    )
    return model, history


def predict_frame(model, frame):
    require(frame.index.is_unique and np.isfinite(frame["feature"].to_numpy()).all(),
            "Invalid prediction features.")
    return pd.DataFrame({
        "score": model.predict(frame["feature"], num_iteration=model.best_iteration),
        "label": frame["label"].iloc[:, 0],
    }, index=frame.index).reorder_levels(["datetime", "instrument"]).sort_index()


def save_prediction_tables(predictions, folder, filename):
    rows = []
    for date, part in predictions.dropna().groupby(level="datetime", sort=True):
        rows.append({
            "datetime": date, "IC": part.score.corr(part.label),
            "RankIC": part.score.corr(part.label, method="spearman"),
            "sample_count": len(part),
        })
    daily = pd.DataFrame(rows).set_index("datetime")
    predictions.to_csv(folder / filename, compression="gzip")
    daily.to_csv(folder / "daily_ic.csv")
    daily.groupby(daily.index.to_period("M")).agg(
        IC=("IC", "mean"), RankIC=("RankIC", "mean"),
        days=("RankIC", "count"), paired_rows=("sample_count", "sum"),
    ).to_csv(folder / "monthly_ic.csv")
    return daily


class RecordedStrategy(TopkDropoutStrategy):
    """沿用 Qlib 策略，通过两个记录方法保存时点与实际成交。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timing_rows = []
        self.trade_rows = []

    def generate_trade_decision(self, execute_result=None):
        step = self.trade_calendar.get_trade_step()
        trade_start, _ = self.trade_calendar.get_step_time(step)
        signal_start, signal_end = self.trade_calendar.get_step_time(step, shift=1)
        scores = self.signal.get_signal(signal_start, signal_end)
        self.trade_date = trade_start.normalize()
        self.signal_date = signal_start.normalize()
        self.timing_rows.append({'n_drop': self.n_drop, 'trade_date': self.trade_date, 'signal_date': self.signal_date, 'has_signal': scores is not None, 'signal_count': 0 if scores is None else len(scores)})
        return super().generate_trade_decision(execute_result)

    def post_exe_step(self, execute_result):
        for order, value, cost, price in execute_result:
            self.trade_rows.append({'n_drop': self.n_drop, 'trade_date': self.trade_date, 'signal_date': self.signal_date, 'instrument': order.stock_id, 'direction': 'buy' if order.direction == 1 else 'sell', 'requested_amount': order.amount, 'deal_amount': order.deal_amount, 'deal_value': value, 'deal_cost': cost, 'adjusted_price': price, 'factor': order.factor})
        super().post_exe_step(execute_result)


def metrics(frame):
    """从日报计算统一研究指标。年化约定为 252 天，标准差取样本标准差。"""
    net_return = frame['return'] - frame['cost']
    net_value = (1 + net_return).cumprod()
    gross_value = (1 + frame['return']).prod()
    benchmark_value = (1 + frame['bench']).prod()
    active = net_return - frame['bench']
    active_std = active.std(ddof=1)
    peak = net_value.cummax().clip(lower=1)
    return {'days': len(frame), 'net_total_return': float(net_value.iloc[-1] - 1), 'net_cagr': float(net_value.iloc[-1] ** (252 / len(frame)) - 1), 'gross_cagr': float(gross_value ** (252 / len(frame)) - 1), 'benchmark_cagr': float(benchmark_value ** (252 / len(frame)) - 1), 'information_ratio': float(active.mean() / active_std * np.sqrt(252)), 'max_drawdown': float((net_value / peak - 1).min()), 'mean_two_way_turnover': float(frame['turnover'].mean()), 'sum_cost_rate': float(frame['cost'].sum())}


def run_backtest(predictions, daily_ic, start, end, n_drops, folder, multiplier=1.0):
    folder.mkdir(parents=True)
    signal = predictions["score"].sort_index()
    require(signal.index.is_unique and np.isfinite(signal).all(), "Invalid signal.")
    expected_dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    exchange = dict(EXCHANGE)
    exchange["codes"] = sorted(predictions.index.get_level_values("instrument").unique())
    for key in ["open_cost", "close_cost", "min_cost"]:
        exchange[key] *= multiplier
    reports, trades, positions, timings, rows, diagnostics = [], [], [], [], [], []
    for n_drop in n_drops:
        print(f"回测 {folder.name}，n_drop={n_drop}，费用倍数={multiplier}")
        strategy = RecordedStrategy(
            signal=signal, topk=TOPK, n_drop=n_drop, risk_degree=0.95,
            method_buy="top", method_sell="bottom", hold_thresh=1,
            only_tradable=False, forbid_all_trade_at_limit=True,
        )
        portfolio, indicators = backtest(
            start_time=start, end_time=end, strategy=strategy,
            executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                      "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
            account=INITIAL_CASH, benchmark="SH000300", exchange_kwargs=exchange,
        )
        report, hist_positions = portfolio["1day"]
        report = report.sort_index()
        require(report.index.equals(expected_dates), "Backtest dates differ from the market calendar.")
        report.index.name = "datetime"
        trade_frame = pd.DataFrame(strategy.trade_rows)
        timing = pd.DataFrame(strategy.timing_rows)
        previous = report.account.shift(1).fillna(INITIAL_CASH)
        report["n_drop"] = n_drop
        report["net_return"] = report["return"] - report.cost
        report["net_value"] = (1 + report.net_return).cumprod()
        report["benchmark_value"] = (1 + report.bench).cumprod()
        report["holding_count"] = [len(hist_positions[d].get_stock_list()) for d in report.index]
        daily_value = trade_frame.groupby("trade_date").deal_value.sum().reindex(report.index, fill_value=0)
        daily_cost = trade_frame.groupby("trade_date").deal_cost.sum().reindex(report.index, fill_value=0)
        diagnostics.append({
            "n_drop": n_drop,
            "account_return_max_error": float((report.net_return - (report.account / previous - 1)).abs().max()),
            "turnover_max_error": float((report.turnover - daily_value / previous).abs().max()),
            "cost_max_error": float((report.cost - daily_cost / previous).abs().max()),
            "minimum_cash": float(report.cash.min()),
            "same_or_future_signal_rows": int((timing.signal_date >= timing.trade_date).sum()),
            "final_day_orders": int((trade_frame.trade_date == pd.Timestamp(end)).sum()),
        })
        last = diagnostics[-1]
        require(max(last[k] for k in ["account_return_max_error", "turnover_max_error",
                                      "cost_max_error"]) < 1e-9, "Account reconciliation failed.")
        require(last["minimum_cash"] >= -1e-6 and last["same_or_future_signal_rows"] == 0,
                "Negative cash or future signal.")
        previous_dates = pd.Series(calendar[:-1], index=calendar[1:])
        require((timing.signal_date.to_numpy() ==
                 timing.trade_date.map(previous_dates).to_numpy()).all(), "Signal lag differs from one day.")
        require(last["final_day_orders"] == 0 and not timing.iloc[-1].has_signal,
                "Final day must only value existing holdings.")
        executed = trade_frame[trade_frame.deal_value > 0]
        for row in executed.itertuples():
            rate = exchange["open_cost"] if row.direction == "buy" else exchange["close_cost"]
            require(np.isclose(row.deal_cost, max(exchange["min_cost"], row.deal_value * rate),
                               rtol=1e-9, atol=1e-6), "Trade fee mismatch.")
            if row.direction == "buy":
                require((row.signal_date, row.instrument) in signal.index,
                        "Buy outside historical candidate list.")
        reports.append(report.reset_index())
        trades.append(trade_frame)
        timings.append(timing)
        for date, position in hist_positions.items():
            for symbol in sorted(position.get_stock_list()):
                positions.append({
                    "n_drop": n_drop, "datetime": date, "instrument": symbol,
                    "amount": position.get_stock_amount(symbol),
                    "adjusted_price": position.get_stock_price(symbol),
                    "weight": position.get_stock_weight(symbol),
                    "holding_days": position.get_stock_count(symbol, bar="day"),
                })
        for period, part in [("all", report)] + [
            (str(year), report.loc[str(year)]) for year in report.index.year.unique()
        ]:
            rank_ic = daily_ic.RankIC if period == "all" else daily_ic.loc[period, "RankIC"]
            rows.append({"n_drop": n_drop, "period": period,
                         "mean_daily_rank_ic": float(rank_ic.mean()), **metrics(part)})
    summary = pd.DataFrame(rows)
    combined_reports = pd.concat(reports, ignore_index=True)
    combined_trades = pd.concat(trades, ignore_index=True)
    combined_positions = pd.DataFrame(positions)
    combined_reports.to_csv(folder / "daily_reports.csv.gz", index=False, compression="gzip")
    combined_trades.to_csv(folder / "trades.csv.gz", index=False, compression="gzip")
    combined_positions.to_csv(folder / "positions.csv.gz", index=False, compression="gzip")
    pd.concat(timings, ignore_index=True).to_csv(folder / "signal_timing.csv", index=False)
    summary.to_csv(folder / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(folder / "account_diagnostics.csv", index=False)
    save_json(folder / "run.json", {
        "start": start, "end": end, "n_drops": n_drops,
        "cost_multiplier": multiplier, "exchange": exchange,
        "initial_cash": INITIAL_CASH, "topk": TOPK, "risk_degree": 0.95,
    })
    save_hashes(folder)
    return {"summary": summary, "reports": combined_reports,
            "trades": combined_trades, "positions": combined_positions,
            "timings": pd.concat(timings, ignore_index=True)}


def load_raw_alpha(start, end):
    handler = Alpha158(
        instruments="csi300", start_time=start, end_time=end,
        infer_processors=[], learn_processors=[],
    )
    return handler.fetch(
        col_set=["feature", "label"], data_key=DataHandlerLP.DK_R,
    ).reorder_levels(["datetime", "instrument"]).sort_index()


def monthly_windows(month, calendar, label_end, prediction_dates):
    first = calendar[calendar.to_period("M") == month][0]
    update = calendar[calendar.get_loc(first) - 1]
    valid_start = month.start_time - pd.DateOffset(months=6)
    train_start = valid_start - pd.DateOffset(years=5)
    train_dates = calendar[
        (calendar >= train_start) & (calendar < valid_start) &
        (calendar.map(label_end) < valid_start)
    ]
    valid_dates = calendar[
        (calendar >= valid_start) & (calendar <= update) &
        (calendar.map(label_end) <= update)
    ]
    predict_dates = prediction_dates[prediction_dates.to_period("M") == month]
    return update, train_dates, valid_dates, predict_dates


def monthly_retrain(raw, months, prediction_dates, folder, filename):
    folder.mkdir(parents=True)
    predictions, records = [], []
    raw_dates = raw.index.get_level_values("datetime")
    with zipfile.ZipFile(folder / "monthly_models.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for month in months:
            started = time.perf_counter()
            update, train_dates, valid_dates, predict_dates = monthly_windows(
                month, calendar, label_end, prediction_dates,
            )
            train_raw = raw.loc[raw_dates.isin(train_dates)].copy()
            valid_raw = raw.loc[raw_dates.isin(valid_dates)].copy()
            predict_raw = raw.loc[raw_dates.isin(predict_dates)].copy()
            require(all(not f.empty for f in [train_raw, valid_raw, predict_raw]), "Empty monthly segment.")
            require(label_end.loc[train_dates].max() < valid_dates.min(), "Training label crosses validation.")
            require(label_end.loc[valid_dates].max() <= update < predict_dates.min(), "Immature validation label.")
            norm = RobustZScoreNorm(
                train_dates.min(), train_dates.max(), fields_group="feature", clip_outlier=True,
            )
            norm.fit(train_raw)
            train = CSRankNorm("label")(DropnaLabel()(Fillna("feature")(norm(train_raw))))
            valid = CSRankNorm("label")(DropnaLabel()(Fillna("feature")(norm(valid_raw))))
            predict = Fillna("feature")(norm(predict_raw))
            print(f"{month}：更新日 {update.date()}，训练 {len(train)} 行，验证 {len(valid)} 行")
            model, history = fit_model(train, valid)
            prediction = predict_frame(model, predict)
            predictions.append(prediction)
            model_text = model.model_to_string(num_iteration=model.best_iteration)
            archive.writestr(f"{month}/model.txt", model_text)
            buffer = io.BytesIO()
            np.savez(buffer, median=norm.mean_train, scale=norm.std_train,
                     feature_names=np.array(predict["feature"].columns.tolist(), dtype=str))
            archive.writestr(f"{month}/normalization.npz", buffer.getvalue())
            archive.writestr(f"{month}/training_history.json", json.dumps(history))
            records.append({
                "model_id": str(month), "update_after_close": str(update.date()),
                "train_start": str(train_dates.min().date()), "train_end": str(train_dates.max().date()),
                "valid_start": str(valid_dates.min().date()), "valid_end": str(valid_dates.max().date()),
                "latest_train_label_end": str(label_end.loc[train_dates].max().date()),
                "latest_valid_label_end": str(label_end.loc[valid_dates].max().date()),
                "prediction_start": str(predict_dates.min().date()),
                "prediction_end": str(predict_dates.max().date()),
                "train_rows": len(train), "valid_rows": len(valid), "prediction_rows": len(prediction),
                "best_iteration": model.best_iteration, "best_valid_l2": model.best_score["valid"]["l2"],
                "model_sha256": hashlib.sha256(model_text.encode()).hexdigest(),
                "seconds": time.perf_counter() - started,
            })
            pd.DataFrame(records).to_csv(folder / "monthly_training.csv", index=False)
            del train_raw, valid_raw, predict_raw, train, valid, predict, model
            gc.collect()
    predictions = pd.concat(predictions).sort_index()
    expected = raw.loc[raw_dates.isin(prediction_dates)].sort_index()
    pd.testing.assert_index_equal(predictions.index, expected.index)
    require(predictions.index.is_unique, "Repeated monthly prediction keys.")
    daily = save_prediction_tables(predictions, folder, filename)
    windows = pd.DataFrame(records)
    windows["train_labels_before_validation"] = windows.latest_train_label_end < windows.valid_start
    windows["validation_labels_mature"] = windows.latest_valid_label_end <= windows.update_after_close
    windows["update_before_prediction"] = windows.update_after_close < windows.prediction_start
    windows.to_csv(folder / "window_diagnostics.csv", index=False)
    save_hashes(folder)
    return predictions, daily, windows


def experiment_table(fixed, monthly, scope):
    rows = []
    for name, result in [("B0", fixed), ("A", monthly), ("B", fixed), ("AB", monthly)]:
        row = result["summary"].query("period == 'all' and n_drop == 5").iloc[0].to_dict()
        rows.append({"scope": scope, "experiment": name, **row})
    return pd.DataFrame(rows)


def comparison_plot(fixed, monthly, fixed_ic, monthly_ic, title, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for name, result in [("Fixed", fixed), ("Monthly", monthly)]:
        report = result["reports"].query("n_drop == 5").set_index("datetime")
        axes[0].plot(report.index, report.net_value, label=name)
    report = fixed["reports"].query("n_drop == 5").set_index("datetime")
    axes[0].plot(report.index, report.benchmark_value, "--", color="black", label="CSI300")
    for name, daily in [("Fixed", fixed_ic), ("Monthly", monthly_ic)]:
        monthly_ic_mean = daily.RankIC.groupby(daily.index.to_period("M")).mean()
        axes[1].plot(monthly_ic_mean.index.to_timestamp(), monthly_ic_mean, marker="o", label=name)
    axes[0].set(title=title, ylabel="Net value")
    axes[1].set(ylabel="Monthly mean Rank IC", xlabel="Date")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def contribution_analysis(result, model):
    report = result["reports"].query("n_drop == 5").set_index("datetime")
    positions = result["positions"].query("n_drop == 5").copy()
    trades = result["trades"].query("n_drop == 5").copy()
    previous_account = report.account.shift(1).fillna(INITIAL_CASH)
    prices = positions[["datetime", "instrument", "adjusted_price"]]
    trade_prices = trades[trades.deal_value > 0][
        ["trade_date", "instrument", "adjusted_price"]
    ].rename(columns={"trade_date": "datetime"})
    prices = pd.concat([prices, trade_prices]).drop_duplicates(["datetime", "instrument"])
    next_date = dict(zip(report.index[:-1], report.index[1:]))
    previous = positions[positions.datetime < report.index[-1]].copy()
    previous["datetime"] = previous.datetime.map(next_date)
    previous = previous.rename(columns={"adjusted_price": "previous_price"})
    contributions = previous.merge(prices, on=["datetime", "instrument"], how="left")
    contributions["gross_pnl"] = contributions.amount * (
        contributions.adjusted_price - contributions.previous_price
    )
    reconstructed = contributions.groupby("datetime").gross_pnl.sum().reindex(report.index, fill_value=0)
    stocks = contributions.groupby("instrument").gross_pnl.sum().sort_values(ascending=False)
    positive = stocks.clip(lower=0).sum()
    stock_table = stocks.rename("gross_pnl").reset_index()
    stock_table["positive_pnl_share"] = stocks.clip(lower=0).to_numpy() / positive
    stock_table["model"] = model
    position_value = positions.amount * positions.adjusted_price
    positions["calculated_weight"] = position_value / positions.datetime.map(report.account)
    stock_value = position_value.groupby(positions.datetime).sum().reindex(report.index, fill_value=0)
    bought = trades[(trades.direction == "buy") & (trades.deal_value > 0)]
    lots = bought.deal_amount * bought.factor / EXCHANGE["trade_unit"]
    best_days = report.net_return.nlargest(5)
    diagnostics = {
        "gross_return_reconstruction_max_error": float((reconstructed / previous_account - report["return"]).abs().max()),
        "missing_contribution_prices": int(contributions.adjusted_price.isna().sum()),
        "portfolio_value_max_error": float((stock_value + report.cash - report.account).abs().max()),
        "weight_max_error": float((positions.calculated_weight - positions.weight).abs().max()),
        "buy_lot_max_fractional_error": float((lots - lots.round()).abs().max()),
        "duplicate_position_keys": int(positions.duplicated(["datetime", "instrument"]).sum()),
        "top5_stock_positive_pnl_share": float(stocks.iloc[:5].sum() / positive),
        "top5_day_positive_return_share": float(best_days.sum() / report.net_return.clip(lower=0).sum()),
        "net_return_excluding_top5_days": float((1 + report.net_return.drop(best_days.index)).prod() - 1),
    }
    return report, stock_table, contributions, diagnostics


def main():
    global RUN, DATA, calendar, label_end, study_dates, study_members, fields
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/qlib_bin.tar.gz")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/qlib_bin")
    parser.add_argument("--download", action="store_true", help="Download the pinned archive if absent")
    parser.add_argument("--stage", choices=["data", "baseline", "all"], default="all")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "20261027":
        raise SystemExit("Set PYTHONHASHSEED=20261027 before starting Python; see README.md.")
    RUN, DATA = args.output_dir.resolve(), args.data_dir.resolve()
    if RUN.exists():
        raise FileExistsError(f"Output already exists: {RUN}")
    prepare_data(args.archive.resolve(), DATA, args.download)
    RUN.mkdir(parents=True)
    audit_dir = RUN / "data_audit"
    audit_dir.mkdir()
    print(f"Output: {RUN}", flush=True)

    # 1. 保存固定配置和实际环境。
    save_json(RUN / "config.json", {
        "snapshot": LOCK, "segments": SEGMENTS, "model_params": MODEL_PARAMS,
        "n_drop_candidates": N_DROPS, "final_n_drop": FROZEN_N_DROP,
        "topk": TOPK, "risk_degree": 0.95, "initial_cash": INITIAL_CASH,
        "exchange": EXCHANGE, "monthly_train_years": 5, "monthly_valid_months": 6,
        "num_boost_round": NUM_BOOST_ROUND, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "historical_final_interval_already_evaluated": True,
    })
    save_json(RUN / "environment.json", {
        "qlib": qlib.__version__, "lightgbm": lgb.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "python": sys.version,
        "script_sha256": sha256(Path(__file__)),
        "packages": dict(sorted((d.metadata["Name"], d.version)
                               for d in importlib.metadata.distributions() if d.metadata["Name"])),
    })

    # 2. 数据结构、字段关系、历史成员和标签时点。
    calendar = pd.DatetimeIndex(pd.read_csv(DATA / "calendars/day.txt", header=None)[0])
    members = pd.read_csv(
        DATA / "instruments/csi300.txt", sep="\t",
        names=["symbol", "start", "end"], parse_dates=["start", "end"],
    )
    study_start, study_end = pd.Timestamp("2008-01-01"), pd.Timestamp("2020-07-31")
    study_dates = calendar[(calendar >= study_start) & (calendar <= study_end)]
    study_members = members[(members.end >= study_start) & (members.start <= study_end)].copy()
    symbols = sorted(study_members.symbol.unique())
    fields = ["open", "high", "low", "close", "adjclose", "factor",
              "volume", "amount", "vwap", "change"]
    benchmark_close = read_field("SH000300", "close")
    benchmark_close.to_csv(audit_dir / "SH000300_close.csv")
    print("研究期股票数量：", len(symbols))
    example = read_stock("SH600000").loc["2017-01-03":"2017-01-06"].copy()
    example["raw_close"] = example.close / example.factor
    example["amount_ratio"] = example.volume * example.vwap / (example.amount * 10)
    example["adjclose_scale"] = example.adjclose / example.close

    records = []
    daily_member_count = pd.Series(0, index=study_dates, dtype="int64")
    daily_available_count = pd.Series(0, index=study_dates, dtype="int64")
    for number, symbol in enumerate(symbols, 1):
        stock = read_stock(symbol).reindex(study_dates)
        member_dates = member_dates_for(symbol)
        member = stock.reindex(member_dates)
        complete = stock.dropna(subset=fields)
        daily_member_count.loc[member_dates] += 1
        daily_available_count.loc[member_dates] += member.close.notna().astype("int64").to_numpy()
        price_present = member[["open", "high", "low", "close"]].notna().sum(axis=1)
        amount_error = (complete.volume * complete.vwap / (complete.amount * 10) - 1).abs()
        raw_close = stock.close / stock.factor
        change_error = (stock.change - raw_close.ffill().pct_change(fill_method=None)).abs()
        scale = (stock.adjclose / stock.close).dropna()
        scale_error = (scale / scale.median() - 1).abs()
        records.append({
            "symbol": symbol, "member_days": len(member),
            "member_close_missing_days": int(member.close.isna().sum()),
            "member_partial_price_days": int(((price_present > 0) & (price_present < 4)).sum()),
            "nonpositive_price_rows": int((complete[["open", "high", "low", "close", "vwap"]] <= 0).any(axis=1).sum()),
            "negative_volume_or_amount_rows": int(((complete.volume < 0) | (complete.amount < 0)).sum()),
            "nonpositive_factor_rows": int((complete.factor <= 0).sum()),
            "ohlc_violation_rows": int((
                (complete.high < complete[["open", "close", "low"]].max(axis=1)) |
                (complete.low > complete[["open", "close", "high"]].min(axis=1))
            ).sum()),
            "amount_relation_bad_rows": int((amount_error > 1e-4).sum()),
            "amount_relation_max_error": float(amount_error.max()),
            "change_relation_bad_rows": int((change_error > 1e-5).sum()),
            "change_relation_max_error": float(change_error.max()),
            "adjclose_scale_bad": bool((scale_error > 1e-4).any()),
            "adjclose_scale_max_deviation": float(scale_error.max()),
        })
        if number % 100 == 0:
            print(f"已检查 {number}/{len(symbols)} 只股票")
    stock_audit = pd.DataFrame(records)
    daily_coverage = pd.DataFrame({
        "member_count": daily_member_count, "close_available_count": daily_available_count,
    })
    daily_coverage["close_missing_count"] = daily_coverage.member_count - daily_coverage.close_available_count
    daily_coverage["close_available_rate"] = daily_coverage.close_available_count / daily_coverage.member_count
    daily_coverage.index.name = "date"
    stock_audit.to_csv(audit_dir / "stock_field_audit.csv", index=False)
    daily_coverage.to_csv(audit_dir / "daily_coverage.csv")
    critical = ["member_partial_price_days", "nonpositive_price_rows",
                "negative_volume_or_amount_rows", "nonpositive_factor_rows", "ohlc_violation_rows"]
    require((stock_audit[critical].sum() == 0).all(), "Invalid OHLCV data; inspect data_audit.")
    require(calendar.is_unique and calendar.is_monotonic_increasing, "Invalid trading calendar.")
    require(benchmark_close.reindex(study_dates).notna().all(), "Benchmark has missing prices.")

    label_end = pd.Series(calendar[2:].to_numpy(), index=calendar[:-2])
    sample = read_field("SH600000", "close").loc["2017-01-03":"2017-01-12"].to_frame()
    sample["trade_date_t+1"] = pd.Series(sample.index, index=sample.index).shift(-1)
    sample["label_known_t+2"] = pd.Series(sample.index, index=sample.index).shift(-2)
    sample["label"] = sample.close.shift(-2) / sample.close.shift(-1) - 1
    sample.to_csv(audit_dir / "label_example.csv")
    boundary_rows = []
    for name, first, last in [
        ("train", "2008-01-01", "2014-12-31"),
        ("valid", "2015-01-01", "2016-12-31"),
        ("development", "2017-01-01", "2018-12-31"),
        ("final", "2019-01-01", "2020-07-31"),
    ]:
        dates = calendar[(calendar >= pd.Timestamp(first)) & (calendar <= pd.Timestamp(last))]
        boundary_rows.append({
            "segment": name, "first_feature": dates[0], "last_feature": dates[-3],
            "last_trade_for_label": dates[-2], "last_label_maturity": dates[-1],
        })
    boundaries = pd.DataFrame(boundary_rows)
    boundaries.to_csv(audit_dir / "label_boundaries.csv", index=False)
    if args.stage == "data":
        print("Data and timing audit completed.")
        return

    # 3. 固定基线：学习、早停、预测、保存模型。
    qlib.init(provider_uri=str(DATA), region=REG_CN, kernels=1)
    fixed_dir = RUN / "fixed_training"
    fixed_dir.mkdir()
    fixed_handler = Alpha158(
        instruments="csi300", start_time=SEGMENTS["train"][0],
        end_time=SEGMENTS["development"][1],
        fit_start_time=SEGMENTS["train"][0], fit_end_time=SEGMENTS["train"][1],
        infer_processors=INFER_PROCESSORS, learn_processors=LEARN_PROCESSORS,
    )
    fixed_dataset = DatasetH(handler=fixed_handler, segments=SEGMENTS)
    train = fixed_dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid = fixed_dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    development = fixed_dataset.prepare("development", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

    fixed_model, fixed_history = fit_model(train, valid)
    fixed_normalizer = fixed_handler.infer_processors[0]
    fixed_dev_predictions = predict_frame(fixed_model, development)
    fixed_dev_ic = save_prediction_tables(fixed_dev_predictions, fixed_dir, "development_predictions.csv.gz")
    (fixed_dir / "lightgbm_model.txt").write_text(
        fixed_model.model_to_string(num_iteration=fixed_model.best_iteration), encoding="utf-8"
    )
    np.savez(fixed_dir / "normalization.npz",
             median=fixed_normalizer.mean_train, scale=fixed_normalizer.std_train)
    history_rows = [
        {"dataset": dataset, "metric": metric, "iteration": i, "value": value}
        for dataset, metrics_dict in fixed_history.items()
        for metric, values in metrics_dict.items()
        for i, value in enumerate(values, 1)
    ]
    pd.DataFrame(history_rows).to_csv(fixed_dir / "training_history.csv", index=False)
    importance = pd.DataFrame({
        "feature": fixed_model.feature_name(),
        "split_importance": fixed_model.feature_importance("split"),
        "gain_importance": fixed_model.feature_importance("gain"),
    }).sort_values("gain_importance", ascending=False)
    importance.to_csv(fixed_dir / "feature_importance.csv", index=False)
    fixed_ic_summary = pd.DataFrame({
        "mean": fixed_dev_ic[["IC", "RankIC"]].mean(),
        "std": fixed_dev_ic[["IC", "RankIC"]].std(ddof=1),
    })
    fixed_ic_summary["nonannualized_ir"] = fixed_ic_summary["mean"] / fixed_ic_summary["std"]
    save_json(fixed_dir / "summary.json", {
        "best_iteration": fixed_model.best_iteration,
        "best_valid_l2": fixed_model.best_score["valid"]["l2"],
        "prediction_rows": len(fixed_dev_predictions),
        "paired_rows": int(fixed_dev_ic.sample_count.sum()),
    })
    save_hashes(fixed_dir)
    del train, valid, development, fixed_dataset, fixed_handler
    gc.collect()
    if args.stage == "baseline":
        print("Fixed baseline completed.")
        return

    # 4. 开发段换股对照，核对历史选择5。
    fixed_dev_backtest = run_backtest(
        fixed_dev_predictions, fixed_dev_ic, "2017-01-04", "2018-12-28",
        N_DROPS, RUN / "development_fixed",
    )
    development_ranking = fixed_dev_backtest["summary"].query("period == 'all'").sort_values(
        ["information_ratio", "mean_two_way_turnover", "n_drop"],
        ascending=[False, True, True],
    )
    recomputed_dev_choice = int(development_ranking.iloc[0].n_drop)
    require(recomputed_dev_choice == FROZEN_N_DROP,
            "Development choice differs from frozen 5; inspect results before continuing.")
    print("本次开发段排序第一：", recomputed_dev_choice)
    print("历史最终规则冻结的换股数：", FROZEN_N_DROP)

    # 5. 开发段24个月滚动训练；复用同一回测函数。
    raw_dev = load_raw_alpha("2011-07-01", "2018-12-26")
    dev_prediction_dates = fixed_dev_predictions.index.get_level_values("datetime").unique()
    monthly_dev_predictions, monthly_dev_ic, monthly_dev_windows = monthly_retrain(
        raw_dev, pd.period_range("2017-01", "2018-12", freq="M"),
        dev_prediction_dates, RUN / "monthly_development_predictions", "development_predictions.csv.gz",
    )
    monthly_dev_backtest = run_backtest(
        monthly_dev_predictions, monthly_dev_ic, "2017-01-04", "2018-12-28",
        N_DROPS, RUN / "development_monthly",
    )
    del raw_dev
    gc.collect()

    # 6. 最终区间：原固定模型与19个月滚动重训。
    final_prediction_dir = RUN / "final_predictions"
    final_prediction_dir.mkdir()
    final_fixed_dir = final_prediction_dir / "fixed"
    final_fixed_dir.mkdir()
    save_json(final_prediction_dir / "frozen_config.json", {
        "written_before_final_evaluation": datetime.now().isoformat(),
        "n_drop": FROZEN_N_DROP, "model_params": MODEL_PARAMS,
        "fixed_model_sha256": sha256(fixed_dir / "lightgbm_model.txt"),
        "monthly_train_years": 5, "monthly_valid_months": 6,
        "early_stopping": EARLY_STOPPING_ROUNDS, "num_boost_round": NUM_BOOST_ROUND,
        "snapshot_sha256": LOCK["archive_sha256"], "exchange": EXCHANGE,
        "boundary": "首日信号次日交易；末两日不生成新信号；期末不清仓",
    })
    raw_final = load_raw_alpha("2013-07-01", "2020-07-29")
    final_dates = calendar[
        (calendar >= pd.Timestamp("2019-01-01")) & (calendar <= pd.Timestamp("2020-07-31"))
    ][:-2]
    raw_final_dates = raw_final.index.get_level_values("datetime")
    fixed_final_frame = Fillna("feature")(fixed_normalizer(
        raw_final.loc[raw_final_dates.isin(final_dates)].copy()
    ))
    fixed_final_predictions = predict_frame(fixed_model, fixed_final_frame)
    fixed_final_ic = save_prediction_tables(
        fixed_final_predictions, final_fixed_dir, "holdout_predictions.csv.gz",
    )
    save_hashes(final_fixed_dir)
    monthly_final_predictions, monthly_final_ic, monthly_final_windows = monthly_retrain(
        raw_final, pd.period_range("2019-01", "2020-07", freq="M"), final_dates,
        final_prediction_dir / "monthly", "holdout_predictions.csv.gz",
    )
    sample_diagnostics = pd.DataFrame([{
        "same_index": fixed_final_predictions.index.equals(monthly_final_predictions.index),
        "raw_label_max_error": float((fixed_final_predictions.label - monthly_final_predictions.label).abs().max()),
        "fixed_rows": len(fixed_final_predictions), "monthly_rows": len(monthly_final_predictions),
        "fixed_duplicated_keys": int(fixed_final_predictions.index.duplicated().sum()),
        "monthly_duplicated_keys": int(monthly_final_predictions.index.duplicated().sum()),
    }])
    sample_diagnostics.to_csv(final_prediction_dir / "sample_diagnostics.csv", index=False)
    pd.testing.assert_index_equal(fixed_final_predictions.index, monthly_final_predictions.index)
    np.testing.assert_allclose(fixed_final_predictions.label, monthly_final_predictions.label,
                               rtol=1e-6, atol=1e-9, equal_nan=True)
    del raw_final, fixed_final_frame
    gc.collect()

    fixed_final_backtest = run_backtest(
        fixed_final_predictions, fixed_final_ic, "2019-01-03", "2020-07-31",
        [FROZEN_N_DROP], RUN / "final_fixed",
    )
    monthly_final_backtest = run_backtest(
        monthly_final_predictions, monthly_final_ic, "2019-01-03", "2020-07-31",
        [FROZEN_N_DROP], RUN / "final_monthly",
    )

    # 7. 汇总主实验、年度指标和比较图。
    main_results = pd.concat([
        experiment_table(fixed_dev_backtest, monthly_dev_backtest, "development"),
        experiment_table(fixed_final_backtest, monthly_final_backtest, "final"),
    ], ignore_index=True)
    main_results.to_csv(RUN / "main_experiments.csv", index=False)
    yearly_results = pd.concat([
        result["summary"].assign(scope=scope, model=model)
        for scope, model, result in [
            ("development", "fixed", fixed_dev_backtest), ("development", "monthly", monthly_dev_backtest),
            ("final", "fixed", fixed_final_backtest), ("final", "monthly", monthly_final_backtest),
        ]
    ], ignore_index=True)
    yearly_results.to_csv(RUN / "period_results.csv", index=False)

    comparison_plot(fixed_dev_backtest, monthly_dev_backtest, fixed_dev_ic, monthly_dev_ic,
                    "Development: fixed vs monthly", RUN / "development_comparison.png")
    comparison_plot(fixed_final_backtest, monthly_final_backtest, fixed_final_ic, monthly_final_ic,
                    "Final reproduction: fixed vs monthly", RUN / "final_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, result in [("Fixed", fixed_dev_backtest), ("Monthly", monthly_dev_backtest)]:
        part = result["summary"].query("period == 'all'").sort_values("n_drop")
        axes[0].plot(part.n_drop, part.mean_two_way_turnover * 100, marker="o", label=name)
        axes[1].plot(part.n_drop, part.sum_cost_rate * 100, marker="o", label=name)
    for ax, label in zip(axes, ["Mean daily two-way turnover (%)", "Sum of daily cost rates (%)"]):
        ax.set(xlabel="n_drop", ylabel=label, xticks=[1, 3, 5])
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RUN / "turnover_cost.png", dpi=160)

    plt.close(fig)

    # 8. 费用敏感性：改变费用后重新计算成交与账户路径。
    cost_dir = RUN / "cost_stress"
    cost_dir.mkdir()
    cost_rows = []
    with tempfile.TemporaryDirectory(prefix="qlib_cost_") as temporary:
        with zipfile.ZipFile(cost_dir / "details.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name, predictions, daily, original, original_folder in [
                ("fixed", fixed_final_predictions, fixed_final_ic, fixed_final_backtest, RUN / "final_fixed"),
                ("monthly", monthly_final_predictions, monthly_final_ic, monthly_final_backtest, RUN / "final_monthly"),
            ]:
                for multiplier in [1.0, 1.5, 2.0]:
                    result, folder = original, original_folder
                    if multiplier != 1:
                        folder = Path(temporary) / f"{name}_cost_{multiplier:g}"
                        result = run_backtest(
                            predictions, daily, "2019-01-03", "2020-07-31",
                            [FROZEN_N_DROP], folder, multiplier=multiplier,
                        )
                    cost_rows.extend(result["summary"].assign(
                        model=name, cost_multiplier=multiplier,
                    ).to_dict("records"))
                    for path in sorted(folder.iterdir()):
                        archive.write(path, f"{name}_cost_{multiplier:g}/{path.name}")
    cost_summary = pd.DataFrame(cost_rows)
    cost_summary.to_csv(cost_dir / "summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, part in cost_summary.query("period == 'all'").groupby("model"):
        ax.plot(part.cost_multiplier, part.net_cagr * 100, marker="o", label=name)
    ax.set(xlabel="Cost multiplier", ylabel="Net CAGR (%)", xticks=[1, 1.5, 2])
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RUN / "cost_sensitivity.png", dpi=160)

    plt.close(fig)

    # 9. 收益集中度和条件区块区间。
    analysis_dir = RUN / "contribution_analysis"
    analysis_dir.mkdir()
    diagnostics, final_reports = {}, {}
    with zipfile.ZipFile(analysis_dir / "details.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, result in [("fixed", fixed_final_backtest), ("monthly", monthly_final_backtest)]:
            report, stocks, contributions, findings = contribution_analysis(result, name)
            final_reports[name] = report
            diagnostics[name] = findings
            archive.writestr(f"{name}_stocks.csv", stocks.to_csv(index=False))
            archive.writestr(f"{name}_daily_stock_contributions.csv", contributions.to_csv(index=False))
            require(findings["missing_contribution_prices"] == 0 and
                    findings["gross_return_reconstruction_max_error"] < 1e-9,
                    "Stock contribution reconciliation failed.")
    difference = (final_reports["monthly"].net_return - final_reports["fixed"].net_return).to_numpy()
    block_length, repetitions = 20, 5000
    rng = np.random.default_rng(20261027)
    starts = rng.integers(
        0, len(difference) - block_length + 1,
        size=(repetitions, int(np.ceil(len(difference) / block_length))),
    )
    indexes = (starts[:, :, None] + np.arange(block_length)).reshape(repetitions, -1)[:, :len(difference)]
    means = difference[indexes].mean(axis=1)
    interval = {
        "daily_mean_difference": float(difference.mean()),
        "percentile_95_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "block_days": block_length, "repetitions": repetitions, "seed": 20261027,
        "conditional_on_existing_predictions_and_strategy": True,
        "includes_training_or_selection_uncertainty": False,
    }
    save_json(analysis_dir / "findings.json", {"contributions": diagnostics, "conditional_block_interval": interval})

    stability_rows = []
    for model, result in [("fixed", fixed_dev_backtest), ("monthly", monthly_dev_backtest)]:
        for n_drop in N_DROPS:
            positions = result["positions"].query("n_drop == @n_drop")
            groups = positions.groupby("datetime", sort=True)
            sets = [set(part.instrument) for date, part in groups]
            retained = [len(previous & current) / len(previous)
                        for previous, current in zip(sets[:-1], sets[1:])]
            stability_rows.append({
                "model": model, "n_drop": n_drop,
                "mean_current_holding_age": float(positions.holding_days.mean()),
                "mean_next_day_retention": float(np.mean(retained)),
            })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(RUN / "holding_stability.csv", index=False)

    # 10. 报告、历史指标比较和产物哈希。
    comparison = check_reference(main_results, cost_summary)
    report_text = f"""# Qlib多因子选股：本次运行报告

运行时间：{datetime.now().isoformat()}

策略：历史沪深300成员，Alpha158+LightGBM，TopkDropout约50只，
95%目标风险资金比例，冻结n_drop=5，每月五年训练+六个月验证。

固定模型最佳轮数：{fixed_model.best_iteration}。
本次固定模型开发段换股排序第一：{recomputed_dev_choice}；历史最终规则仍为5。
历史指标核对通过：{comparison['matched']}。差异明细见reproduction_check.json。

## 主实验

```csv
{main_results.to_csv(index=False)}```

## 最终段费用压力

```csv
{cost_summary.query("period == 'all'").to_csv(index=False)}```

## 条件区块分析

日收益差均值：{interval['daily_mean_difference']:.8f}；
95%区间：{interval['percentile_95_interval']}。
条件于既有预测与交易路径，不包含训练及选参不确定性，CAGR差需要单独估计。

该最终区间已经在历史研究中被评价，本次为复现。
月度方案同时改变更新频率和训练窗口，不能把收益差单独归因于更新频率。
单一社区历史快照、简化收盘成交、缺少冲击和容量模型限制实盘解释。
"""
    (RUN / "本次研究报告.md").write_text(report_text, encoding="utf-8")
    save_hashes(cost_dir)
    save_hashes(analysis_dir)
    all_artifacts = {
        p.relative_to(RUN).as_posix(): sha256(p)
        for p in sorted(RUN.rglob("*")) if p.is_file()
    }
    save_json(RUN / "artifacts_sha256.json", all_artifacts)


if __name__ == "__main__":
    main()
