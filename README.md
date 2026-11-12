# Qlib多因子选股：月度滚动重训与换手控制

使用历史沪深300股票池、Alpha158量价特征、LightGBM和TopkDropout，研究两个问题：

1. 相比训练一次后保持参数固定，每月滚动重训能否改善后续预测和组合表现？
2. 减少计划换股数量节省的费用，能否弥补持仓更新变慢造成的收益损失？

**单脚本执行完整主项目，仓库共7个文件。** 数据下载、时点检查、固定训练、月度重训、回测、费用压力、收益集中度及报告均由 `run.py` 完成。运行结果自动生成在独立目录中。

## 研究改进与贡献

本项目基于Qlib官方示例开展受控实验，研究贡献在实验设计、实现及证据核对。Alpha158、LightGBM、滚动窗口和TopkDropout均有既有方法来源。

- **A：月度滚动重训。** 每月首个交易日前、上一交易日收盘后，使用五年训练段及随后六个月验证段从头拟合。每月单独拟合标准化参数，剔除跨训练/验证边界和当时尚未实现的标签，按原日期与股票拼接预测。
- **B：换手控制实验。** 同一份固定模型分数比较 `n_drop ∈ {1,3,5}`，按开发段扣费后IR选择；平局时依次优先较低换手、较小换股数。历史结果选择5，保留降低换股数效果较差的证据。
- **统一评价与核对。** 固定数据、候选池、费用、资金规模及成交规则；核对信号日期、账户收益、真实成交费用及逐股毛损益。用1/1.5/2倍费用重跑组合，补充收益集中度和条件区块区间。
- **可追溯复现。** 下载时校验完整归档SHA-256，使用前逐文件核对本研究涉及的行情及日历、股票池；保存窗口、模型、参数、版本、预测、订单和产物哈希。

“固定模型 vs 月度模型”同时改变了历史样本范围、验证窗口和更新方式，结论适用于整套方案。目前尚未开展每日重训对照、滚动窗与扩展窗对照。

## 实验结果

下表为原研究的固定参考值。精简代码的本次运行结果由程序独立计算，再与 `reference_results.json` 比较。

| 区间 | 模型 | 日均Rank IC | 扣费后年化 | 主动IR | 最大回撤幅度 |
|---|---|---:|---:|---:|---:|
| 开发：2017—2018 | 固定模型 | 0.06419 | 9.36% | 2.173 | 26.89% |
| 开发：2017—2018 | 月度滚动重训 | 0.06899 | 14.29% | 2.836 | 21.03% |
| 最终：2019—2020年7月 | 固定模型 | 0.04024 | 40.34% | 0.527 | 18.00% |
| 最终：2019—2020年7月 | 月度滚动重训 | 0.03728 | 46.95% | 1.101 | 16.43% |

以上均使用 `n_drop=5`。B与B0一致，AB与A一致，程序明确保留这种对应关系。降低换股数节省了费用，开发段的毛收益损失更大。月度方案在最终段的平均Rank IC更低，组合收益优势主要来自2020年前七个月。

最终段日收益差“月度−固定”的条件区块重采样95%区间约为 **[−1.20，4.73]个基点/日**，包含零。该区间条件于已有预测和交易路径，未包含重新训练及参数选择的不确定性，也未估计CAGR差的区间。

2019—2020年7月已在原研究中被评价，本仓库执行的是历史复现。后续在同一区间修改方案应标为扩展分析。

## 文件

```text
run.py                  完整执行入口，配置、函数和顺序主流程
README.md               研究设计、运行方式、结果和局限
requirements.txt        运行依赖及间接依赖版本
data_snapshot.json      固定数据版本、下载地址和SHA-256
reference_results.json  历史参考指标与来源文件哈希
.gitignore              排除行情、环境、缓存及生成结果
LICENSE                 本仓库代码使用MIT许可
```

学习顺序：先读本文的数据与实验设置，再从 `run.py` 的 `main()` 沿步骤阅读。`monthly_windows()` 定义窗口，`fit_model()` 训练与早停，`run_backtest()` 把分数转换成实际成交，`metrics()` 计算评价指标。

## 安装

参考环境：**CPython 3.11.9、Windows x64、CPU**。建议16GB以上内存，为归档、解压行情及结果留出至少10GB空间。首次全流程涉及1个固定模型、43个月度模型和12组组合回测，耗时取决于硬件和磁盘。

Windows请使用包含 `pip` 和 `venv` 的标准Python 3.11 x64安装包，并安装 [Microsoft Visual C++ x64运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe)。LightGBM需要其中的OpenMP运行库 `vcomp140.dll`；缺失时可能报 `lib_lightgbm.dll` 或其依赖无法加载。Python嵌入式精简发行包通常不包含 `venv`。

```powershell
git clone https://github.com/Burning-Lyu/qlib-rolling-research.git
cd qlib-rolling-research
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONHASHSEED = "20261027"
```

`PYTHONHASHSEED` 必须在启动Python前设置，以稳定策略中的集合遍历；模型随机种子也固定为20261027。程序检查环境变量，缺失时给出提示。

Linux/macOS的相应命令：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PYTHONHASHSEED=20261027
```

下文的解释器路径在Linux/macOS替换为 `.venv/bin/python`。跨平台数值一致性仍需实测；依赖版本以当前参考环境为准，完整实际包版本写入每次运行的 `environment.json`。Qlib自身的依赖会引入Jupyter等软件包，执行本项目只需Python命令。

## 从零运行

所有命令在仓库目录执行。首次运行从数据提供方下载固定归档，约568MB：

```powershell
.\.venv\Scripts\python.exe -u -B run.py --download --output-dir results\first_run
```

程序依次执行：

1. 校验完整快照、核对研究涉及的解压文件、检查行情字段和历史股票池。
2. 检查标签边界，训练固定模型并生成开发段预测。
3. 固定分数下比较换股5/1/3，检查选择结果仍为历史冻结值5。
4. 完成2017—2018的24次月度重训及同条件回测。
5. 写入冻结配置，完成2019—2020年7月的固定预测、19次月度重训与两组回测。
6. 运行费用压力、收益贡献和持仓稳定性检查，生成报告与图表。
7. 重算并核对历史参考指标，保存完整产物哈希。

数据已存在时省略 `--download`。已有输出目录会被拒绝覆盖，更换 `--output-dir` 后可重新运行。

只检查数据，或先运行一次固定训练：

```powershell
.\.venv\Scripts\python.exe -u -B run.py --download --stage data --output-dir results\data_check
.\.venv\Scripts\python.exe -u -B run.py --stage baseline --output-dir results\baseline_check
```

这两个短入口分别停在数据检查和固定模型训练结束处。`all` 默认从头执行完整流程，当前入口未实现中断续跑。

已有同版本归档和解压目录时可复用，避免重复下载：

```powershell
.\.venv\Scripts\python.exe -u -B run.py --archive "D:\market_data\qlib_bin.tar.gz" --data-dir "D:\market_data\qlib_bin" --output-dir results\reproduce
```

归档及解压目录都需匹配清单；解压目录应直接包含 `calendars/`、`instruments/` 和 `features/`。归档一致且解压目录不存在时会自动解压。校验失败时停止执行，请核对文件完整性和数据版本。

## 查看结果

在指定输出目录查看：

| 产物 | 内容 |
|---|---|
| `本次研究报告.md` | 根据本次计算生成的主结果、成本压力及条件区间 |
| `main_experiments.csv`、`period_results.csv` | 四组主实验及年度指标 |
| `reproduction_check.json` | 原参考指标与当前结果是否在 `rtol=1e-8, atol=1e-10` 内一致 |
| `development_comparison.png`、`final_comparison.png` | 净值和月度Rank IC |
| `turnover_cost.png`、`cost_sensitivity.png` | 换手费用和成本敏感性 |
| `monthly_development_predictions/`、`final_predictions/monthly/` | 月度窗口、IC、预测和打包模型 |
| `development_fixed/`、`final_fixed/`等回测目录 | 日报、订单、持仓、信号日期、账户诊断 |
| `data_audit/` | 字段核验、缺失覆盖、标签时点示例 |
| `contribution_analysis/`、`holding_stability.csv` | 收益来源、条件区间、持仓稳定性 |
| `config.json`、`environment.json`、`artifacts_sha256.json` | 参数、实际依赖、代码与产物校验 |

`reproduction_check.json` 出现差异时，应先核对Python、依赖、种子和平台，再查看逐项差异；程序保留实际计算结果。数据错误、账户不一致、未来信号或开发选择变化会中止流程。

## 数据与时点

数据来自社区项目 [chenditc/investment_data](https://github.com/chenditc/investment_data)，固定为 [2026-10-19 release](https://github.com/chenditc/investment_data/releases/tag/2026-10-19)，来源提交为 `b8c129b4d9b838f050eac8b1135111b3b79fc894`。快照SHA-256：

```text
84148086a1137d209b801079ebc1bb7e0d7229b7028f55982c0502766bd56e99
```

每日候选采用当日历史 `csi300` 成员；基准为 `SH000300`。价格为社区处理后的复权及标准化研究字段。原检查覆盖727只历史成员股，存在停牌式整行缺失，缺失价格保持缺失。标准化只拟合训练段，特征处理后填零；缺标签只在学习及评价时过滤，预测股票名单保留。

| 阶段 | 特征日期 | 用途 |
|---|---|---|
| 固定训练 | 2008-01-02—2014-12-29 | 拟合树模型与标准化 |
| 固定验证 | 2015-01-05—2016-12-28 | 决定训练轮数 |
| 开发预测 | 2017-01-03—2018-12-26 | 检查流程，比较换股数 |
| 最终预测 | 2019-01-02—2020-07-29 | 按冻结方案统一评价 |

标签为 `close[t+2] / close[t+1] - 1`，其中t为交易日。t日收盘后生成分数，t+1日收盘按规则交易，t+2日收盘后标签完整实现。训练及验证末端按交易日剔除未成熟标签；开发回测至2018-12-28，最终回测至2020-07-31，末日仅估值、期末保留持仓。

月度方案每次使用五年训练＋随后六个月验证。2017年1月实际训练范围为2011-07-01—2016-06-28、验证为2016-07-01—2016-12-28；模型在2016-12-30收盘后更新，用于1月每日预测。下一月窗口向前移动一个月。当月模型参数固定，每天输入当天可用特征。

Alpha158内的5/10/20/30/60交易日等特征窗口，控制量价指标的回看长度；五年窗口控制模型拟合所用的历史样本。

## 模型、策略与评价口径

- 模型采用官方LightGBM示例超参数，线程数8，最多1000轮；验证集L2连续50轮未改善时停止，使用验证误差最低的轮数预测。标签经过每日横截面排名标准化，模型输出用于排序。
- 目标约50只多头股票，目标风险资金比例95%；原生TopkDropout规则保留持仓，实际成交受现金、停牌及价格限制影响。
- 每组初始资金1亿元，收盘价成交，买入费率0.05%、卖出费率0.15%、最低费用5元，简化涨跌幅限制9.5%，交易单位100股。最终两组各从现金开始；年度分解沿用连续账户。
- 日净收益=`return − cost`，净值按日复利；年化采用252交易日。主动IR=`mean(日净收益−基准日收益) / std(ddof=1) × sqrt(252)`。
- 最大回撤按组合净值相对历史峰值计算，峰值包含初始净值1。CSV中保存负回撤，本文展示正的跌幅。
- IC与Rank IC分别为每天股票横截面预测和原始标签的Pearson、Spearman相关，再对有效日期等权平均。
- 成本压力将买入、卖出、最低费用一起放大；费用改变现金和成交路径，因此每组重新回测。

## 复现状态与局限

2026-11-12已在上述Windows参考环境完整执行 `--stage all`，复用固定归档和已有解压目录，约耗时25分钟，不含下载。完成1个固定模型、43个月度模型及12组回测；10组参考结果共70项指标均在 `rtol=1e-8, atol=1e-10` 内匹配，`reproduction_check.json` 为 `matched: true`。训练窗口、账户与信号日期检查通过，83个产物哈希及模型压缩包完整性均已核对。

另在同机新建Python 3.11.9隔离环境，完成锁定依赖安装、`pip check` 和 `run.py --help` 启动验证。隔离环境通过 `virtualenv` 创建，并为LightGBM补齐经Microsoft签名验证的本地OpenMP运行库。完整研究计算在参考环境执行；尚未在第二台机器或Linux/macOS独立完成全流程复现。

研究使用单一社区历史快照，历史修订及上游数据授权存在局限。固定收盘成交和统一涨跌幅阈值为简化假设，缺少冲击、排队及容量模型；1亿元为演示账户设置。样本中的收益差不能保证未来盈利。

本仓库发布代码、数据指针和汇总指标。行情请从提供方获取，使用前遵守其及上游供应商的条件；MIT许可覆盖本仓库代码，未授予行情再分发权。依赖组件各自遵循其许可。

## 开源来源

- [Microsoft Qlib v0.9.7](https://github.com/microsoft/qlib/tree/v0.9.7)：数据接口、Alpha158、处理器、TopkDropout及回测引擎，MIT许可。
- [Qlib LightGBM Alpha158配置](https://github.com/microsoft/qlib/blob/v0.9.7/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)：基线配置起点。
- [LightGBM](https://github.com/microsoft/LightGBM)：梯度提升树实现，MIT许可。
- [investment_data](https://github.com/chenditc/investment_data/tree/b8c129b4d9b838f050eac8b1135111b3b79fc894)：固定行情来源，仓库许可与上游行情授权需分别核对。

项目名称和贡献表述限定于多因子选股的受控研究；当前范围未包含神经网络训练、自动因子挖掘、实盘交易系统或全新算法。
