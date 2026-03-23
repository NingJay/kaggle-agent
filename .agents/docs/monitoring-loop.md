# Kaggle Monitoring Loop

用于本地 BirdCLEF 2026 实验的长期监控，不直接控制集群，只维护知识工件和实验状态。

## 目标

- 每 `10` 分钟更新当前实验状态。
- 每 `30` 分钟重建 Markdown / CSV / HTML 报告。
- 每 `6` 小时评估是否需要生成新的 submission plan。

## 规则

- 只监控 `state/experiments.json` 中 `status=pending|running|failed|completed` 的实验。
- 不要擅自杀掉外部训练进程，除非用户在当前线程明确要求。
- 不要擅自推送 Kaggle dataset / kernel。
- 所有状态变更后，都要调用 `python -m kaggle_agent.cli refresh`。

## 标准循环

1. 查看 `state/runtime.json` 和 `state/experiments.json`。
2. 如果有 `running` 实验：
   - 检查 `log_path` 是否更新。
   - 如果记录了 `pid`，通过 `python -m kaggle_agent.cli poll-running` 收尾已结束进程。
   - 把关键进展写进 `knowledge/master_overview.md`。
3. 如果没有 `running` 实验，但存在满足依赖且已有命令的 `pending` 实验：
   - 由调度器决定是否执行 `run-next`。
4. 如果最近 6 小时没有新的 submission plan，且存在更强的 completed scored experiment：
   - 提醒或生成新的 submission scaffold。
5. 循环等待。

## 失败分级

- 小失败：命令不存在、脚本 import 错误、输出缺少 `KAGGLE_AGENT_RESULT`。可以修命令或脚本后重试。
- 大失败：环境损坏、数据缺失、评估泄漏、Kaggle notebook 提交失败。先记录，再请求人类判断。
