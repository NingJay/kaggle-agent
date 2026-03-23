# BirdCLEF 2026 System Plan

## Goal

在 `kaggle_agent` 内构建一个可持续运行的 Kaggle 自研究系统，而不是一次性脚本集合。

## Scope

- 管理 BirdCLEF 2026 的 ideas / experiments / submissions / runtime state。
- 自动生成知识库、dashboard、agent briefs。
- 以实验工件为核心，支持外部训练脚本接入。
- 生成 Kaggle code competition 的 notebook submission scaffold。

## Non-Goals

- 不在这个仓里实现完整模型训练代码。
- 不默认接管真实 Kaggle 在线提交。
- 不耦合单一 LLM 提供商。

## Key Decisions

- 使用标准库 Python + JSON/TOML，避免基建依赖。
- 让训练脚本通过 stdout 协议回传分数，而不是把 harness 绑死到某个训练框架。
- 将 BirdCLEF 2026 的领域假设固化进默认 backlog，但保留 `workspace.toml` 可覆盖能力。
- 报告输出三份：Markdown、CSV、HTML。

## MVP Loop

1. `init`
2. `doctor`
3. 为实验补命令并切到 `pending`
4. `run-next`
5. `refresh`
6. `plan-submission`

## Expansion Paths

- 接入真实 agent command 模板，自动消费 `prompts/*.md`
- 增加数据审计工具和 leaderboard probing 管线
- 增加定时守护进程和 webhook/消息通知

