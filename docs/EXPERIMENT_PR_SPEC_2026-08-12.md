# 实验 PR 规范(2026-08-12 起强制)

> 每个实验 = 一个 PR。一个实验只有一份提案文档,状态在其中流转;指标先行,解读后置,
> 分析独立。总览入口:`docs/EXPERIMENT_INDEX.md`;发现合并目标:`docs/WHERE_STATE_2026-08-11.md`。

## 目录与文档

```
experiments/prs/EPR-<三位序号>_<slug>/
  PROPOSAL.md    # 唯一提案文档(见下),批准后判据冻结
  RESULT.md      # 执行 agent 写:指标变化先行,禁解读
  ANALYSIS.md    # 独立分析 agent 写:只读交付,不读实现过程
  metrics.json   # 机器可读结算
  config/        # 配置快照 + seed + commit
```

## PROPOSAL.md 强制字段(批准即冻结,改动走文末 changelog)

1. **目标指标**:要提升什么 + 当前基线值 + 基线出处(如「headline mIoU(normal-only,
   面积匹配 top-k)= 0.7909,出处 amort_P3prime_cont2 eval_final」);
2. **预注册数字**:预期提升门 / 证伪线(如「Δ≥+0.015 晋级;<+0.008 作废」);
3. **假设一句话**:成功则能说 X,失败则不能说 X(禁「评估性能」空话);
4. **输入**:数据集代号 + n + 切分(全量/探针如实标注);
5. **方法**:与基线的**唯一差异**一段(一个 PR 只改一件事;多变量须拆多个 EPR 或写成消融矩阵);
6. **判据表**:主指标 + 强制守卫列(中心先验 / 随机地板 / Δ_shuffle+Δ_const / 分层口径;禁 AUC)
   + **每个判据的运行时断言**(eval 启动时校验判据函数被调用——「定义了没接线」已发生三次);
7. **资源预算**(GPU·h)与回退预案;
8. **状态行**(文档首行):`PROPOSED → APPROVED → RUNNING → SETTLED → ANALYZED → MERGED / REJECTED`。

## RESULT.md 强制格式(实验结束立即,执行 agent)

- **第一行**:`<目标指标>: <基线> → <实测>(Δ=, p=)` ——先说指标变化,再说其他;
- 判据表逐行 PASS/FAIL(含全部守卫列);
- 口径声明(normal-only/pooled、步数匹配、n);
- **禁止写结论解读**——那是 ANALYSIS 的职责。

## ANALYSIS.md(独立分析 agent,零上下文,只读本 EPR 交付 + WHERE_STATE)

- 假设被支持/否定/不可判(与预注册数字对表);
- 失败归因与意外发现(引 per-sample 证据);
- **给 WHERE_STATE 的增量 S 条草案**(一句话 + 出处);
- 对后续 EPR 的建议。

## 角色与流转

- **主 agent**:批准 PROPOSAL(冻结判据)→ 收 RESULT → 派独立分析 → 把 S 条草案 merge 进
  WHERE_STATE(带 EPR 编号)→ 更新 EXPERIMENT_INDEX 状态与一句话结论;
- **执行 agent**(零上下文 Opus 5):按 PROPOSAL 执行,只交 RESULT + metrics + config;
- **独立分析 agent**(零上下文 Opus5):只看交付文件夹,不看实现过程与执行 agent 的对话。

## 纪律衔接

D-20 提交纪律、显存共存规则(<65GB)、满载补位、trash 禁读、AUC 禁用、
预注册不许出数后改门——全部沿用 CLAUDE.md 与 WHERE_STATE 既有条款。
历史实验已按 **EPR-H 系**回填(单文件 `experiments/prs/EPR-Hxx_<slug>/RETRO.md`,不拆三文件),
索引见 `docs/EXPERIMENT_INDEX.md`「历史实验」表;
本规范生效起的新实验(含在途的 PCH 三档 / A1 试点 / SHAPE3 补跑)一律适用。
