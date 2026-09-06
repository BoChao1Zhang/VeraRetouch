"""学生 / 特权教师 prompt 构造（EPR-052 骨架）。

学生 prompt = 现行 SFT/评测口径（`data.q3vl_text.prompt_text` + `data.cot_text.instruction_for` 的 70/20/10 逐样本指令）。
教师 prompt = 学生 prompt + GT CoT 作为前缀上下文（**格式占位，待调研确认**；见 rl/README.md）。
"""
from __future__ import annotations

from veraretouch_sprf.data import cot_text as C
from veraretouch_sprf.data import q3vl_text as T

TEACHER_PREFIX_HEADER = "\n\nReference grade (privileged context, do not copy verbatim):\n"   # 待调研确认


def student_instruction(record: dict, key: str | None = None, mode: str = "per_sample",
                        salt: str = C.INSTRUCTION_SALT) -> tuple[str, str]:
    """-> (指令文本, tier)。mode=per_sample 走 sha1 三档抽样；mode=fixed 用 C.INSTRUCTION（历史口径，已作废，仅供对照）。"""
    if mode == "per_sample":
        return C.instruction_for(record, key, salt)
    if mode == "fixed":
        return C.INSTRUCTION, "fixed"
    raise ValueError(mode)


def student_prompt(record: dict, key: str | None = None, mode: str = "per_sample") -> str:
    """学生看到的完整 prompt 文本（含 <|image_pad|> 占位与 assistant 头），与 dump_readout / train 同一字符串。"""
    instr, _ = student_instruction(record, key, mode)
    return T.prompt_text(instr)


def teacher_prompt(record: dict, key: str | None = None, mode: str = "per_sample") -> str:
    """教师（同权重 + 特权上下文）prompt：学生指令 + GT CoT 六段作为前缀上下文，再接 assistant 头。占位格式。"""
    instr, _ = student_instruction(record, key, mode)
    gt = C.target_text(record)
    return T.prompt_text(instr + TEACHER_PREFIX_HEADER + gt)


def gt_target(record: dict) -> str:
    """GT CoT 序列（六段 + 阶段 token），与 SFT 目标同一序列化。"""
    return C.target_text(record)
