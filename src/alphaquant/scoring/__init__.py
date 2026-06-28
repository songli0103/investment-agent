"""评分模块 —— 仅保留工具和 LLM 工具使用的确定性辅助函数。

子项目 3:`rating`、`competitive`、`risk_score` 模块已移除(现由 LLM 驱动)。
`dcf` 和 `financial_health` 保留,因为估值分析师和报告撰写代理可以在推理过程中将它们作为工具调用。
"""
from alphaquant.scoring import dcf, financial_health

__all__ = ["dcf", "financial_health"]