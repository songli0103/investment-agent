"""AnalysisFlow:包裹 AnalysisCrew 的轻量级外壳。

两步式 Flow:

1. ``run_crew`` (``@start``) — 在 :func:`asyncio.to_thread` 中调用
   :class:`AnalysisCrew.kickoff`,带每步超时,然后分发到
   :func:`parse_crew_output` 以填充下游状态字段。4 个数据代理通过工具在
   Crew 内部自行获取数据(子项目 2);Flow 不再预取数据。
2. ``synthesize_report`` (``@listen(run_crew)``) — 根据已填充的数据确定性地
   计算 3 个分析字段(竞争/风险/估值),然后从数据 + 确定性分析 +
   LLM 的 :class:`ReportWriterOutput` 组装完整的 :class:`InvestmentReport`。

子项目 3 让 3 个分析代理(竞争/风险/估值)产出结构化 Pydantic 输出。
LLM 在发出结构无效的输出(错误的字段名、口语化文本)时,会导致 CrewAI
转换器在 180 秒 Flow 超时内反复重试,阻塞前端。
我们已将这些任务还原为仅产出文本,并在 Flow 中确定性地计算这 3 项分析。
report_writer LLM 现在产出精简的 :class:`ReportWriterOutput`
(评级、置信度、持有期、催化剂、markdown);Flow 组装完整的 :class:`InvestmentReport`。
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from crewai.flow import Flow, listen, start
from pydantic import BaseModel, Field, ValidationError

from alphaquant.crews import AnalysisCrew
from alphaquant.exceptions import (
    AllDataSourcesDown,
    CrewExecutionError,
    InvalidTickerFormat,
    LLMRateLimited,
    ReportGenerationError,
)
from alphaquant.infrastructure.crew_events import get_default_bridge
from alphaquant.models.company import Company
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis, NewsItem
from alphaquant.models.report import InvestmentReport, ReportWriterOutput
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult
from alphaquant.observability import get_logger
from alphaquant.scoring.dcf import compute_dcf_value
from alphaquant.scoring.financial_health import compute as compute_financial_health

log = get_logger("alphaquant.flows.analysis_flow")

# 发送到可选进度回调的步骤 ID。顺序与 Streamlit(或其他实时 UI)调用方
# 渲染的阶段一致:
#   1. validate_ticker      — 去空格 + 大写 + 长度校验
#   2. run_crew             — CrewAI 8 代理 kickoff(耗时最长的步骤)
#   3. parse_crew_output    — 从 CrewOutput 提取数据 + writer_output
#   4. compute_analyses     — 确定性的竞争/风险/估值
#   5. assemble_report      — 构建最终 InvestmentReport
ProgressCallback = Callable[[str, str], None]

# 所有生成报告的免责声明文本。按规范使用中文。
DISCLAIMER_TEXT = (
    "本报告由 AI 自动生成，仅供参考，不构成任何投资建议。"
    "投资有风险，决策需谨慎。"
)

# §3.4:整个 Flow 的超时。子项目 3 回退验证(Task 5)显示
# MiniMax-M3 对 7 个 LLM 任务通常需要超过 300 秒(在最佳一次 AAPL 运行中,
# 仍进行中的 18 次成功 LLM 调用就撞上了 300 秒限制)。
# 将 300→600 秒扩展可恢复原始规范上限,并为真实世界延迟给 LLM 留出足够空间。
FLOW_TIMEOUT_SECONDS = 600.0


# 将 crew 任务描述映射到 AnalysisState 字段键。顺序必须与
# crews/analysis_crew.py::_TASK_TEMPLATES 中的顺序一致。
_TASK_KEYWORDS: list[str] = [
    "company_resolver",
    "market_analyst",
    "news_analyst",
    "financial_analyst",
    "competitor_analyst",
    "risk_analyst",
    "valuation_analyst",
    "report_writer",
]


class AnalysisState(BaseModel):
    """在 Flow 步骤间传递的状态。"""

    ticker: str = ""
    company: Any | None = None
    market: MarketData | None = None
    news: NewsAnalysis | None = None
    financial: FinancialStatements | None = None
    competitor: CompetitorAnalysis | None = None
    risk: RiskAssessment | None = None
    valuation: ValuationResult | None = None
    # 来自 report_writer LLM 的精简输出(子项目 3 回退)。
    # Flow 据此以及数据字段、确定性竞争/风险/估值分析组装完整的 ``InvestmentReport``。
    writer_output: ReportWriterOutput | None = None
    report: InvestmentReport | None = None
    errors: list[str] = Field(default_factory=list)


def _normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not t or len(t) > 6:
        raise InvalidTickerFormat(raw)
    return t


def _news_items_to_analysis(items: list[NewsItem], ticker: str) -> NewsAnalysis:
    """将 list[NewsItem](注册表约定)转换为 NewsAnalysis(Flow 约定)。

    聚合情绪计数,按相关性取前 3 作为关键事件。空输入 → 按 §3.2 降级返回
    ``NewsAnalysis.empty()``。
    """
    if not items:
        return NewsAnalysis.empty(ticker)

    sentiment_to_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    pos = neg = neu = 0
    weighted = 0.0
    for it in items:
        if it.sentiment == "positive":
            pos += 1
        elif it.sentiment == "negative":
            neg += 1
        else:
            neu += 1
        weighted += sentiment_to_score[it.sentiment] * float(it.relevance_score)

    total = pos + neg + neu
    avg = weighted / total if total else 0.0
    avg = max(-1.0, min(1.0, avg))

    key_events = sorted(items, key=lambda x: x.relevance_score, reverse=True)[:3]
    source = key_events[0].source if key_events else "unavailable"

    return NewsAnalysis(
        ticker=ticker,
        as_of=datetime.utcnow(),
        total_count=total,
        positive_pct=pos / total if total else 0.0,
        negative_pct=neg / total if total else 0.0,
        neutral_pct=neu / total if total else 1.0,
        sentiment_score=avg,
        key_events=key_events,
        source=source,
    )


def _collect_sources(
    market: MarketData | None,
    news: NewsAnalysis | None,
    financial: FinancialStatements | None,
    competitor: CompetitorAnalysis | None,
) -> list[str]:
    """从非平凡的上游组合 ``InvestmentReport.sources``。

    排除字面量 ``"degraded"``(它表示状态而非来源),并去重同时保留
    首次出现的顺序。当 method 为 ``"gics"`` 时,竞争对手来源报告为
    ``"gics_peers"``;否则记录底层 method。
    """
    raw: list[str] = []
    if market is not None and market.source and market.source != "degraded":
        raw.append(market.source)
    if news is not None and news.source:
        raw.append(news.source)
    if financial is not None and financial.source:
        raw.append(financial.source)
    if competitor is not None:
        if competitor.method == "gics":
            raw.append("gics_peers")
        elif competitor.method:
            raw.append(competitor.method)
    # dict.fromkeys 在去重时保留插入顺序。
    return list(dict.fromkeys(raw))


# --- 子项目 3 回退:3 个分析字段的确定性辅助函数。
# LLM 代理(竞争/风险/估值)已回退到仅文本;Flow 在此处根据已填充的数据
# 计算结构化 Pydantic 模型。

# 当竞争对手工具未返回结果时的回退对等集合。镜像子项目 3 之前的
# GICS_PEERS 映射(在提交 b646b75 中删除)。
GICS_PEERS: dict[str, list[str]] = {
    "Technology": ["MSFT", "GOOGL", "META"],
    "Financial Services": ["JPM", "BAC", "WFC"],
    "Healthcare": ["JNJ", "PFE", "UNH"],
    "Energy": ["XOM", "CVX", "COP"],
    "Consumer Cyclical": ["WMT", "AMZN", "HD"],
    "Consumer Defensive": ["PG", "KO", "COST"],
    "Communication Services": ["META", "NFLX", "DIS"],
    "Industrials": ["CAT", "BA", "GE"],
    "Automotive": ["TM", "F", "GM"],
    "Basic Materials": ["LIN", "APD", "FCX"],
    "Real Estate": ["AMT", "PLD", "CCI"],
    "Utilities": ["NEE", "DUK", "SO"],
}


def _gics_peers_for(ticker: str, sector: str | None) -> list[Competitor]:
    """为给定 ticker/sector 构建 3 个 GICS 回退 Competitor 条目。"""
    peers = GICS_PEERS.get(sector or "", ["SPY", "QQQ", "DIA"])[:3]
    return [
        Competitor(
            ticker=t,
            name=t,
            market_cap=0,
            revenue_ttm=Decimal("0"),
        )
        for t in peers
    ]


def _compute_competitor_analysis(state: "AnalysisState") -> CompetitorAnalysis:
    """子项目 3 回退:基于数据的确定性竞争分析。

    使用按公司行业键控的静态 GICS 对等映射。我们不在此代码路径中调用
    ``CompetitorTool``:该工具的嵌套事件循环与 Flow 的异步运行时冲突
    (Python 3.12 抛出"Cannot run the event loop while another loop is running"),
    对等映射是 MVP 回退的确定性事实来源。
    """
    sector = getattr(state.company, "sector", None) if state.company else None
    peers = _gics_peers_for(state.ticker, sector)
    method = "gics"

    # 简单的竞争评分:基线 50,根据每个对等 P/E 差异进行 +/-。
    # GICS_PEERS stub Competitor 没有 P/E,所以 peer_pes 为空,评分保持 50;
    # 这对 MVP 回退来说没问题(没有真实对等数据流过)。
    target_pe = state.market.pe_ratio if state.market and state.market.pe_ratio else None
    score = 50
    if target_pe is not None and peers:
        peer_pes = [p.pe_ratio for p in peers if p.pe_ratio is not None]
        if peer_pes:
            median_pe = sorted(peer_pes)[len(peer_pes) // 2]
            if median_pe > 0:
                ratio = target_pe / median_pe
                # P/E 低于对等 = 估值更好 → 评分更高
                score = max(0, min(100, int(50 + (1.0 - ratio) * 50)))

    return CompetitorAnalysis(
        target_ticker=state.ticker,
        competitors=peers,
        industry_rank=1,
        industry_size=max(10, len(peers) + 1),
        competitive_score=score,
        strengths=[],
        weaknesses=[],
        method=method,
    )


def _default_risk_subscores(state: "AnalysisState") -> list[RiskScore]:
    """从数据派生的 6 类风险子评分(子项目 3 回退)。"""
    fin_score = 50
    if state.financial and state.financial.balance_sheets:
        bs = state.financial.balance_sheets[0]
        if bs.total_assets and bs.total_assets > 0:
            debt_ratio = float(bs.total_liabilities / bs.total_assets * 100)
            # 债务越低 → 财务风险越低
            fin_score = max(0, min(100, int(100 - debt_ratio)))
    mkt_score = 50
    if state.market and state.market.beta is not None:
        # beta 越高 → 市场风险越高
        mkt_score = max(0, min(100, int(abs(state.market.beta) * 50)))
    sentiment_score = 50
    if state.news and state.news.sentiment_score is not None:
        # 负面情绪 → 风险更高
        sentiment_score = max(0, min(100, int(50 - state.news.sentiment_score * 50)))
    return [
        RiskScore(
            category="financial",
            score=fin_score,
            rationale=f"债务/资产比率暗示财务风险 {fin_score}/100",
            evidence=[],
        ),
        RiskScore(
            category="market",
            score=mkt_score,
            rationale=f"由 beta 推断的市场风险:{mkt_score}/100",
            evidence=[],
        ),
        RiskScore(
            category="operational",
            score=50,
            rationale="默认中性(无运营数据)",
            evidence=[],
        ),
        RiskScore(
            category="regulatory",
            score=50,
            rationale="默认中性(无监管数据)",
            evidence=[],
        ),
        RiskScore(
            category="governance",
            score=50,
            rationale="默认中性(无治理数据)",
            evidence=[],
        ),
        RiskScore(
            category="macro",
            score=sentiment_score,
            rationale=f"由新闻情绪推断的宏观风险:{sentiment_score}/100",
            evidence=[],
        ),
    ]


def _compute_risk_assessment(state: "AnalysisState") -> RiskAssessment:
    """子项目 3 回退:基于数据的确定性风险评估。"""
    sub_scores = _default_risk_subscores(state)
    total = int(sum(s.score for s in sub_scores) / len(sub_scores))
    # 等级映射:0-25 低,26-50 中,51-75 高,76-100 极高
    if total <= 25:
        level = "low"
    elif total <= 50:
        level = "medium"
    elif total <= 75:
        level = "high"
    else:
        level = "extreme"
    return RiskAssessment(
        ticker=state.ticker,
        total_score=total,
        level=level,  # 验证器会规范化大小写(此处已经为小写)
        sub_scores=sub_scores,
        top_risks=[s.rationale for s in sorted(sub_scores, key=lambda x: -x.score)[:3]],
        method="weighted_sum_v1",
    )


def _compute_valuation(state: "AnalysisState") -> ValuationResult:
    """子项目 3 回退:确定性的 DCF + 相对估值。"""
    current = state.market.price if state.market else Decimal("0")
    pe = state.market.pe_ratio if state.market and state.market.pe_ratio else 20.0
    peer_pe_avg = 20.0
    relative_value = current * Decimal(str(peer_pe_avg / pe)) if pe > 0 else current

    fcf_data = (
        state.financial.cash_flows[0].free_cash_flow
        if state.financial and state.financial.cash_flows
        else None
    )
    growth_pct = state.market.revenue_growth_yoy if state.market else None
    growth_rate = (growth_pct / 100.0) if growth_pct is not None else 0.05
    shares_outstanding = (
        int(state.market.market_cap / state.market.price)
        if state.market and state.market.price and state.market.price > 0
        else 0
    )
    dcf_value = None
    if fcf_data is not None and fcf_data > 0 and shares_outstanding > 0:
        dcf_value = compute_dcf_value(
            fcf=fcf_data,
            growth_rate=growth_rate,
            shares_outstanding=shares_outstanding,
        )
    if dcf_value is not None and relative_value is not None:
        intrinsic = (dcf_value + relative_value) / 2
        method = "dcf_relative_peg"
    else:
        intrinsic = relative_value
        method = "relative_only"
    upside = float((intrinsic - current) / current) if current else 0.0
    return ValuationResult(
        ticker=state.ticker,
        intrinsic_value_per_share=intrinsic,
        current_price=current,
        upside_pct=round(upside, 4),
        dcf_value=dcf_value,
        relative_value=relative_value,
        peg_ratio=None,
        method=method,  # 验证器将未知值强制转为 "dcf_relative_peg"
        assumptions={"peer_pe_avg": peer_pe_avg, "growth_rate": growth_rate},
    )


def parse_crew_output(
    result: Any, state: "AnalysisState" | None = None
) -> dict[str, Any]:
    """从 ``CrewOutput`` 中提取代理输出,并(可选地)填充 state。

    子项目 2:每个任务输出的 ``raw`` 文本要么是 JSON(成功),
    要么是符合 4 个数据工具的 ``"Error..."`` / ``"No ..."`` /
    ``"...data available..."`` 约定的错误字符串。我们解析 4 个数据
    字段(company、market、news、financial)并相应填充 ``state``。
    company 拉取失败时抛出 ``AllDataSourcesDown``(保留 FastAPI 错误码路径)。

    返回 ``{role_key: parsed_data}`` 映射,以便调用方(及测试)
    检查提取的内容。当提供 ``state`` 时,此函数同时会原地修改 ``state``。
    """
    tasks_output = getattr(result, "tasks_output", []) or []
    extracted: dict[str, Any] = {}

    # 从实际任务构建按 role_key 索引的 {key: raw_text} 查找表。
    raw_by_key: dict[str, str] = {}
    for idx, task_out in enumerate(tasks_output):
        if idx >= len(_TASK_KEYWORDS):
            break
        key = _TASK_KEYWORDS[idx]
        raw_by_key[key] = getattr(task_out, "raw", "") or ""
        extracted[key] = raw_by_key[key]

    # 如果未提供 state,我们只收集 raw 文本。
    if state is None:
        return extracted

    # --- 子项目 2:从代理任务输出解析 4 个数据字段 ---

    # 1. Company(关键路径 —— 失败时抛出 AllDataSourcesDown)
    company, company_err = _extract_data_field(
        raw_by_key.get("company_resolver", ""),
        Company,
        "company_data_unavailable",
    )
    if company is None:
        raise AllDataSourcesDown(
            f"无法解析 {state.ticker}:公司数据不可用"
        )
    state.company = company

    # 2. Market(降级:None + error)
    state.market, market_err = _extract_data_field(
        raw_by_key.get("market_analyst", ""),
        MarketData,
        "market_data_unavailable",
    )
    if market_err:
        state.errors.append(market_err)

    # 3. News(降级:空 NewsAnalysis + error)。工具返回 JSON list。
    news_raw = raw_by_key.get("news_analyst", "").strip()
    if not news_raw or news_raw.startswith("Error") or news_raw.startswith("No ") or "data available" in news_raw.lower():
        state.news = NewsAnalysis.empty(state.ticker)
        state.errors.append("news_data_unavailable")
    else:
        try:
            items_raw = json.loads(news_raw)
            news_items = [NewsItem(**i) for i in items_raw]
            state.news = _news_items_to_analysis(news_items, state.ticker)
        except Exception:
            state.news = NewsAnalysis.empty(state.ticker)
            state.errors.append("news_data_unavailable")

    # 4. Financial(降级:空 FinancialStatements shell + error)
    state.financial, fin_err = _extract_data_field(
        raw_by_key.get("financial_analyst", ""),
        FinancialStatements,
        "financial_data_unavailable",
    )
    if state.financial is None:
        state.financial = FinancialStatements(ticker=state.ticker)
        state.errors.append(fin_err or "financial_data_unavailable")

    # --- 子项目 3 回退:3 个分析任务仅产出文本。Flow 在 ``synthesize_report``
    # 中确定性地计算 competitor/risk/valuation(参见 ``_compute_competitor_analysis``
    # 等)。report_writer LLM 也产出文本(不是 output_pydantic —— 在层级 manager 中
    # 会失败,提示 "Agent must be provided if converter_cls is not specified")。
    # Flow 通过解析 LLM 从原始文本发出的 JSON 来提取精简的 ``ReportWriterOutput``
    # (评级、置信度、持有期、催化剂、markdown),解析失败时回退为 None,以便
    # ``synthesize_report`` 使用合理的默认值。
    state.writer_output = _extract_writer_output(tasks_output, 7, state)

    return extracted


def _extract_pydantic_field(
    tasks_output: list[Any],
    idx: int,
    key: str,
    model_cls: type[BaseModel],
    state: "AnalysisState",
) -> BaseModel | None:
    """从 CrewAI 任务输出中提取 Pydantic 模型。

    当任务配置了 ``output_pydantic=...`` 时,CrewAI 0.203.2 会将 ``task_out.pydantic``
    设置为已校验的模型实例。根据子项目 3 的决定(严格无回退),我们只读取该属性。
    如果缺失或不是预期的模型类型,则将 "<key>_unavailable" 追加到 state.errors
    并返回 None。我们不会尝试通过解析 task_out.raw 来恢复。

    返回模型实例,任何失败情况下返回 ``None``。
    """
    if idx >= len(tasks_output):
        state.errors.append(f"{key}_unavailable")
        return None
    task_out = tasks_output[idx]

    pyd_obj = getattr(task_out, "pydantic", None)
    if isinstance(pyd_obj, model_cls):
        return pyd_obj

    state.errors.append(f"{key}_unavailable")
    return None


def _extract_writer_output(
    tasks_output: list[Any],
    idx: int,
    state: "AnalysisState",
) -> "ReportWriterOutput | None":
    """从仅文本的 CrewAI 任务输出中提取 ``ReportWriterOutput``。

    子项目 3 修复:report_writer 任务不再使用
    ``output_pydantic=ReportWriterOutput``,因为 CrewAI 层级 manager 的 Pydantic
    转换器需要一个 ``agent`` 参数,而在 manager 分派的任务中没有提供
    (``Agent must be provided if converter_cls is not specified``)。LLM 现在
    被提示以内联方式输出 JSON 对象;我们在此处进行解析。

    策略:
      1. 如果 ``task_out.pydantic`` 恰好是有效的 ``ReportWriterOutput``
         (例如在测试路径或未来的 CrewAI 升级中),则直接使用。
      2. 否则首先尝试 ``json.loads(raw)``。
      3. 否则在 raw 文本中搜索第一个 JSON 对象(LLM 可能在其前添加散文、
         将其包裹在 markdown 代码块中,或两者兼有)。
      4. 否则返回 None,并追加 ``"report_writer_unavailable"``。

    遇到任何 JSON 解码或 Pydantic 验证错误时,我们返回 None,让
    ``synthesize_report`` 回退到 rating=Hold / confidence=None。
    """
    if idx >= len(tasks_output):
        state.errors.append("report_writer_unavailable")
        return None
    task_out = tasks_output[idx]

    pyd_obj = getattr(task_out, "pydantic", None)
    if isinstance(pyd_obj, ReportWriterOutput):
        return pyd_obj

    raw = getattr(task_out, "raw", "") or ""
    if not raw.strip():
        state.errors.append("report_writer_unavailable")
        return None

    # 路径 1:纯 JSON 负载。
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # 路径 2:嵌入在散文/markdown 中的 JSON 对象。选择第一个平衡的
        # ``{...}`` 块并尝试解析。使用平衡的花括号遍历而不是正则,
        # 以避免字符串字面量内的花括号误导我们。
        obj = _first_json_object(raw)
        if obj is None:
            state.errors.append("report_writer_unavailable")
            return None

    if not isinstance(obj, dict):
        state.errors.append("report_writer_unavailable")
        return None

    try:
        return ReportWriterOutput.model_validate(obj)
    except ValidationError:
        state.errors.append("report_writer_unavailable")
        return None


def _first_json_object(text: str) -> dict | None:
    """返回 ``text`` 中找到的第一个平衡的顶级 JSON 对象。

    遍历字符串时跟踪嵌套深度,将 ``"..."`` 字符串字面量视为不透明
    (因此其内部的花括号不会改变深度计数器)。返回解析后的字典,
    如果找不到或无法解码平衡对象则返回 ``None``。
    """
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    # 不平衡或格式错误;继续扫描下一个开始的花括号。
                    start = None
    return None


def _extract_data_field(
    raw: str, model_cls: type, error_msg: str
) -> tuple[Any | None, str | None]:
    """将工具输出字符串解析为 Pydantic 模型,或返回 None + 错误。

    失败检测顺序:
      1. 空 / 仅空白 → 失败
      2. 以 "Error" / "No " 开头 / 包含 "data available" → 失败
         (匹配所有 5 个数据工具使用的错误字符串约定)
      3. JSON 数组包装(CrewAI 的 manager LLM 将工具调用轨迹发出为
         ``[{"ticker": ...}, {"error": "..."}]``);解包到第一个对象
      4. 包含顶级 ``"error"`` 键的 JSON 对象 → 失败(上游工具返回错误,
         manager LLM 把它包装到结构化对象中,而不是传播错误字符串)
      5. 尝试 ``model_cls.model_validate_json``;出现 ValidationError → 失败
      6. 退化 Company shell 检查:成功验证但 ``market_cap == 0`` 且
         ``sector in {"Unknown", ""}`` 的 Company 几乎肯定是 LLM 幻觉
         (真实的 Yahoo/AlphaVantage/Finnhub/SECEdgar 数据源总是从真实
         数据填充这些字段)
      7. 否则 → 成功,返回解析后的模型

    成功时返回 ``(model, None)``,失败时返回 ``(None, error_msg)``。
    """
    raw = raw.strip() if raw else ""
    if not raw:
        return None, error_msg
    # 错误字符串约定:工具返回 "Error fetching X: ..." 或 "No X data..."
    lowered = raw.lower()
    if (
        raw.startswith("Error")
        or raw.startswith("No ")
        or "data available" in lowered
    ):
        return None, error_msg
    # JSON 数组包装:CrewAI 层级 manager LLM 将每个任务的工具调用
    # 轨迹发出为 JSON 对象数组(例如子项目 3 Task 5 中的 "Repaired JSON"
    # 调试行)。解包到第一个对象,以便 ``model_validate_json`` 可以尝试
    # 对其进行验证。
    if raw.startswith("["):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and arr:
                raw = json.dumps(arr[0])
            # 空数组 → 无负载 → 视为失败
            else:
                return None, error_msg
        except Exception:
            return None, error_msg
    # 包含顶级 "error" 键的 JSON 对象:上游工具失败,manager LLM 将其
    # 作为结构化对象呈现。视为失败,以便 company_fetch 路径抛出
    # AllDataSourcesDown,而不是使用退化 shell 继续。
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "error" in obj:
                return None, error_msg
        except Exception:
            return None, error_msg
    try:
        model = model_cls.model_validate_json(raw)
    except Exception:
        return None, error_msg
    # 退化 Company shell:通过验证但 market_cap 为零且 sector 为占位符
    # 的 Company 几乎肯定是 LLM 幻觉(真实数据源总是从真实数据填充这些字段)。
    # 视为失败,以便 company_fetch 路径抛出 AllDataSourcesDown。
    if (
        model_cls is Company
        and getattr(model, "market_cap", 1) == 0
        and getattr(model, "sector", "") in {"Unknown", ""}
    ):
        return None, error_msg
    return model, None


class AnalysisFlow(Flow[AnalysisState]):
    """顶层 Flow:包裹 AnalysisCrew 的 2 步轻量级外壳。"""

    def _emit_progress(self, step: str, state: str) -> None:
        """为 Flow 步骤边界触发可选的进度回调。

        该回调是 :meth:`kickoff_with_timeout` 设置的仅运行时属性;
        ``Flow`` 是 Pydantic 模型,因此我们通过 ``getattr`` 访问,默认值为 ``None``。
        回调异常会被捕获并记录,以避免有缺陷的 UI 崩溃整个 Flow。
        """
        cb = getattr(self, "_progress_callback", None)
        if cb is None:
            return
        try:
            cb(step, state)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "progress_callback_failed",
                step=step,
                state=state,
                error=str(exc),
            )

    @start()
    async def run_crew(
        self,
        ticker: str | None = None,
        crewai_trigger_payload: dict[str, Any] | None = None,
    ) -> None:
        """第 1 步:驱动 8 代理 Crew 产出分析结果。

        子项目 2:Flow 不再预取数据。4 个数据代理(CompanyResolver、MarketAnalyst、
        NewsAnalyst、FinancialAnalyst)各自在 Crew 内部调用其工具以获取最新数据。
        我们仅将 ticker 传递给 ``crew.kickoff``;产生的任务输出由
        ``parse_crew_output`` 解析回 ``state``。
        """
        # 从任何支持的通道解析 ticker。
        raw_ticker = (
            ticker
            or (crewai_trigger_payload or {}).get("ticker")
            or self.state.ticker
            or ""
        )
        self._emit_progress("validate_ticker", "running")
        try:
            normalized = _normalize_ticker(raw_ticker)
        except InvalidTickerFormat:
            self._emit_progress("validate_ticker", "failed")
            raise
        self.state.ticker = normalized
        self._emit_progress("validate_ticker", "complete")
        log.info("flow_step_started", step="run_crew", ticker=normalized)
        self._emit_progress("run_crew", "running")

        # 驱动 8 代理 crew。4 个数据任务(company_resolver、market_analyst、
        # news_analyst、financial_analyst)通过 async_execution=True 并行运行;
        # 每个任务都调用自己的工具以获取最新数据。Crew.kickoff 是同步的 →
        # 包装到 _kickoff_sync + to_thread 中,以便 ``asyncio.wait_for`` 可以在
        # 执行中途取消(子项目 2 延期阻塞项 #1:asyncio 关闭竞争)。
        def _kickoff_sync() -> Any:
            return AnalysisCrew().kickoff(inputs={"ticker": normalized})

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_kickoff_sync),
                timeout=FLOW_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._emit_progress("run_crew", "failed")
            log.error("crew_timeout", ticker=normalized)
            raise
        except Exception as exc:
            # CrewAI 有时会将上游 HTTP 429(Token Plan 已用完)表现为
            # ``AttributeError: 'NoneType' object has no attribute 'choices'``,
            # 因为 LLM SDK 返回了错误响应体而不是正常响应。同时检测字面 429
            # 字符串和下游症状,并重新抛出为 ``LLMRateLimited``,以便 API 路由
            # 可以返回 503,前端可以显示清晰的"稍后重试"消息,而不是 5 分钟超时。
            self._emit_progress("run_crew", "failed")
            msg = str(exc)
            if (
                "429" in msg
                or "rate_limit_error" in msg
                or "Token Plan" in msg
                or ("choices" in msg and "NoneType" in msg)
            ):
                log.error("crew_llm_rate_limited", ticker=normalized, error=msg)
                raise LLMRateLimited(
                    f"LLM 被限流(HTTP 429 / Token Plan 已用完)。"
                    f"请稍后几分钟再试。底层错误:{msg[:200]}"
                ) from exc
            log.error("crew_execution_error", ticker=normalized, error=msg)
            raise CrewExecutionError(
                f"CrewAI 执行失败:{msg[:300]}"
            ) from exc

        # 解析 crew 输出 → 填充下游任务使用的 self.state 字段。
        # 如果 company 拉取失败,则抛出 AllDataSourcesDown。
        self._emit_progress("parse_crew_output", "running")
        try:
            parse_crew_output(result, self.state)
        except Exception:
            self._emit_progress("parse_crew_output", "failed")
            raise
        self._emit_progress("parse_crew_output", "complete")

        self._emit_progress("run_crew", "complete")
        log.info("flow_step_completed", step="run_crew", ticker=normalized)

    @listen(run_crew)
    async def synthesize_report(self) -> None:
        """子项目 3 回退:从数据 + 确定性 competitor/risk/valuation + LLM 的
        ``ReportWriterOutput``(评级、置信度、持有期、催化剂、markdown)
        构建完整的 ``InvestmentReport``。

        一旦合成失败,抛出 ``ReportGenerationError``,以便调用方
        (按规范 §5.2 的 FastAPI 处理器)可以返回 HTTP 500。
        """
        log.info("flow_step_started", step="synthesize_report", ticker=self.state.ticker)
        assert self.state.company is not None
        assert self.state.news is not None
        assert self.state.financial is not None

        # §3.2:market 可能为 None(降级)—— 替换为最小占位符,
        # 以便仍能构建 InvestmentReport。
        market = self.state.market
        if market is None:
            market = MarketData(
                ticker=self.state.ticker,
                as_of=datetime.utcnow(),
                price=Decimal("0"),
                change_pct=0.0,
                volume=0,
                market_cap=self.state.company.market_cap,
                source="degraded",
            )
            self.state.market = market

        # 3 项确定性分析。这些取代了已删除的 LLM 驱动路径。
        self._emit_progress("compute_analyses", "running")
        self.state.competitor = _compute_competitor_analysis(self.state)
        self.state.risk = _compute_risk_assessment(self.state)
        self.state.valuation = _compute_valuation(self.state)
        self._emit_progress("compute_analyses", "complete")

        # LLM 合成(评级、置信度、持有期、催化剂、markdown)。
        # 如果 LLM 未能产出 ReportWriterOutput,则回退到保守的默认值,
        # 以便前端仍能渲染一些内容。
        wo = self.state.writer_output
        if wo is None:
            log.warning("writer_output_missing", ticker=self.state.ticker)
            wo = ReportWriterOutput(
                rating="Hold",
                confidence=None,
                investment_horizon="medium",
                catalysts=[],
                markdown=(
                    f"## {self.state.ticker} 投资研究报告\n\n"
                    "报告合成器未能从数据中提取完整结论。"
                    f"请参考风险评级 ({self.state.risk.level}) 和估值结果 "
                    f"(${self.state.valuation.intrinsic_value_per_share}) 判断投资价值。"
                ),
            )
            self.state.errors.append("writer_output_unavailable")

        try:
            self._emit_progress("assemble_report", "running")
            health_score = compute_financial_health(self.state.financial)

            self.state.report = InvestmentReport(
                report_id=str(uuid.uuid4()),
                ticker=self.state.ticker,
                generated_at=datetime.utcnow(),
                data_as_of={
                    "market": market.as_of,
                    "news": self.state.news.as_of,
                },
                company=self.state.company,
                market=market,
                financial=self.state.financial,
                financial_health_score=health_score,
                news=self.state.news,
                competitors=self.state.competitor,
                risk=self.state.risk,
                valuation=self.state.valuation,
                rating=wo.rating,
                confidence=wo.confidence,
                investment_horizon=wo.investment_horizon,
                catalysts=wo.catalysts,
                markdown=wo.markdown,
                sources=_collect_sources(
                    market,
                    self.state.news,
                    self.state.financial,
                    self.state.competitor,
                ),
                disclaimer=DISCLAIMER_TEXT,
            )
            self._emit_progress("assemble_report", "complete")
            log.info(
                "flow_step_completed",
                step="synthesize_report",
                ticker=self.state.ticker,
                report_id=self.state.report.report_id,
                rating=wo.rating,
                confidence=wo.confidence,
                health_score=health_score,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._emit_progress("assemble_report", "failed")
            log.error(
                "flow_step_failed",
                step="synthesize_report",
                ticker=self.state.ticker,
                error=str(exc),
            )
            raise ReportGenerationError(
                f"为 {self.state.ticker} 合成报告失败:{exc}"
            ) from exc

    async def kickoff_with_timeout(
        self,
        inputs: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """§3.4 整个 Flow 的超时包装。

        CrewAI Flow 的 ``kickoff`` 是同步的;``kickoff_async`` 返回一个协程,
        我们可以将其包装到 ``asyncio.wait_for`` 中。超时时底层协程会被取消,
        ``asyncio.TimeoutError`` 传播到调用方,(按规范 §5.2)对应 HTTP 504
        GATEWAY_TIMEOUT。

        如果提供了 ``progress_callback``,它会在 kickoff 期间绑定到此 Flow 实例,
        并在每个主要步骤边界(``validate_ticker`` / ``run_crew`` / ``parse_crew_output`` /
        ``compute_analyses`` / ``assemble_report``)调用,以便实时 UI 调用方渲染进度。
        ``Flow[AnalysisState]`` 是 Pydantic 模型,因此我们通过 ``object.__setattr__``
        附加,以绕过字段验证。
        """
        if progress_callback is not None:
            object.__setattr__(self, "_progress_callback", progress_callback)
        # 重置 CrewAI 事件桥接,以便 UI 的轮询线程在每次运行时看到干净的滚动日志。
        # ``install()`` 是幂等的——这里第一次调用也会订阅处理器(如果尚未完成)。
        bridge = get_default_bridge()
        bridge.reset()
        try:
            return await asyncio.wait_for(
                self.kickoff_async(inputs=inputs),
                timeout=FLOW_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.error(
                "flow_timeout",
                ticker=(inputs or {}).get("ticker", self.state.ticker),
                timeout_seconds=FLOW_TIMEOUT_SECONDS,
            )
            raise
