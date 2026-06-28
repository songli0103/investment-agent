"""AnalysisCrew:编排分析流水线的 8 代理 CrewAI Crew。

================================================================
设计决策解读(为什么这么写)
================================================================

本文件展示了 CrewAI 的 4 个核心概念如何协作:Agent、Task、Crew、Process。
每一个选择都有明确理由,记录在此供学习参考。

**1. 8 个 Agent + 8 个 Task 一一对应**

不是所有 CrewAI 项目都这么做 —— 常见模式是 3-5 个 Agent 各自处理
多个 Task。这里 1:1 对应是因为:
- 任务输入/输出契约清晰(每个 Task 只对应一个 Agent 的专业知识)
- 调试时容易定位(知道是哪个 Agent 出了问题)
- 切换模型时粒度细(可以单独给某个 Agent 换 LLM)

**2. Process.hierarchical + 共享 manager_llm**

CrewAI 有 3 种 process:
- sequential:严格按列表顺序,前一个 task 的输出自动注入下一个
- hierarchical:增加一个 manager LLM,manager 决定派谁做什么
- consensualic(已废弃)

我们用 hierarchical,因为:
- 8 个 task 之间的依赖是隐式的(数据 → 分析 → 写报告)
- manager 能根据中间结果动态调整(例如某个数据源失败时跳过)
- 代价:多 1 次 manager LLM 调用,但换来可观测性(manager 决策可见)

``manager_llm`` 跟 agent ``llm`` 共享同一个实例,简化配置。生产中
也常把 manager 单独配一个更强或更便宜的模型,作为优化方向。

**3. memory=False(子项目 1 故意关闭)**

``memory=True`` 时,CrewAI 会自动把历史 task 输出注入到新 task 的
context,跨 run 也保留。这对生产有副作用:
- token 消耗不可预测(上下文会无限增长)
- 调试困难(同样输入可能产生不同输出,memory 不一样)

我们关掉它,让 Flow 层(``flows/analysis_flow.py``)显式控制状态。
所有"记忆"都在 ``AnalysisState`` 这个 Pydantic 模型里。

子项目 4 会重新打开,届时需要额外的 token 预算管理。

**4. _ASYNC_TASK_INDICES = {0..6}**

CrewAI 的 ``Task(async_execution=True)`` 表示这个 task 不阻塞下一个。
我们用这个标志:
- 数据 task (0-3):完全独立,并行 OK
- 分析 task (4-6):独立,并行 OK
- 报告写作者 (7):必须等分析结果,**不能** async

为什么不全 async?如果全 async,manager 调度时可能让报告写作者跟
数据 task 抢资源,而且它的 context 注入会变得不可靠。显式控制并行
边界比全部 async 稳定。

**5. max_iter=10 + max_execution_time=480**

CrewAI 的 hierarchical 模式有个隐性的 manager 委托-重试循环:
manager 让 agent 干活 → 拿到结果 → 决定下一步 → 派给另一个 agent。
LLM 偶尔会"卡住"(例如 429 限流),manager 会重试。

``max_iter=10``:manager 最多委托 10 次。少了不够完成 8 个 task,
多了在 429 时会消耗整个 600s 流程超时,前端一直转圈。这是权衡值。

``max_execution_time=480``:任何单个 task 最多 480s。兜底,防止
LLM 调用挂死导致整个流程卡住。

**6. _TASK_TEMPLATES 是 3-tuple**

每个 entry 是 ``(key, description, pydantic_model_or_None)``。
第三个字段历史上是 pydantic 模型(让 LLM 输出结构化 JSON),后来
回退为 None —— LLM 在生产中输出结构无效的内容(错误的字段名、
口语化文本),导致 CrewAI 转换器在 180s Flow 超时内反复重试。
现在 8 个 task 都用纯文本输出,Flow 层确定性地做结构化计算。

================================================================
子项目 1:Crew 使用 Process.hierarchical 和共享的 manager_llm 构建,
但 ``memory=False``,代理是调用现有数据工具的"瘦"包装器。
子项目 4 将启用 ``memory`` 和 ``allow_delegation=True``。

子项目 2:4 个数据代理(CompanyResolver、MarketAnalyst、NewsAnalyst、
FinancialAnalyst)通过 Crew 内部的数据工具(``CompanyLookupTool`` +
3 个现有数据工具)获取各自的数据。Flow(``flows/analysis_flow.py``)
现在是纯编排:它只向 ``crew.kickoff`` 传递 ``{"ticker": ...}``,
并对结果调用 ``parse_crew_output()`` 以填充 ``AnalysisState``。
"""
from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, Process, Task

from alphaquant.agents.company_resolver import build_company_resolver_agent
from alphaquant.agents.competitor_analyst import build_competitor_analyst_agent
from alphaquant.agents.financial_analyst import build_financial_analyst_agent
from alphaquant.agents.market_analyst import build_market_analyst_agent
from alphaquant.agents.news_analyst import build_news_analyst_agent
from alphaquant.agents.report_writer import build_report_writer_agent
from alphaquant.agents.risk_analyst import build_risk_analyst_agent
from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent
from alphaquant.infrastructure.llm import get_llm
from pydantic import BaseModel


# ----------------------------------------------------------------
# _TASK_TEMPLATES:8 个 task 的 3 层流水线
# ----------------------------------------------------------------
# 整个 crew 的任务按"数据 → 分析 → 写报告"三段式编排:
#   idx 0-3 (data)    :拉外部数据(公司元数据 / 市场 / 新闻 / 财务)
#   idx 4-6 (analysis):基于数据做三种结构化分析(竞争 / 风险 / 估值)
#   idx 7   (writer)  :把前 7 步的输出合成 markdown 报告
#
# 每个 entry 是 (key, description, pydantic_model_or_None)。
#
# 子项目 3(然后回退):3 个分析任务(idx 4-6)最初设置了 output_pydantic,
# 让 LLM 产出结构化 Pydantic 输出。在生产环境中,LLM 输出结构无效的
# 内容(错误的字段名、口语化文本),导致 CrewAI 转换器在 180 秒 Flow
# 超时内反复重试。我们将这些任务回退为纯文本 —— Flow 现在确定性地
# 计算 competitor/risk/valuation。report_writer(idx 7)产出
# ``ReportWriterOutput``(``InvestmentReport`` 的精简子集);
# Flow 组装完整的 ``InvestmentReport``。
_TASK_TEMPLATES: list[tuple[str, str, type[BaseModel] | None]] = [
    (
        "company_resolver",
        "验证 ticker '{ticker}' 并返回规范的公司元数据。",
        None,
    ),
    (
        "market_analyst",
        "获取 '{ticker}' 的市场数据。",
        None,
    ),
    (
        "news_analyst",
        "获取 '{ticker}' 的近期新闻。",
        None,
    ),
    (
        "financial_analyst",
        "获取 '{ticker}' 的财务报表。",
        None,
    ),
    (
        "competitor_analyst",
        "用纯文本总结 '{ticker}' 的竞争格局。"
        "不要产出结构化 Pydantic 输出;Flow 会根据数据计算"
        "结构化 CompetitorAnalysis。你的文本用作报告撰写者的上下文。",
        None,
    ),
    (
        "risk_analyst",
        "用纯文本总结 '{ticker}' 的关键风险因素。"
        "不要产出结构化 Pydantic 输出;Flow 会根据数据计算"
        "结构化 RiskAssessment。你的文本用作报告撰写者的上下文。",
        None,
    ),
    (
        "valuation_analyst",
        "用纯文本总结 '{ticker}' 的估值分析(DCF + 相对估值)。"
        "不要产出结构化 Pydantic 输出;Flow 会根据数据计算"
        "结构化 ValuationResult。你的文本用作报告撰写者的上下文。",
        None,
    ),
    (
        "report_writer",
        "为 '{ticker}' 合成最终 markdown 报告和评级。"
        "输出一个 JSON 对象,包含以下字段:rating、confidence、"
        "investment_horizon、catalysts、markdown。Flow 解析 JSON;"
        "不要使用结构化 Pydantic 输出(与层级 manager 不兼容)。",
        None,
    ),
]


class AnalysisCrew:
    """将 8 个 CrewAI 代理包装在层级 Crew 中。

    子项目 1 将 crew 保留为结构外壳:它可以端到端调用,但其输出由
    调用 Flow 中的 ``parse_crew_output`` 规范化。子项目 3 将让代理
    进行真正的推理;子项目 4 将启用内存和对等委托。
    """

    # 通过 ``async_execution=True`` 并行运行的任务索引。
    # 数据(0-3)和分析(4-6)独立 → 并行。
    # 报告撰写者(7)依赖分析输出 → 串行。
    #
    # 设计原则:CrewAI 的 ``async_execution`` 表示"这个 task 不阻塞
    # 下一个"。如果把报告写作者也设成 async,manager 调度时可能让它
    # 跟数据 task 抢资源,而且 context=[tasks[4,5,6]] 的注入会变得
    # 不可靠。显式控制并行边界比"全部 async"稳定。
    _ASYNC_TASK_INDICES: set[int] = {0, 1, 2, 3, 4, 5, 6}

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.1)
        self.agents: list[Agent] = self._build_agents()
        self.tasks: list[Task] = self._build_tasks()
        self.crew: Crew = self._build_crew()

    def _build_agents(self) -> list[Agent]:
        return [
            build_company_resolver_agent(self._llm),
            build_market_analyst_agent(self._llm),
            build_news_analyst_agent(self._llm),
            build_financial_analyst_agent(self._llm),
            build_competitor_analyst_agent(self._llm),
            build_risk_analyst_agent(self._llm),
            build_valuation_analyst_agent(self._llm),
            build_report_writer_agent(self._llm),
        ]

    def _build_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        for idx, (role_key, description, pydantic_model) in enumerate(_TASK_TEMPLATES):
            agent = self.agents[idx]
            task_kwargs: dict[str, Any] = {
                "description": description,
                "expected_output": pydantic_model.__name__ if pydantic_model else "raw text",
                "agent": agent,
                "async_execution": idx in self._ASYNC_TASK_INDICES,
            }
            if pydantic_model is not None:
                task_kwargs["output_pydantic"] = pydantic_model
            # 报告撰写者(idx 7)消费 3 个分析任务的输出作为 context。
            # 注意:即便 idx 4-6 是 async,manager 调度时会保证 context
            # 准备好才启动 idx 7 —— 这就是 ``Task.context`` 的作用。
            if idx == 7:
                task_kwargs["context"] = [tasks[4], tasks[5], tasks[6]]
            tasks.append(Task(**task_kwargs))
        return tasks

    def _build_crew(self) -> Crew:
        # ``max_iter`` 限制层级 manager 在放弃前会运行多少 CrewAI
        # 委托-重试循环。如果没有上限,被限流的 LLM(HTTP 429)会触发
        # 无界重试循环,消耗整个 ``FLOW_TIMEOUT_SECONDS``(600 秒),
        # 让前端一直转。``max_execution_time`` 是兜底,防止任何单个
        # 调用挂起。
        #
        # memory=False 见模块顶部"设计决策解读 §3" —— 让 Flow 层
        # 显式控制状态,避免 token 不可控增长。
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=self._llm,
            memory=False,
            verbose=False,
            max_iter=10,
            max_execution_time=480,
        )

    def kickoff(self, inputs: dict[str, Any]):
        """同步入口点 —— 包装 Crew.kickoff()。

        Flow 层负责在 ``asyncio.to_thread`` 内调用此方法,以避免
        阻塞事件循环。
        """
        return self.crew.kickoff(inputs=inputs)


__all__ = ["AnalysisCrew"]