# AlphaQuant Architecture

> 项目结构与模块边界指南。给新加入的贡献者一张地图。

## Overview

AlphaQuant 是一个 AI 投资研究分析师。输入美股代码（`AAPL`），约 30 秒内输出结构化的投资研究报告（评级 / 风险雷达 / 财务面板 / 估值 / 舆情 / 竞争对手 / Markdown 全文）。

支持三种使用方式：

| 入口 | 命令 | 适用场景 |
| --- | --- | --- |
| CLI | `uv run python -m alphaquant AAPL` | 一次性查询、CI 集成、shell 管道 |
| API | `uv run uvicorn alphaquant.main:app` | 程序化访问、与其他系统集成 |
| Web | `uv run streamlit run src/alphaquant/interfaces/frontend/app.py` | 人工浏览、历史追踪、多 ticker 对比 |

## Layered Architecture

代码按 4 层组织，每层的依赖方向严格向下：

```
┌─────────────────────────────────────────────────────────┐
│  interfaces/   ← 用户入口（CLI / API / Web UI）          │
└──────────────┬──────────────────────────────────────────┘
               │ 调 core.run_analysis_async
               ▼
┌─────────────────────────────────────────────────────────┐
│  core.py        ← 应用服务入口（唯一公开入口）          │
└──────────────┬──────────────────────────────────────────┘
               │ 调 flows + infrastructure
               ▼
┌─────────────────────────────────────────────────────────┐
│  agents/  flows/  scoring/  models/  tools/             │
│  ← 领域层（LLM 编排、业务逻辑、纯计算、模型 schema）    │
└──────────────┬──────────────────────────────────────────┘
               │ 调 LLM、数据源
               ▼
┌─────────────────────────────────────────────────────────┐
│  infrastructure/                                        │
│  ← 外部资源适配（LLM / 数据源 / SQLite / 配置）         │
└─────────────────────────────────────────────────────────┘

       observability/   ← 横切关注点（日志 / 成本追踪）
```

依赖规则：

| 层 | 可依赖 | 不可依赖 |
| --- | --- | --- |
| `interfaces.*` | `core`、`infrastructure.persistence` | `agents` / `flows` 直接 |
| `infrastructure.*` | `models` / `exceptions` / `observability` | `interfaces.*` / `agents` / `flows` |
| `core.py` | `flows` / `agents` / `models` / `scoring` / `infrastructure` | `interfaces.*` |
| `flows/`, `agents/`, `scoring/`, `tools/` | `models` / `infrastructure.llm` / `infrastructure.data_sources` | `interfaces.*` / `infrastructure.persistence` |

## Module Map

```
src/alphaquant/
├── __init__.py           # package marker, __version__
├── __main__.py           # python -m alphaquant entry → interfaces.cli.main
├── core.py               # run_analysis / run_analysis_async（唯一应用入口）
├── exceptions.py         # 自定义异常层级
│
├── agents/               # 8 个 CrewAI Agent（LLM 编排）
├── flows/                # analysis_flow.py（编排 8 个 Agent）
├── models/               # 8 个 Pydantic 模型（InvestmentReport 等）
├── scoring/              # 4 个纯计算模块（rating / financial_health / risk / competitive）
├── tools/                # 5 个 Agent 工具（market_data / financial / news / competitor / dcf）
├── observability/        # structlog 配置 + token 成本追踪
│
├── infrastructure/       # 外部资源适配层
│   ├── config.py             # pydantic-settings 单例
│   ├── llm.py                # LiteLLM 客户端封装
│   ├── data_sources/         # 5 个数据源 + DataSource 抽象
│   │                          # (Yahoo → Alpha Vantage → Finnhub → SEC EDGAR → NewsAPI)
│   └── persistence/          # SQLite 持久化（DB + ReportRecord）
│
└── interfaces/           # 用户入口层
    ├── cli.py                # argparse CLI
    ├── api/                  # FastAPI 路由层
    │                          # (routes / schemas / rate_limiter)
    └── frontend/             # Streamlit UI（4 个页面：Analyze / History / Compare / Settings）
        ├── app.py                # st.navigation 入口
        ├── components/           # rating_card / metrics_panel / charts
        └── pages/                # 1_Analyze / 2_History / 3_Compare / 4_Settings
```

**核心约束：前端不写数据库逻辑**。`interfaces/frontend/` 只调 `core.run_analysis_async` 和 `infrastructure.persistence.DB`；不直接碰 SQLite、SQL、Pydantic 模型 schema。

## Data Flow

一次完整分析请求的路径（以 CLI `python -m alphaquant AAPL` 为例）：

```
1. python -m alphaquant
   ↓
2. alphaquant.__main__
   ↓ from alphaquant.interfaces.cli import main
3. alphaquant.interfaces.cli.main
   ↓ argparse 解析 args.ticker
4. alphaquant.core.run_analysis_async(ticker)
   ↓ 注入 timeout=120s，捕获 ticker 格式异常
5. alphaquant.flows.analysis_flow.AnalysisFlow
   ├─ [1] CompanyResolver          → tools/market_data_tool → DataSourceRegistry
   ├─ [2] MarketAnalyst    ─┐                            ↓
   ├─ [3] NewsAnalyst      ─┤  并行 (asyncio.gather)   infrastructure.data_sources
   ├─ [4] FinancialAnalyst ─┘                            ↓
   ├─ [5] CompetitorAnalyst                            yfinance / AlphaVantage / ...
   ├─ [6] RiskAnalyst
   ├─ [7] ValuationAnalyst
   └─ [8] ReportWriter
        ↓
        scoring.rating.determine_rating(valuation, risk, sentiment)
        ↓
        InvestmentReport (Pydantic)
   ↓
6. 回传 core.run_analysis_async
   ↓
7. interfaces.cli 输出 JSON / Markdown
```

每个分析步骤都可能降级（degradation mode）：数据源失败时返回 `*.empty()` 占位，`InvestmentReport.sources` 标记降级，前端 UI 显示"已降级"。整个 Flow 永不因单个数据源失败而中断。

## Tests

```
tests/
├── test_agents.py            # 8 个 Agent 的 LLM mock 测试
├── test_api.py               # FastAPI 路由 + 限流 + 异常映射
├── test_db.py                # SQLite 持久化（tmp_path，7 个用例）
├── test_flow.py              # AnalysisFlow 端到端
├── test_observability.py     # structlog + 成本追踪
├── test_observability_wiring.py  # 跨模块接线
├── test_rating.py            # 5 级评级纯函数
├── test_scoring.py           # 4 个 scoring 模块
├── test_tools.py             # 5 个工具
└── smoke.py                  # 7 步端到端冒烟
```

总计 186 个测试。运行：`uv run pytest tests/ -q`。

## Container Deployment

`Dockerfile`（多阶段）：builder 用 uv export 生成 wheel，runtime 用 `python:3.11-slim` + 非 root 用户。

`docker-compose.yml`：单服务，端口 8501，挂载 `./data:/app/data` 用于 SQLite 持久化，环境变量从 `.env` 读取。