# AlphaQuant 生产化重构设计

> **For agentic workers:** 本 spec 仅描述设计目标。具体实施步骤会在后续的 implementation plan 中给出（用 writing-plans skill 生成）。

**日期：** 2026-06-21
**范围：** 项目结构 + 顶层文件归类 + 一份 ARCHITECTURE.md
**目标形态：** 开源项目品质（README + ARCHITECTURE + CI 可选 + 徽章可选 + 语义化版本可选）

---

## Goal

把 `src/alphaquant/` 从"功能堆叠式"重构为"分层清晰式"，使外部贡献者一眼能看懂模块边界、内部开发者知道新代码该往哪里放。具体子目标：

1. **前端层不再持有持久化逻辑**（最高优先级 — 用户明确要求）
2. 顶层散落的 `.py` 文件按职责归入子包
3. 引入 `infrastructure/` 和 `interfaces/` 两个新顶层子包，明确"外部适配"和"用户入口"的边界
4. 写一份 `docs/ARCHITECTURE.md`，给读者一张"项目怎么组织"的地图
5. **不动任何业务逻辑、算法、prompt、模型 schema**

---

## Non-Goals（明确不做）

- ❌ 不拆 `flows/analysis_flow.py`（623 行单体按用户决定保留）
- ❌ 不引入 mypy 严格模式 / pre-commit 钩子 / GitHub Actions CI
- ❌ 不写 `CONTRIBUTING.md` / `CHANGELOG.md` / 加 `LICENSE`
- ❌ 不加徽章、release 自动化、semantic versioning
- ❌ 不改业务代码、scoring 算法、LLM prompt、模型 schema
- ❌ 不动 `core.py` / `exceptions.py` / `observability/` 的位置和内部代码

---

## Architecture

### 现状（重构前）

```
src/alphaquant/
├── __init__.py
├── __main__.py
├── cli.py                  ← 散落顶层
├── main.py                 ← 散落顶层
├── core.py                 ← 散落顶层
├── config.py               ← 散落顶层
├── llm.py                  ← 散落顶层
├── exceptions.py           ← 散落顶层
├── agents/
├── flows/
├── models/
├── scoring/
├── tools/
├── observability/
├── data_sources/           ← 散落顶层（已经是子包但不在分层里）
├── api/                    ← 散落顶层
└── frontend/
    ├── app.py
    ├── db.py               ← ❌ 持久化逻辑在前端
    ├── models.py           ← ❌ 持久化 DTO 在前端
    ├── components/
    └── pages/
```

### 目标（重构后）

```
src/alphaquant/
├── __init__.py
├── __main__.py
├── core.py                 ← 顶层：公共应用服务入口
├── exceptions.py           ← 顶层：跨层异常
│
├── agents/                 ← 领域：LLM 编排的 8 个 Agent
├── flows/                  ← 领域：analysis_flow.py 编排器
├── models/                 ← 领域：8 个 Pydantic 模型
├── scoring/                ← 领域：4 个纯计算模块
├── tools/                  ← 领域：Agent 可调用的工具
├── observability/          ← 横切：structlog + cost_tracker
│
├── infrastructure/         ← 🆕 外部资源适配层
│   ├── __init__.py
│   ├── config.py           ← 从顶层移入
│   ├── llm.py              ← 从顶层移入
│   ├── data_sources/       ← 从顶层移入（含 5 个源 + base）
│   └── persistence/        ← 🆕
│       ├── __init__.py
│       ├── db.py           ← 从 frontend/db.py 移入
│       └── models.py       ← 从 frontend/models.py 移入
│
└── interfaces/             ← 🆕 用户入口层
    ├── __init__.py
    ├── cli.py              ← 从顶层移入
    ├── api/                ← 从顶层移入（routes/schemas/rate_limiter）
    └── frontend/           ← 从顶层移入（db.py / models.py 已移除）
        ├── app.py
        ├── components/
        └── pages/
```

### 分层依赖规则（强制）

```
interfaces  ───────►  core.py  ───────►  flows / agents / scoring
     │                                    │
     │                                    ▼
     └──►  infrastructure.persistence ◄── infrastructure.llm
                    │
                    ▼
              infrastructure.data_sources
```

| 层 | 可依赖 | 不可依赖 |
| --- | --- | --- |
| `interfaces.*` | `core`、`infrastructure.persistence` | `agents` / `flows` 直接（必须经 `core`） |
| `infrastructure.*` | `models` / `exceptions` / `observability` | `interfaces.*` / `agents` / `flows` |
| `core.py` | `flows` / `agents` / `models` / `scoring` / `infrastructure` | `interfaces.*` |
| `flows/`, `agents/`, `scoring/`, `tools/` | `models` / `infrastructure.llm` / `infrastructure.data_sources` | `interfaces.*` / `infrastructure.persistence` |

> 注：`infrastructure.persistence` 当前只被 `interfaces.frontend` 用。CLI 和 API 不持久化（按 YAGNI 保留此约束）。

---

## Components（新增/移动的）

### 1. `infrastructure/persistence/`（新模块）

**目的：** 把 SQLite 持久化从 `frontend/` 抽出来，让前端层只关心 UI。

**文件：**
- `db.py` — 原 `frontend/db.py`，`DB` 类（8 个方法）
- `models.py` — 原 `frontend/models.py`，`ReportRecord` dataclass

**公共接口（保持不变）：**
```python
from alphaquant.infrastructure.persistence import DB, ReportRecord

db = DB(path="./data/reports.db")
db.init()
record_id = db.insert_report("AAPL", investment_report)
records = db.get_history(tickers=["AAPL"], since=...)
```

### 2. `infrastructure/`（新顶层包）

包含 4 个子模块：
- `config.py`（从顶层移入）— `pydantic-settings` 单例
- `llm.py`（从顶层移入）— LiteLLM 客户端封装
- `data_sources/`（从顶层移入）— 5 个数据源适配器 + `DataSource` 抽象基类
- `persistence/`（新，见上）

### 3. `interfaces/`（新顶层包）

包含 3 个子模块：
- `cli.py`（从顶层移入）— argparse CLI 入口
- `api/`（从顶层移入）— FastAPI 路由层（routes/schemas/rate_limiter）
- `frontend/`（从顶层移入）— Streamlit UI（移入时同时**移除** `db.py` 和 `models.py`）

### 4. `docs/ARCHITECTURE.md`（新文档）

4 节内容：
1. **Overview** — 一段话讲项目做什么、谁是用户
2. **Layered Architecture** — 4 层图（领域 / 基础设施 / 接口 / 横切）+ 依赖规则
3. **Module Map** — 每个子包一句话职责 + 关键文件列表
4. **Data Flow** — 一次完整分析请求从 CLI/API/frontend 到报告输出的路径

---

## Migration Plan（高层步骤，详细 plan 由 writing-plans skill 生成）

每个步骤独立 commit、独立 review、`uv run pytest tests/ -q` 必须保持 186/186 通过。

### Step 1: `infrastructure/persistence/` 抽出

- 新建 `src/alphaquant/infrastructure/persistence/` 目录
- 移动 `src/alphaquant/frontend/db.py` → `infrastructure/persistence/db.py`
- 移动 `src/alphaquant/frontend/models.py` → `infrastructure/persistence/models.py`
- 更新 6 处 import：
  - `frontend/pages/1_Analyze.py`
  - `frontend/pages/2_History.py`
  - `frontend/pages/3_Compare.py`
  - `frontend/pages/4_Settings.py`
  - `tests/test_db.py`
- 验证：`uv run pytest tests/ -q` → 186 passed

### Step 2: `infrastructure/` 顶层包化

- 新建 `src/alphaquant/infrastructure/`
- 移动 `config.py` → `infrastructure/config.py`
- 移动 `llm.py` → `infrastructure/llm.py`
- 移动 `data_sources/` → `infrastructure/data_sources/`
- 更新所有 `from alphaquant.config`、`from alphaquant.llm`、`from alphaquant.data_sources` 的 import
- 验证：`uv run pytest tests/ -q` → 186 passed

### Step 3: `interfaces/` 顶层包化

- 新建 `src/alphaquant/interfaces/`
- 移动 `cli.py` → `interfaces/cli.py`
- 移动 `api/` → `interfaces/api/`
- 移动 `frontend/` → `interfaces/frontend/`
- 更新 `__main__.py` 和 `main.py` 中的 import
- 验证：
  - `uv run pytest tests/ -q` → 186 passed
  - `uv run python -m alphaquant AAPL` → 仍可工作
  - `uv run streamlit run src/alphaquant/interfaces/frontend/app.py` → 仍可启动

### Step 4: `docs/ARCHITECTURE.md`

- 新建 `docs/ARCHITECTURE.md`（4 节，见 Components #4）
- 更新 `README.md` 的 "Architecture" 章节指向新文档

---

## Acceptance Criteria

重构完成的判断标准：

1. ✅ `src/alphaquant/` 顶层仅剩 4 个 `.py` 文件：`__init__.py`、`__main__.py`、`core.py`、`exceptions.py`
2. ✅ `src/alphaquant/frontend/db.py` 不再存在（已搬到 `infrastructure/persistence/`）
3. ✅ `src/alphaquant/frontend/models.py` 不再存在（已搬到 `infrastructure/persistence/`）
4. ✅ 4 个新的子包就位：`infrastructure/`（含 4 个子模块）、`interfaces/`（含 3 个子模块）
5. ✅ `uv run pytest tests/ -q` 仍报 186 passed
6. ✅ `python -m alphaquant AAPL` 仍可工作（CLI 入口未坏）
7. ✅ `uvicorn alphaquant.main:app` 仍可启动（FastAPI 入口未坏）
8. ✅ `uv run streamlit run src/alphaquant/interfaces/frontend/app.py` 仍可启动（前端入口路径变了）
9. ✅ `docs/ARCHITECTURE.md` 存在，包含 4 节内容
10. ✅ `README.md` 更新了 `architecture` 章节链接到新文档
11. ✅ `git log` 显示 4 个独立 commit（每个 Step 一个）

---

## Error Handling

本重构是结构性改动，**不改异常类型或错误处理逻辑**。但需要确保：
- 所有 import 路径更新到位（用 `grep -r "from alphaquant\."` 验证没有遗漏）
- `tests/test_db.py` 的 import 必须指向新位置
- 测试运行后还要做一次 `python -c "import alphaquant; import alphaquant.main; import alphaquant.frontend"` 验证模块链可加载

---

## Testing

不写新测试。现有 186 个测试必须保持 100% 通过。

每 Step 验证命令：
```bash
uv run pytest tests/ -q   # 必须 186 passed
```

外加（只在 Step 3 完成后跑一次）：
```bash
uv run python -c "import alphaquant.main; import alphaquant.interfaces.frontend.app; print('imports OK')"
```

---

## Documentation

仅 1 个新文档：`docs/ARCHITECTURE.md`（见 Components #4）。

README 更新：在现有 "Architecture" 段后加一行 `See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full module map.`

---

## Risks

| 风险 | 缓解 |
| --- | --- |
| 遗漏 import 更新导致运行时 `ImportError` | 每 Step 完成后跑 `pytest -q` + grep `from alphaquant\.` 检查 |
| `__main__.py` 引用 `alphaquant.cli.main`，移动后破坏 CLI 入口 | Step 3 必跑 `python -m alphaquant AAPL` 冒烟测试 |
| Streamlit 入口路径从 `src/alphaquant/frontend/app.py` 变成 `src/alphaquant/interfaces/frontend/app.py`，旧文档/脚本失效 | README/Dockerfile/CI（如有）同步更新；Dockerfile 用相对路径也要检查 |
| 用户的工作区还引用旧路径 | 推送后用户在 GitHub 看到 README 已更新；本地 pull 后用 `find` 检查 |

---

## Open Questions

无。设计已与用户对齐。