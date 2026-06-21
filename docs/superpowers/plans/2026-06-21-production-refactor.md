# AlphaQuant Production Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `src/alphaquant/` 从扁平散落重构为分层包结构（`infrastructure/` + `interfaces/`），让前端层不再持有持久化逻辑，并产出 `docs/ARCHITECTURE.md`。

**Architecture:** 4 个原子任务，每个独立 commit、独立 review、独立可测。
- Task 1：`frontend/db.py` + `frontend/models.py` → `infrastructure/persistence/`
- Task 2：`config.py` + `llm.py` + `data_sources/` → `infrastructure/` 下
- Task 3：`cli.py` + `api/` + `frontend/` → `interfaces/` 下
- Task 4：写 `docs/ARCHITECTURE.md` + 更新 `README.md`

**Tech Stack:** 无变化（仍为 Python 3.11 + uv + pytest 8 + sqlite3 stdlib）。

## Global Constraints

（每条从 spec 直接复制，所有任务都默认遵守）

- 不拆 `flows/analysis_flow.py`（623 行按用户决定保留）
- 不引入 mypy 严格模式 / pre-commit 钩子 / GitHub Actions CI
- 不写 `CONTRIBUTING.md` / `CHANGELOG.md` / 加 `LICENSE`
- 不加徽章、release 自动化、semantic versioning
- 不改业务代码、scoring 算法、LLM prompt、模型 schema
- 不动 `core.py` / `exceptions.py` / `observability/` 的位置和内部代码
- **任何任务完成后 `uv run pytest tests/ -q` 必须仍然 186 passed**
- 每个任务一个独立 commit，commit message 形如 `<type>(scope): <subject>`
- 工作分支：`main`（单开发者项目）；完成后 push 到 `origin/main`

---

## File-by-File Migration Map

| 当前位置 | 目标位置 | 所属 Task |
| --- | --- | --- |
| `src/alphaquant/frontend/db.py` | `src/alphaquant/infrastructure/persistence/db.py` | Task 1 |
| `src/alphaquant/frontend/models.py` | `src/alphaquant/infrastructure/persistence/models.py` | Task 1 |
| `src/alphaquant/config.py` | `src/alphaquant/infrastructure/config.py` | Task 2 |
| `src/alphaquant/llm.py` | `src/alphaquant/infrastructure/llm.py` | Task 2 |
| `src/alphaquant/data_sources/__init__.py` | `src/alphaquant/infrastructure/data_sources/__init__.py` | Task 2 |
| `src/alphaquant/data_sources/base.py` | `src/alphaquant/infrastructure/data_sources/base.py` | Task 2 |
| `src/alphaquant/data_sources/yahoo.py` | `src/alphaquant/infrastructure/data_sources/yahoo.py` | Task 2 |
| `src/alphaquant/data_sources/alpha_vantage.py` | `src/alphaquant/infrastructure/data_sources/alpha_vantage.py` | Task 2 |
| `src/alphaquant/data_sources/finnhub.py` | `src/alphaquant/infrastructure/data_sources/finnhub.py` | Task 2 |
| `src/alphaquant/data_sources/sec_edgar.py` | `src/alphaquant/infrastructure/data_sources/sec_edgar.py` | Task 2 |
| `src/alphaquant/data_sources/news.py` | `src/alphaquant/infrastructure/data_sources/news.py` | Task 2 |
| `src/alphaquant/cli.py` | `src/alphaquant/interfaces/cli.py` | Task 3 |
| `src/alphaquant/api/__init__.py` | `src/alphaquant/interfaces/api/__init__.py` | Task 3 |
| `src/alphaquant/api/routes.py` | `src/alphaquant/interfaces/api/routes.py` | Task 3 |
| `src/alphaquant/api/schemas.py` | `src/alphaquant/interfaces/api/schemas.py` | Task 3 |
| `src/alphaquant/api/rate_limiter.py` | `src/alphaquant/interfaces/api/rate_limiter.py` | Task 3 |
| `src/alphaquant/frontend/app.py` | `src/alphaquant/interfaces/frontend/app.py` | Task 3 |
| `src/alphaquant/frontend/components/__init__.py` | `src/alphaquant/interfaces/frontend/components/__init__.py` | Task 3 |
| `src/alphaquant/frontend/components/charts.py` | `src/alphaquant/interfaces/frontend/components/charts.py` | Task 3 |
| `src/alphaquant/frontend/components/metrics_panel.py` | `src/alphaquant/interfaces/frontend/components/metrics_panel.py` | Task 3 |
| `src/alphaquant/frontend/components/rating_card.py` | `src/alphaquant/interfaces/frontend/components/rating_card.py` | Task 3 |
| `src/alphaquant/frontend/pages/__init__.py` | `src/alphaquant/interfaces/frontend/pages/__init__.py` | Task 3 |
| `src/alphaquant/frontend/pages/1_Analyze.py` | `src/alphaquant/interfaces/frontend/pages/1_Analyze.py` | Task 3 |
| `src/alphaquant/frontend/pages/2_History.py` | `src/alphaquant/interfaces/frontend/pages/2_History.py` | Task 3 |
| `src/alphaquant/frontend/pages/3_Compare.py` | `src/alphaquant/interfaces/frontend/pages/3_Compare.py` | Task 3 |
| `src/alphaquant/frontend/pages/4_Settings.py` | `src/alphaquant/interfaces/frontend/pages/4_Settings.py` | Task 3 |
| （新文件）`src/alphaquant/infrastructure/__init__.py` | （空包标记） | Task 2 |
| （新文件）`src/alphaquant/infrastructure/persistence/__init__.py` | （暴露 `DB` 和 `ReportRecord`） | Task 1 |
| （新文件）`src/alphaquant/interfaces/__init__.py` | （空包标记） | Task 3 |
| （新文件）`docs/ARCHITECTURE.md` | 4 节架构文档 | Task 4 |
| `README.md` | 添加 ARCHITECTURE.md 链接 | Task 4 |

---

## Task 1: Move persistence layer out of frontend/

**Files:**
- Create: `src/alphaquant/infrastructure/__init__.py`（空文件）
- Create: `src/alphaquant/infrastructure/persistence/__init__.py`（导出 `DB` 和 `ReportRecord`）
- Move: `src/alphaquant/frontend/db.py` → `src/alphaquant/infrastructure/persistence/db.py`
- Move: `src/alphaquant/frontend/models.py` → `src/alphaquant/infrastructure/persistence/models.py`
- Delete: `src/alphaquant/frontend/db.py`（move 后删除原文件）
- Delete: `src/alphaquant/frontend/models.py`（move 后删除原文件）
- Modify: `src/alphaquant/infrastructure/persistence/db.py:10`（内部 import 改新路径）
- Modify: `src/alphaquant/infrastructure/persistence/models.py`（如有内部 import）
- Modify: `src/alphaquant/frontend/pages/1_Analyze.py:19`（`from alphaquant.frontend.db` → `from alphaquant.infrastructure.persistence`）
- Modify: `src/alphaquant/frontend/pages/2_History.py:10, 75`（同上）
- Modify: `src/alphaquant/frontend/pages/3_Compare.py:32`（同上）
- Modify: `src/alphaquant/frontend/pages/4_Settings.py:13`（同上）
- Modify: `src/alphaquant/frontend/components/charts.py:8`（`from alphaquant.frontend.models` → `from alphaquant.infrastructure.persistence`）
- Modify: `tests/test_db.py:10-11`（同上）
- Test: `uv run pytest tests/ -q` 必须仍 186 passed

**Interfaces:**
- Consumes: 现有 `DB` 类（8 方法）和 `ReportRecord` dataclass
- Produces: 新公共接口 `alphaquant.infrastructure.persistence.DB` 和 `alphaquant.infrastructure.persistence.ReportRecord`（与原 `alphaquant.frontend.db.DB` / `alphaquant.frontend.models.ReportRecord` 等价）

- [ ] **Step 1: 创建新包骨架**

```bash
mkdir -p src/alphaquant/infrastructure/persistence
touch src/alphaquant/infrastructure/__init__.py
touch src/alphaquant/infrastructure/persistence/__init__.py
```

预期：无输出，目录创建成功。

- [ ] **Step 2: 移动 db.py 和 models.py**

```bash
git mv src/alphaquant/frontend/db.py src/alphaquant/infrastructure/persistence/db.py
git mv src/alphaquant/frontend/models.py src/alphaquant/infrastructure/persistence/models.py
```

预期：git 显示两个文件 rename（`R` 状态），无内容变更。

- [ ] **Step 3: 更新 db.py 内部 import**

修改 `src/alphaquant/infrastructure/persistence/db.py` 第 10 行：

- 旧：`from alphaquant.frontend.models import ReportRecord`
- 新：`from alphaquant.infrastructure.persistence.models import ReportRecord`

- [ ] **Step 4: 让 `__init__.py` 导出公共接口**

编辑 `src/alphaquant/infrastructure/persistence/__init__.py`：

```python
"""SQLite persistence layer for AlphaQuant report history."""
from alphaquant.infrastructure.persistence.db import DB
from alphaquant.infrastructure.persistence.models import ReportRecord

__all__ = ["DB", "ReportRecord"]
```

- [ ] **Step 5: 更新 4 个 page 文件的 import**

对每个文件，把 `from alphaquant.frontend.db import DB` 改为 `from alphaquant.infrastructure.persistence import DB`：

- `src/alphaquant/frontend/pages/1_Analyze.py`（line 19）
- `src/alphaquant/frontend/pages/2_History.py`（line 10）
- `src/alphaquant/frontend/pages/3_Compare.py`（line 32）
- `src/alphaquant/frontend/pages/4_Settings.py`（line 13）

- [ ] **Step 6: 更新 History.py 第 75 行的 ReportRecord import**

`src/alphaquant/frontend/pages/2_History.py` 第 75 行：

- 旧：`from alphaquant.frontend.models import ReportRecord  # local import: avoid cycles in tests`
- 新：`from alphaquant.infrastructure.persistence import ReportRecord  # local import: avoid cycles in tests`

- [ ] **Step 7: 更新 charts.py 的 ReportRecord import**

`src/alphaquant/frontend/components/charts.py` 第 8 行：

- 旧：`from alphaquant.frontend.models import ReportRecord`
- 新：`from alphaquant.infrastructure.persistence import ReportRecord`

- [ ] **Step 8: 更新 test_db.py 的 import**

`tests/test_db.py` 第 10-11 行：

- 旧：
  ```python
  from alphaquant.frontend.db import DB
  from alphaquant.frontend.models import ReportRecord
  ```
- 新：
  ```python
  from alphaquant.infrastructure.persistence import DB, ReportRecord
  ```

- [ ] **Step 9: 验证测试通过**

```bash
uv run pytest tests/ -q
```

预期：`186 passed, 22 warnings in ~5s`

- [ ] **Step 10: 验证无残留 import**

```bash
grep -rn "from alphaquant.frontend.db\|from alphaquant.frontend.models" src/ tests/ || echo "CLEAN"
```

预期输出：`CLEAN`

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(persistence): move DB layer out of frontend/ into infrastructure/persistence/

Frontend now imports DB and ReportRecord from
alphaquant.infrastructure.persistence. No business logic changed;
db.py and models.py moved verbatim with only their internal
import updated to the new location.

Tests: 186/186 passing.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Move config/llm/data_sources into infrastructure/

**Files:**
- Move: `src/alphaquant/config.py` → `src/alphaquant/infrastructure/config.py`
- Move: `src/alphaquant/llm.py` → `src/alphaquant/infrastructure/llm.py`
- Move: `src/alphaquant/data_sources/*.py` (6 files) → `src/alphaquant/infrastructure/data_sources/*.py`
- Modify: `src/alphaquant/infrastructure/llm.py:7`（`from alphaquant.config` → `from alphaquant.infrastructure.config`）
- Modify: `src/alphaquant/infrastructure/data_sources/*.py`（5 个数据源文件 + `__init__.py` + `base.py`，所有内部 import 改前缀）
- Modify: 所有外部调用方（约 20 处 import）：
  - `src/alphaquant/observability/logger.py:9`
  - `src/alphaquant/observability/cost_tracker.py:8`
  - `src/alphaquant/agents/company_resolver.py:6`
  - `src/alphaquant/agents/competitor_analyst.py:6`
  - `src/alphaquant/agents/financial_analyst.py:6`
  - `src/alphaquant/agents/news_analyst.py:6`
  - `src/alphaquant/agents/market_analyst.py:6`
  - `src/alphaquant/agents/risk_analyst.py:6`
  - `src/alphaquant/agents/valuation_analyst.py:6`
  - `src/alphaquant/agents/report_writer.py:6`
  - `src/alphaquant/flows/analysis_flow.py:13`
  - `src/alphaquant/tools/market_data_tool.py:7`
  - `src/alphaquant/tools/competitor_tool.py:6, 16`
  - `src/alphaquant/tools/financial_tool.py:6`
  - `src/alphaquant/tools/news_tool.py:6`
- Test: `uv run pytest tests/ -q` 必须仍 186 passed

**Interfaces:**
- Consumes: Task 1 产出的 `alphaquant.infrastructure.persistence.*`（不直接用，但路径前缀对齐）
- Produces: 新公共接口：
  - `alphaquant.infrastructure.config.Settings`, `alphaquant.infrastructure.config.get_settings`
  - `alphaquant.infrastructure.llm.get_llm`
  - `alphaquant.infrastructure.data_sources.DataSourceRegistry`（及 `DataSourceInterface` 等所有原导出）
  - `alphaquant.infrastructure.data_sources.{Yahoo,AlphaVantage,Finnhub,SECEdgar,NewsAPI}Source`

- [ ] **Step 1: 移动 config.py**

```bash
git mv src/alphaquant/config.py src/alphaquant/infrastructure/config.py
```

- [ ] **Step 2: 移动 llm.py**

```bash
git mv src/alphaquant/llm.py src/alphaquant/infrastructure/llm.py
```

- [ ] **Step 3: 更新 llm.py 内部 import**

`src/alphaquant/infrastructure/llm.py` 第 7 行：

- 旧：`from alphaquant.config import Settings, get_settings`
- 新：`from alphaquant.infrastructure.config import Settings, get_settings`

- [ ] **Step 4: 移动 data_sources 整个目录**

```bash
git mv src/alphaquant/data_sources src/alphaquant/infrastructure/data_sources
```

预期：git 自动检测到目录重命名，所有内部文件变成 rename（`R` 状态）。

- [ ] **Step 5: 更新 data_sources 内部 import（7 个文件）**

对每个文件，把 `from alphaquant.data_sources.X` 改为 `from alphaquant.infrastructure.data_sources.X`：

- `src/alphaquant/infrastructure/data_sources/__init__.py:6-11`（6 处）
- `src/alphaquant/infrastructure/data_sources/base.py`（如有）
- `src/alphaquant/infrastructure/data_sources/yahoo.py:9`
- `src/alphaquant/infrastructure/data_sources/alpha_vantage.py:8-9`（2 处）
- `src/alphaquant/infrastructure/data_sources/finnhub.py:7-8`（2 处）
- `src/alphaquant/infrastructure/data_sources/sec_edgar.py:9`
- `src/alphaquant/infrastructure/data_sources/news.py:8-9`（2 处）

- [ ] **Step 6: 更新 observability 的 config import（2 个文件）**

- `src/alphaquant/observability/logger.py:9`：`from alphaquant.config import get_settings` → `from alphaquant.infrastructure.config import get_settings`
- `src/alphaquant/observability/cost_tracker.py:8`：同上

- [ ] **Step 7: 更新 agents 的 llm import（8 个文件）**

对每个 agent 文件，把 `from alphaquant.llm import get_llm` 改为 `from alphaquant.infrastructure.llm import get_llm`：

- `src/alphaquant/agents/company_resolver.py:6`
- `src/alphaquant/agents/competitor_analyst.py:6`
- `src/alphaquant/agents/financial_analyst.py:6`
- `src/alphaquant/agents/news_analyst.py:6`
- `src/alphaquant/agents/market_analyst.py:6`
- `src/alphaquant/agents/risk_analyst.py:6`
- `src/alphaquant/agents/valuation_analyst.py:6`
- `src/alphaquant/agents/report_writer.py:6`

- [ ] **Step 8: 更新 flows 的 data_sources import**

`src/alphaquant/flows/analysis_flow.py:13`：

- 旧：`from alphaquant.data_sources import DataSourceRegistry`
- 新：`from alphaquant.infrastructure.data_sources import DataSourceRegistry`

- [ ] **Step 9: 更新 tools 的 data_sources import（4 个文件）**

- `src/alphaquant/tools/market_data_tool.py:7`：`from alphaquant.data_sources` → `from alphaquant.infrastructure.data_sources`
- `src/alphaquant/tools/competitor_tool.py:6, 16`：2 处同上
- `src/alphaquant/tools/financial_tool.py:6`：同上
- `src/alphaquant/tools/news_tool.py:6`：同上

- [ ] **Step 10: 验证测试通过**

```bash
uv run pytest tests/ -q
```

预期：`186 passed, 22 warnings in ~5s`

- [ ] **Step 11: 验证无残留 import**

```bash
grep -rn "from alphaquant\.config\|from alphaquant\.llm\|from alphaquant\.data_sources" src/ tests/ | grep -v "infrastructure\." || echo "CLEAN"
```

预期输出：`CLEAN`

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(infra): move config/llm/data_sources under infrastructure/

Groups external resource adapters (LLM provider, data sources,
pydantic-settings) under alphaquant.infrastructure/ to mirror
Task 1's persistence move. Top-level src/alphaquant/ now contains
only __init__, __main__, core, and exceptions.

Tests: 186/186 passing.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Move cli/api/frontend into interfaces/

**Files:**
- Move: `src/alphaquant/cli.py` → `src/alphaquant/interfaces/cli.py`
- Move: `src/alphaquant/api/*.py` (4 files + __init__) → `src/alphaquant/interfaces/api/*.py`
- Move: `src/alphaquant/frontend/*.py` (含 components/, pages/) → `src/alphaquant/interfaces/frontend/*.py`
- Create: `src/alphaquant/interfaces/__init__.py`（空）
- Modify: `src/alphaquant/interfaces/api/routes.py:8-9`（内部 import）
- Modify: `src/alphaquant/interfaces/api/rate_limiter.py`（如有内部 import）
- Modify: `src/alphaquant/__main__.py:1`（`from alphaquant.cli` → `from alphaquant.interfaces.cli`）
- Modify: `src/alphaquant/main.py:10`（`from alphaquant.api.routes` → `from alphaquant.interfaces.api.routes`）
- Modify: `src/alphaquant/interfaces/frontend/pages/4_Settings.py:75`（`from alphaquant.config` → `from alphaquant.infrastructure.config`）
- Modify: `tests/test_api.py`（约 12 处 import 改新路径）
- Test: `uv run pytest tests/ -q` 必须仍 186 passed
- Smoke: `uv run python -c "import alphaquant.main; import alphaquant.interfaces.frontend.app; print('imports OK')"` 必须成功

**Interfaces:**
- Consumes: Task 1+2 产出的所有 `alphaquant.infrastructure.*` 模块
- Produces: 新公共接口：
  - `alphaquant.interfaces.cli.main`（被 `__main__.py` 调用）
  - `alphaquant.interfaces.api.router`（被 `main.py` 挂载到 `/api/v1`）
  - `alphaquant.interfaces.api.{routes,schemas,rate_limiter}.*`
  - `alphaquant.interfaces.frontend.app`（Streamlit 入口，新路径：`src/alphaquant/interfaces/frontend/app.py`）

- [ ] **Step 1: 创建 interfaces 包骨架**

```bash
mkdir -p src/alphaquant/interfaces
touch src/alphaquant/interfaces/__init__.py
```

- [ ] **Step 2: 移动 cli.py**

```bash
git mv src/alphaquant/cli.py src/alphaquant/interfaces/cli.py
```

- [ ] **Step 3: 移动 api/ 整个目录**

```bash
git mv src/alphaquant/api src/alphaquant/interfaces/api
```

预期：git 自动检测目录重命名。

- [ ] **Step 4: 更新 api 内部 import**

`src/alphaquant/interfaces/api/routes.py`：

- 第 8 行：`from alphaquant.api.rate_limiter` → `from alphaquant.interfaces.api.rate_limiter`
- 第 9 行：`from alphaquant.api.schemas` → `from alphaquant.interfaces.api.schemas`

`src/alphaquant/interfaces/api/rate_limiter.py`（如有内部 import）：检查并改为 `alphaquant.interfaces.api.*`。

- [ ] **Step 5: 移动 frontend/ 整个目录**

```bash
git mv src/alphaquant/frontend src/alphaquant/interfaces/frontend
```

预期：git 自动检测目录重命名。`db.py` 和 `models.py` 在 Task 1 已删除，所以这里只有 `app.py`、`components/`、`pages/`。

- [ ] **Step 6: 更新 `__main__.py` 的 import**

`src/alphaquant/__main__.py` 第 1 行：

- 旧：`from alphaquant.cli import main`
- 新：`from alphaquant.interfaces.cli import main`

- [ ] **Step 7: 更新 `main.py` 的 import**

`src/alphaquant/main.py` 第 10 行：

- 旧：`from alphaquant.api.routes import router`
- 新：`from alphaquant.interfaces.api.routes import router`

- [ ] **Step 8: 更新 Settings.py 的 config import**

`src/alphaquant/interfaces/frontend/pages/4_Settings.py` 第 75 行（在 `try` 块内）：

- 旧：`from alphaquant.config import get_settings  # type: ignore[import-not-found]`
- 新：`from alphaquant.infrastructure.config import get_settings  # type: ignore[import-not-found]`

- [ ] **Step 9: 更新 tests/test_api.py 的 import（约 12 处）**

把文件中所有：
- `from alphaquant.api.schemas` → `from alphaquant.interfaces.api.schemas`（line 29 起的 import 块）
- `from alphaquant.api import rate_limiter` → `from alphaquant.interfaces.api import rate_limiter`（lines 193, 208, 219, 230, 248, 264, 280, 295）
- `from alphaquant.api.rate_limiter import TokenBucketRateLimiter` → `from alphaquant.interfaces.api.rate_limiter import TokenBucketRateLimiter`（line 311）
- `from alphaquant.cli import main` → `from alphaquant.interfaces.cli import main`（lines 398, 499）

- [ ] **Step 10: 验证测试通过**

```bash
uv run pytest tests/ -q
```

预期：`186 passed, 22 warnings in ~5s`

- [ ] **Step 11: 验证入口可加载**

```bash
uv run python -c "import alphaquant.main; import alphaquant.interfaces.frontend.app; print('imports OK')"
```

预期输出：`imports OK`

- [ ] **Step 12: 验证 CLI 入口仍可工作**

```bash
uv run python -m alphaquant AAPL --format text 2>&1 | head -5
```

预期：输出分析报告头部（API key 占位符会失败数据获取，但 CLI 框架本身应正常启动并显示错误信息；或者成功分析）。

如果 API key 无效导致失败是正常的——只要不报 `ImportError` 或 `ModuleNotFoundError` 即可。

- [ ] **Step 13: 验证无残留 import**

```bash
grep -rn "from alphaquant\.api\|from alphaquant\.cli\|from alphaquant\.frontend" src/ tests/ | grep -v "interfaces\." || echo "CLEAN"
```

预期输出：`CLEAN`

- [ ] **Step 14: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(interface): move cli/api/frontend under interfaces/

Groups user-facing entry points (argparse CLI, FastAPI routes,
Streamlit UI) under alphaquant.interfaces/. Frontend/ no longer
holds any data persistence logic (db.py and models.py were removed
in Task 1's persistence move).

Public entry points changed:
- python -m alphaquant (unchanged, still works via __main__.py)
- uvicorn alphaquant.main:app (unchanged, still mounts the same router)
- streamlit: now requires the new path
    uv run streamlit run src/alphaquant/interfaces/frontend/app.py

Tests: 186/186 passing. CLI + FastAPI + Streamlit imports all load.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Write docs/ARCHITECTURE.md and update README

**Files:**
- Create: `docs/ARCHITECTURE.md`（4 节内容，见下方代码块）
- Modify: `README.md`（在现有 "Architecture" 段后追加一行链接）

**Interfaces:**
- Consumes: 已重构的 `src/alphaquant/` 目录结构
- Produces: 新文档 `docs/ARCHITECTURE.md`，README 中加一行链接

- [ ] **Step 1: 创建 ARCHITECTURE.md**

```bash
mkdir -p docs
```

然后用 Write 工具创建 `docs/ARCHITECTURE.md`，内容如下：

```markdown
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
```

- [ ] **Step 2: 更新 README.md 引用新文档**

在 `README.md` 现有 "Architecture" 段后插入一行：

```markdown
See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full module map and data flow.
```

定位：在 `README.md` 第 67 行后（即 "Data sources (fallback chain)..." 这一段之前），插入这段链接。

如果现有 "Architecture" 段已经包含简短描述，保留它，新加的这行作为"详细看这里"的链接。

- [ ] **Step 3: 验证 Markdown 链接渲染**

```bash
grep -n "ARCHITECTURE.md" README.md
```

预期：至少 1 个匹配。

- [ ] **Step 4: 验证文档结构**

```bash
test -f docs/ARCHITECTURE.md && wc -l docs/ARCHITECTURE.md
```

预期：文件存在，行数 > 80。

- [ ] **Step 5: 验证整体测试仍通过（最后一次确认）**

```bash
uv run pytest tests/ -q
```

预期：`186 passed, 22 warnings in ~5s`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs(arch): add ARCHITECTURE.md and link from README

ARCHITECTURE.md gives new contributors a 4-layer map
(interfaces / core / domain / infrastructure), a module
reference, and the end-to-end data flow for one analysis
request. README links to it from the existing Architecture
section.

Final test run after the 3 refactor commits: 186/186 passing.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: 推送到 origin/main**

```bash
git push origin main
```

预期：`To github.com:songli0103/investment-agent.git` + 4 commits ahead 推送成功。

---

## Self-Review Checklist

完成后验证（一次性跑完）：

```bash
# 1. 顶层只剩 4 个 .py
ls src/alphaquant/*.py
# 预期：__init__.py  __main__.py  core.py  exceptions.py

# 2. 前端不再有 db.py / models.py
ls src/alphaquant/frontend/ 2>&1 | grep -E "db\.py|models\.py" || echo "FRONTEND_CLEAN"
# 预期：FRONTEND_CLEAN（说明 frontend/ 目录应该已不存在，所有内容在 interfaces/frontend/）

# 3. 新子包就位
test -d src/alphaquant/infrastructure && echo "infra OK"
test -d src/alphaquant/interfaces && echo "ifaces OK"
# 预期：infra OK  ifaces OK

# 4. 测试
uv run pytest tests/ -q
# 预期：186 passed

# 5. 文档
test -f docs/ARCHITECTURE.md && echo "doc OK"
grep -q "ARCHITECTURE.md" README.md && echo "readme linked"
# 预期：doc OK  readme linked

# 6. 4 个 commit
git log --oneline -4
# 预期：4 个 refactor/docs commit
```

---

## Out of Scope（不要做）

- 不动 `flows/analysis_flow.py`（用户明确不拆）
- 不加 mypy / pre-commit / GitHub Actions / LICENSE / CHANGELOG（用户没选）
- 不改任何业务代码、scoring 算法、prompt、模型 schema
- 不动 `core.py` / `exceptions.py` / `observability/` 的位置和内容
- 不重写 README 其他章节（只加一行链接）
- 不动 Docker / docker-compose（路径变更不影响容器）