"""AlphaQuant FastAPI 应用。

共享的分析核心位于 ``alphaquant.core``,以避免与路由模块(也需要调用核心)形成循环导入。
"""
from __future__ import annotations

from fastapi import FastAPI

from alphaquant.interfaces.api.routes import router
from alphaquant.observability import configure_logging, get_logger

# 导入时配置一次结构化日志。structlog 在重复调用时会干净地重新装配全局处理器,因此是幂等的。
configure_logging()
log = get_logger("alphaquant.main")

VERSION = "1.0.0"

app = FastAPI(
    title="AlphaQuant",
    description="AI 投资研究分析师",
    version=VERSION,
)
app.include_router(router, prefix="/api/v1")


# 重新导出共享核心,以便现有的
# ``alphaquant.main.run_analysis`` / ``alphaquant.main.run_analysis_async`` 导入继续工作。
from alphaquant.core import run_analysis, run_analysis_async  # noqa: E402,F401
