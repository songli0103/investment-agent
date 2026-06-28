"""AlphaQuant 的自定义异常类型。"""


class AlphaQuantError(Exception):
    """所有 AlphaQuant 错误的基类异常。"""


class TickerNotFound(AlphaQuantError):
    """当任何数据源都无法解析 ticker 时抛出。"""

    def __init__(self, ticker: str):
        self.ticker = ticker
        super().__init__(f"未找到 ticker:{ticker}")


class InvalidTickerFormat(AlphaQuantError):
    r"""当 ticker 格式无效时抛出(不符合 [A-Z]{1,5}(\.[A-Z])?)。"""

    def __init__(self, ticker: str):
        self.ticker = ticker
        super().__init__(
            f"无效的 ticker 格式:{ticker!r}。应为 'AAPL' 或 'BRK.B' 之类的格式。"
        )


class AllDataSourcesDown(AlphaQuantError):
    """当查询的所有数据源都失败时抛出。"""


class PartialDataFailure(AlphaQuantError):
    """非致命:某些数据源失败,但我们拥有部分结果。"""

    def __init__(self, message: str, missing_fields: list[str] | None = None):
        self.missing_fields = missing_fields or []
        super().__init__(message)


class ReportGenerationError(AlphaQuantError):
    """最终报告合成步骤失败时抛出。

    根据规范 §3.2,这对应 HTTP 500 INTERNAL_ERROR。
    """


class LLMRateLimited(AlphaQuantError):
    """当上游 LLM 返回 HTTP 429(Token Plan 已用完)时抛出。

    在 FastAPI 路由中映射为 HTTP 503 SERVICE_UNAVAILABLE,以便前端可以
    显示清晰的"稍后重试/升级套餐"消息,而不是不透明的多分钟超时。
    """


class CrewExecutionError(AlphaQuantError):
    """当 CrewAI 因非数据原因执行失败时抛出。

    示例:LLM 响应格式错误(429 后没有 ``choices`` 字段)、工具 schema 验证失败,
    或 CrewAI 内部重试耗尽。与 ``LLMRateLimited`` 不同,因为上游 HTTP 状态
    不一定为 429 —— 失败可能是 LLM 已接受请求后的客户端解析错误。
    """