from typing import List

from internal.tools.registry import Registry
from internal.schema.message import ToolCall, ToolDefinition, ToolResult


class MockRegistry(Registry):
    """假手脚：Mock 工具注册表，只暴露 bash，无真实 IO"""
    def get_available_tools(self) -> List[ToolDefinition]:
        return [ToolDefinition(name="bash")]

    def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            output="-rw-r--r-- 1 user group 234 main.py",
            is_error=False,
        )
