from typing import List, Optional

from internal.provider.interface import LLMProvider
from internal.schema.message import Message, ROLE_ASSISTANT, ToolCall, ToolDefinition


class MockProvider(LLMProvider):
    """假大脑：升级为两阶段感知的 Mock 大模型。

    - 发现 available_tools == None 时，假装在慢思考（Phase 1，输出纯文本规划）
    - 发现传入了工具时，才返回 ToolCall（Phase 2，基于自己写下的计划行动）
    """
    def __init__(self):
        # 只在行动阶段递增，模拟不同轮次返回不同内容
        self.turn = 0

    def generate(self,
                 messages: List[Message],
                 tools: Optional[List[ToolDefinition]]) -> Message:
        # 没有工具可用 => 处于 Phase 1 Thinking 阶段
        if not tools:
            return Message(
                role=ROLE_ASSISTANT,
                content="【推理中】目标是检查文件。我不能直接盲猜，"
                        "我需要先调用 bash 执行 ls 看看目录。",
            )

        # 有工具可用 => Phase 2 行动阶段
        self.turn += 1
        if self.turn == 1:
            # Turn 1：执行 Phase 1 中自己定下的计划，调用 bash ls -la
            return Message(
                role=ROLE_ASSISTANT,
                content="我要执行我刚才计划的步骤了。",
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        name="bash",
                        arguments='{"command": "ls -la"}'
                    ),
                ],
            )

        # Turn 2+：看到工具结果，输出最终回答，终止循环
        return Message(
            role=ROLE_ASSISTANT,
            content="根据工具返回结果，我看到了 main.py，任务完成！",
        )
