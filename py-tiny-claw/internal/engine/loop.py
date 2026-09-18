# internal/engine/loop.py
# 第 3 章：引入"慢思考" (Two-Stage ReAct)

import logging
from typing import List

from internal.provider.interface import LLMProvider
from internal.schema.message import Message, ROLE_SYSTEM, ROLE_USER
from internal.tools.registry import Registry

log = logging.getLogger(__name__)


class AgentEngine:
    """ReAct 引擎升级版：把「谋」与「动」物理分开。

    Phase 1 慢思考 Thinking：不传 tools，纯文本请求，逼迫模型输出深度推理与规划；
    Phase 2 Reason + Action：恢复 tools 列表，模型基于自己的思考去调用工具。
    """
    def __init__(self, provider: LLMProvider, registry: Registry, work_dir: str,
                 enable_thinking: bool):
        self.provider = provider
        self.registry = registry
        self.work_dir = work_dir
        # 慢思考全局开关：
        # 简单任务（如只问天气）可关闭，节省 Token；复杂代码任务则打开
        self.enable_thinking = enable_thinking

    def run(self, user_prompt: str) -> None:
        # ========== 初始化上下文 Context ==========
        context_history: List[Message] = [
            Message(
                role=ROLE_SYSTEM,
                content="You are py-tiny-claw, an expert coding assistant. You have full access to tools in the workspace."
            ),
            Message(
                role=ROLE_USER,
                content=user_prompt
            )
        ]

        turn_count = 0
        while True:
            turn_count += 1
            print(f"\n========== [Turn {turn_count}] 开始 ==========")

            # 获取本轮可用工具列表
            available_tools = self.registry.get_available_tools()

            # ================= Phase 1: 慢思考 Thinking =================
            if self.enable_thinking:
                log.info("[Engine][Phase 1] 剥夺工具访问权，强制慢思考与规划...")
                try:
                    think_resp = self.provider.generate(
                        context_history,
                        None  # 传 None 剥夺工具
                    )
                except Exception as e:
                    raise RuntimeError(f"Thinking 阶段失败: {e}") from e

                # 把思考 Trace 追加进上下文，供 Phase 2 自回归引导
                context_history.append(think_resp)
                if think_resp.content != "":
                    print(f"[内部思考 Trace]: {think_resp.content}")

            # ================= Phase 2: Reason 思考 =================
            log.info("[Engine][Phase 2] 恢复工具挂载，等待模型行动...")
            try:
                action_resp = self.provider.generate(
                    context_history,
                    available_tools
                )
            except Exception as e:
                raise RuntimeError(f"Action 阶段失败: {e}") from e

            context_history.append(action_resp)
            if action_resp.content != "":
                print(f"[对外回复]: {action_resp.content}")

            # ================= 执行判断 =================
            if (action_resp.tool_calls is None) or (len(action_resp.tool_calls) == 0):
                log.info("[Engine] 模型未请求调用工具，任务完成。")
                break

            log.info("[Engine] 模型请求调用 %d 个工具...",
                     len(action_resp.tool_calls))

            # ================= 第三阶段 Act + Observe =================
            for tool_call in action_resp.tool_calls:
                print(f"-> 执行工具: {tool_call.name}, 参数: {tool_call.arguments}")
                result = self.registry.execute(tool_call)

                if result.is_error:
                    print(f"-> 工具执行失败: {result.output}")
                else:
                    print(f"-> 工具执行成功 (返回 {len(result.output)} 字节)")

                # 观察结果包装成 Message 写回 Context，进入下一轮
                observation_msg = Message(
                    role=ROLE_USER,
                    content=result.output,
                    tool_call_id=tool_call.id,
                )
                context_history.append(observation_msg)
