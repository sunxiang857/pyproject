import os
import sys
import logging

from internal.engine.loop import AgentEngine
from internal.provider.mock_provider import MockProvider
from internal.tools.mock_registry import MockRegistry


def main():
    # 让 engine 里的 log.info 直接以简洁格式输出到控制台（与 print 保持同一输出流）
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # 组装：假大脑(mock大模型) + 假手脚(mock工具) + 真心脏(Two-Stage ReAct主循环)
    # 第 4 个参数 enable_thinking=True：开启慢思考（传 False 可对比关闭时的日志）
    p = MockProvider()
    r = MockRegistry()
    eng = AgentEngine(p, r, os.getcwd(), True)
    eng.run("帮我检查当前目录的文件")


if __name__ == "__main__":
    main()
