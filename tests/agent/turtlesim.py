"""
Test script for Turtlesim Agent.

This script tests the agent locally without requiring langgraph dev.
Run this to test the agent before deploying with langgraph dev.
"""
import asyncio
import os
import time
from langchain_mcp_adapters.client import MultiServerMCPClient

from robota.agent.turtlesim import TurtlesimAgent
from robota.mcp.math import mcp_math_path

async def _get_tools():
    mcp_client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [mcp_math_path],
        },
    })
    return await mcp_client.get_tools()

async def test_agent():
    tools = await _get_tools()
    agent = TurtlesimAgent(tools=tools)

    print("\n" + "="*60)
    print("🤖 Turtlesim Agent is ready!")
    print("="*60)
    print("\nYou can now interact with the agent.")
    print("Type 'exit' to quit.\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Agent thinking...")
            print("\n" + "-"*60)
            print("Agent Response:")
            print("-"*60)
            time_start = time.time()
            Time_flag = True
            # Stream with messages mode to get real-time token output
            for token, metadata in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="messages",
            ):
                print("|", end="|", flush=True)

                time_end = time.time()
                if Time_flag:
                    print(f"Time: {time_end - time_start:.2f} seconds", end="", flush=True)
                    Time_flag = False
                # Extract text content from token
                # Token can be AIMessageChunk with content_blocks or text attribute
                if hasattr(token, 'content_blocks') and token.content_blocks:
                    for block in token.content_blocks:
                        if block.get("type") == "text" and block.get("text"):
                            print(block["text"], end="|", flush=True)
                        elif block.get("type") == "tool_call_chunk":
                            # Optionally show tool calls being made
                            if block.get("name"):
                                print(f"\n[Calling tool: {block['name']}]", flush=True)
                elif hasattr(token, 'text') and token.text:
                    # Fallback: if token has direct text attribute
                    print(token.text, end="|", flush=True)
            
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(test_agent())
