"""
Test script for Turtlesim Agent.

This script tests the agent locally without requiring langgraph dev.
Run this to test the agent before deploying with langgraph dev.
"""
import asyncio
import os
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
            result = agent.stream({
                "messages": [{"role": "user", "content": user_input}]
            })
            
            print("\n" + "-"*60)
            print("Agent Response:")
            print("-"*60)
            for chunk in result:
                print(chunk, end="|", flush=True)
            print("-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(test_agent())
