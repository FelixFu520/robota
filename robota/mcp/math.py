from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("Math")
mcp_math_path = os.path.join(os.path.dirname(__file__), "math.py")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")