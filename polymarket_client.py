import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import ToolMessage, AIMessage

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model with key
model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

# MCP server parameters
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server_v3.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            resources = await load_mcp_resources(session)
            print("\n Available Tools:")
            for tool in tools:
                print(f"- {tool.name}: {tool.description.split('.')[0]}")

            print("Type 'events' to view already downloaded events")

            agent = create_react_agent(model, tools)

            print("\nType 'quit' to exit.")
            print("\n Chat started")
            while True:
                print("\n __________")
                user_input = input("\nYou: ")
                if user_input.lower() == "quit":
                    print("Exiting chat.")

                    break

                elif user_input.lower() == "events":
                    for r in resources:
                        print(r.data)


                    continue

                result = await agent.ainvoke({"messages": user_input})

                final_response = None
                used_tools = set()

                if isinstance(result, dict) and "messages" in result:
                    for message in result["messages"]:
                        if isinstance(message, ToolMessage):
                            used_tools.add(message.name)
                        elif isinstance(message, AIMessage):
                            tool_calls = message.additional_kwargs.get("tool_calls", [])
                            for call in tool_calls:
                                used_tools.add(call.get("function", {}).get("name"))

                    for message in reversed(result["messages"]):
                        if isinstance(message, AIMessage) and message.content:
                            final_response = message.content
                            break

                if final_response:
                    print("\n🤖 Polymarket bot:")
                    print(final_response)
                else:
                    print("Failed to generate response.")

                if used_tools:
                    print("\nTools used in this response:")
                    for tool_name in used_tools:
                        print(f"- {tool_name}")
                else:
                    print("\nNo preset tools were used in this response.")

if __name__ == "__main__":
    asyncio.run(main())
