"""
CodeAgents are the agents that are used to generate code.
This is in contrast with other frameworks where toolCalling with JSON is used for agents to perform actions.


In a multi-step agent process, the LLM writes and executes actions, typically involving external tool calls. T
raditional approaches use a JSON format to specify tool names and arguments as strings, which the system must parse to determine which tool to execute.

However, research shows that tool-calling LLMs work more effectively with code directly.
This is a core principle of smolagents, as shown in the diagram above from Executable Code Actions Elicit Better LLM Agents.
"""

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

result = agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

print(result)

