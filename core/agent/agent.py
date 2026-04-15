
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ..utils import _make_llm
from ..tools import get_pdb
from .run_output import AgentRunOutput

SYSTEM_PROMPT = """
    You are an expert molecular dynamics scientist, and
    your task is to respond to the question or
    solve the problem to the best of your ability using
    the provided tools.

    You can only respond with a single complete
    'Thought, Action, Action Input, Final Answer' format

    Complete format:
    Thought: (reflect on your progress and decide what to do next)
    Action:
    ```
    {{
        "action": (the action name, it should be the name of a tool),
        "action_input": (the input string for the action)
    }}
    ```
    Final Answer: (the final response to the original input
    question, once all steps are complete)

    You are required to use the tools provided,
    using the most specific tool
    available for each action.
    Your final answer should contain all information
    necessary to answer the question and its subquestions.
    Before you finish, reflect on your progress and make
    sure you have addressed the question in its entirety.

    If you are asked to continue
    or reference previous runs,
    the context will be provided to you.
    If context is provided, you should assume
    you are continuing a chat.

    The user's request is in the following human message(s)."""

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "mdcrow-core-1"}}

class MDCrow:
    def __init__(self, model = "jinx-gpt-oss-20b", temp=0.1):
        self.llm = _make_llm(model, temp)
        self.agent = create_agent(
            self.llm,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
            tools=[get_pdb],
            )

    def run(self, user_input):
        output = self.agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config)
        return AgentRunOutput(output)
