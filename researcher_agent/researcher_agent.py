import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
load_dotenv()

from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from .prompts import INTRODUCTION_PROMPT, PLANNER_PROMPT, WRITING_PROMPT_FUNC
from .indexer import MultiSourceTextExtractor


# llm = init_chat_model(model="anthropic:claude-3-7-sonnet-latest")
llm = init_chat_model(model="openai:o4-mini")

class SectionPlan(BaseModel):
    """Always use this tool to create a list of sections for the given topic."""
    sections: list = Field(description="List of 5 sections for the given topic")

# Define state
class State(TypedDict):

    topic: str
    messages: Annotated[list, add_messages]
    planned_sections: list
    plan_status: str
    
    research: str
    report: str
    end_execution: bool = False

class Researcher:

    def __init__(self, 
                 researcher="anthropic:claude-3-7-sonnet-latest",
                 planner="anthropic:claude-3-7-sonnet-latest",
                 web_search=False,
                 use_sources_only=True,
                 other_tools=[],
                 additional_sources=[],
                 report_length=200):
        """
        Initialize the Researcher agent with the given parameters.
        :param researcher: The model to use for the researcher agent.
        :param planner: The model to use for the planning stage.
        :param web_search: Whether to enable web search.
        :param other_tools: Additional tools to use in the agent.
        :param additional_sources: Additional sources to use for research.
        :param report_length: Desired length of the report in words.
        """
        self.n_words = report_length

        self.researcher = researcher
        self.planner = planner
        self.web_search = web_search
        self.use_sources_only = use_sources_only
        self.other_tools = other_tools
        self.additional_sources = additional_sources

        self.researcher_model = init_chat_model(model=researcher)
        self.planner_model = init_chat_model(model=planner)
        self.extractor = MultiSourceTextExtractor()
        self.extractor.create_vectorstore_index(additional_sources)

        self.graph = self.build_graph()

    def begin(self, state):

        messages = [SystemMessage(INTRODUCTION_PROMPT)] + state["messages"]
        response = self.planner_model.invoke(messages)
    
        if response.content.lower().strip().startswith("##### topic:"):
            topic = response.content.lower().strip().split("##### topic:")[1]
            return Command(update=dict(topic=topic, end_execution=False))
    
        else:
            return Command(update=dict(topic=None, messages=response.content, end_execution=False))

    def router(self, state) -> Literal["outline", "__end__"]:
        if state.get("topic"):
            return "outline"

        else:
            return Command(goto=END, update=dict(end_execution=True))
        

    def outline(self, state):

        messages = [SystemMessage(PLANNER_PROMPT)] + state["messages"]
    
        llm_section = self.planner_model.bind_tools([SectionPlan])
        response = llm_section.invoke(messages)
    
        sections = response.tool_calls[0]['args']['sections']
        return {"planned_sections": sections}

    def human_review(self, state):

        planned_sections = state["planned_sections"]
    
        planned_sections_text = "\n - ".join([""] + planned_sections)
        verification = interrupt(f"\n\n Does this look like a good outline? {planned_sections_text}\n\n"
                                 "Please respond with 'yes' to approve, 'no' to cancel, or any other text to review again. "
                                 "You can also type 'cancel' to cancel the plan. \n\n"
                                 "**Use the query box on the left to respond and hit 'Submit' to continue.**")
    
        if verification.lower() in ["yes", "y", "proceed", "continue", "aye"]:
            return dict(plan_status="approved")
    
        if verification.lower() in ["cancel"]:
            return dict(plan_status="cancelled")
    
        else:
            return dict(plan_status="in_review")

    def post_review_router(self, state) -> Literal["researcher", "outline", "__end__"]:
        if plan_status := state.get("plan_status") == "approved":
            return "researcher"

        elif plan_status == "in_review":
            return "outline"

        else:
            return END
        
    def research(self, state):
        
        topic = state["topic"]
        sections = state["planned_sections"]
    
        # Create the research
        research = ""
        for section in sections:
            section_research = ""

            if self.additional_sources:
                section_sources = self.extractor.search(query=section, k=3)
                section_research += "\n\n".join([f"Reference: {doc.metadata['title']} ({doc.metadata['source']})\nContent: {doc.page_content}" for doc in section_sources])

            research = f"Section: {section}\n{section_research}\n\n"

        if self.web_search:
            tool = TavilySearch(max_results=5, topic="general")
            query = f"{topic} {section}"
            results = tool.invoke(dict(query=query))['results']

            section_research = "\n\n".join([f"Reference: {result['title']} ({result['url']})\nContent: {result['content']}" for result in results])
            research = f"Section: {section}\n{section_research}\n\n"
    
        return {"research": research}

    def report_writer(self, state):

        topic = state["topic"]
        sections = state["planned_sections"]
        research = state["research"]
    
        writing_prompt = WRITING_PROMPT_FUNC(topic, sections, research, use_sources_only=self.use_sources_only, n_words=self.n_words)

        # Assign tools
        tools = []

        if self.other_tools:
            tools.extend(self.other_tools)

        llm_with_tools = self.researcher_model.bind_tools(tools) if tools else llm
        
        report = llm_with_tools.invoke(writing_prompt)
    
        return {"report": report.content, "end_execution": True}

    def build_graph(self):
        workflow = StateGraph(State)
    
        workflow.add_node("begin", self.begin)
        # workflow.add_node("router", self.router)
        workflow.add_node("outline", self.outline)
        workflow.add_node("researcher", self.research)
        workflow.add_node("report_writer", self.report_writer)
        workflow.add_node("human_review", self.human_review)

        
        workflow.add_edge(START, "begin")
        # workflow.add_edge("begin", "router")
        workflow.add_conditional_edges("begin", self.router)
        workflow.add_edge("outline", "human_review")
        workflow.add_conditional_edges("human_review", self.post_review_router)
        workflow.add_edge("researcher", "report_writer")
        workflow.add_edge("report_writer", END)
        
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)

        return graph
