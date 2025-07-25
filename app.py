import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt, Command

from researcher_agent import Researcher, MultiSourceTextExtractor

from fast_dash import fastdash, Chat, update, notify, dmc

# Intialize the researcher agent
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
researcher_agent = Researcher(
    researcher="openai:o4-mini",
    planner="openai:o4-mini")

query_component = dmc.Textarea(
    placeholder="Send a message to the agent",
    autosize=True,
    minRows=4,
    required=True,
    description="Send a message to the agent",
)

web_page_urls_component = dmc.TagsInput(
    description="Include all the reference web URLs (HTML, PDF, YouTube, etc.)",
    placeholder="Enter URLs separated by commas",
    value=[]
)

web_search_component = dmc.Switch(
	labelPosition="right",
	label="Use web search",
    description="If checked, the agent will use web search to find additional information",
	size="sm",
	radius="lg",
	color="#5c7cfa",
	disabled=False,
	withThumbIndicator=True,
)

use_sources_only_component = dmc.Switch(
	labelPosition="right",
	label="Use given sources only",
    description="If checked, the agent will only use the provided sources for research",
	size="sm",
	radius="lg",
	color="#5c7cfa",
    checked=True,
	disabled=False,
	withThumbIndicator=True,
)

extended_report_component = dmc.Switch(
	labelPosition="right",
	# label="Use given sources only",
    description="If checked, the agent will generate a detailed report (~1000 words)",
	size="sm",
	radius="lg",
	color="#5c7cfa",
    checked=False,
	disabled=False,
	withThumbIndicator=True,
)


# Build the app
@fastdash(stream=True, loader=False, mode="external", scale_height=1.5)
def researcher(message,
               sources: web_page_urls_component, 
               web_search: web_search_component,
               use_sources_only: use_sources_only_component,
               extended_report: extended_report_component) -> Chat(stream=True, stream_limit=50):

    update('report', message, property="query")

    response = ""

    researcher_agent.web_search = web_search
    researcher_agent.use_sources_only_component = use_sources_only_component
    researcher_agent.extended_report = extended_report
    researcher_agent.extractor.create_vectorstore_index(sources)

    graph = researcher_agent.graph
    agent_state = graph.get_state(config=config)

    if agent_state.interrupts:
        notify("Thanks for the feedback ...")
        trigger = Command(resume=message)

    else:
        trigger = dict(messages=HumanMessage(message))
    
    notify("Thinking ...")
    for mode, chunk in graph.stream(trigger, config=config, stream_mode=["values", "messages", "updates"]):
    
        if mode == "messages":
            chunk, metadata = chunk
                
            if chunk.content and metadata["langgraph_node"] in ["writer", "begin"]:
                response += str(chunk.content)
                update('report', response, property="response")
    
        elif mode == "updates" and "__interrupt__" in chunk:
            response += str(chunk['__interrupt__'][0].value)
            update('report', response, property="response")
            trigger = Command(resume=response)

    report = dict(query=message, response=response)
    return report