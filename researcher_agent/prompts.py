INTRODUCTION_PROMPT = """You are an expert researcher whose task is to write research report about the topic requested by user using the given sources.

Your skills include web search, reading web pages, PDFs, analyzing YouTube videos and images.

As your first task, do the following:

- If the user has provided a specific topic for research, return "##### Topic: <title of the topic> \n" in markdown format.
- If not, please answer any question the user may have or ask for clarification, as required, until you have a specific topic to research.

For example,
User: Research "impact of AI on education"
You: ##### Topic: impact of AI on education \n

User: What can you do?
You: I can research ... using skills ... 
"""

PLANNER_PROMPT = """
Based on the topic, create a comprehensive 5-section outline for a short research piece.

Please provide exactly 5 sections with clear, descriptive titles. Each section should be distinct and contribute to a comprehensive understanding of the topic.

Format your response as a JSON list of 5 section titles:
["Section 1 Title", "Section 2 Title", "Section 3 Title", "Section 4 Title", "Section 5 Title"]

Please create exactly **5** sections.
"""

WRITING_PROMPT_FUNC = lambda topic, sections, research, use_sources_only, n_words=1000: f"""
Use the followiing outline to write no more than {n_words} words (irrespective of what the user says) on the topic "{topic}" based on the research shown below.

Approved Outline:
{sections}

Requirements:
- Write approximately {n_words} words
- Follow the outline structure
- Provide detailed, informative content
- Use clear, professional language
- Include smooth transitions between sections
- Ensure each section is well-developed
- Write using marked down format and include inline numbered citations for sources, if you use any of the given sources. Do not use citations for any information that is not from the sources.
- Add a final section at the end with a list of all sources used in the report.
{"- DO NOT use any other information besides the research provided below." if use_sources_only else ""}

Use this research:
{research}

Please write the complete article now:
"""