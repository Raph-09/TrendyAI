from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        verbose=True,
        temperature=0.4
    )

search_tool = TavilySearchResults()
   

# Create agents
def create_agents(llm, search_tool,topic):
    researcher = Agent(
        role="Senior Tech Researcher",
        goal=f'''Conduct adequate research on {topic}''',
        backstory="""You are an expert at a tech research company, 
        skilled in identifying trends and analyzing complex data.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )
    writer = Agent(
        role='Content Writer',
        goal=f'''write on {topic} and must follow the format of title, intro, trending topics bold and conclusion.
           Do not include the date or year or month. Simply write on the trending topics''',
        backstory="""You are a tech content writer with ability to write a concise report""",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )
    return researcher, writer

# Create tasks
def create_tasks(researcher, writer, topic):
    task1 = Task(
        description= """Search the web for trending topics on {topic}. Analyze posts, discussions and social media threads to get comprehensive
        idea about it""",
        agent=researcher,
        expected_output=f'''Provide a detailed report on trending topics and discussions on {topic}'''
    )

    task2 = Task(
        description=f"""Give a concise summary of the trending topics on {topic} using your insights. 
        Make it clear and go straight to points with little expanation for each points. 
        Provide it like a list example 1,2,3. It must have intro, trending topics and conclusion""",
        agent=writer,
        expected_output='A well refined, creative and concised report'
    )
    return task1, task2

# Run the crew process
def run_crew_process(topic):
    
    researcher, writer = create_agents(llm, search_tool, topic)
    task1, task2 = create_tasks(researcher, writer, topic)
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=2
    )
    result = crew.kickoff()
    return result

# Streamlit UI
def main():
    st.title("TrendyAI")
    st.subheader("Get trending topics in tech using AI")

    with open("style.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>",unsafe_allow_html=True)
    
    topic = st.text_input("Enter any tech related topic:")

    if st.button("Get what is trending"):
        if topic:
            with st.spinner('Geting the topics...'):
                result = run_crew_process(topic)
                st.success("Process completed!")
                st.write("### Result")
                st.write(result)
        else:
            st.error("Enter the topic again")

if __name__ == "__main__":
    main()