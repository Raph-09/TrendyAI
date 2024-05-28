# TrendyAI Documentation

## Overview
TrendyAI is an application that leverages AI to fetch and present trending topics in the tech domain. It uses the LangChain library, Google Generative AI, and Streamlit for the web interface. This application enables users to input a tech-related topic and get a concise report on the latest trends related to that topic.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/trendyai.git
   cd trendyai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and add the necessary environment variables.

## Configuration

### Environment Variables
Ensure you have the following environment variables set in your `.env` file:

```
GOOGLE_API_KEY=<your_google_api_key>
```

### Style
The application uses a custom CSS file for styling. Ensure `style.css` is in the root directory or update the path in the code accordingly.

## Code Explanation

### Imports

```python
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import find_dotenv, load_dotenv
```

### Environment Setup

```python
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
```

### Large Language Model Initialization

```python
llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        verbose=True,
        temperature=0.4
    )
```

### Search Tool Initialization

```python
search_tool = TavilySearchResults()
```

### Creating Agents

```python
def create_agents(llm, search_tool, topic):
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
```

### Creating Tasks

```python
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
```

### Running the Crew Process

```python
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
```

### Streamlit UI

```python
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
```

## Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Access the app:**
   Open your web browser and go to `http://localhost:8501`.

## Usage

1. Enter a tech-related topic in the input field.
2. Click the "Get what is trending" button.
3. Wait for the AI to process the request.
4. View the results displayed on the page.

## Dependencies

- Python 3.7 or later
- Streamlit
- LangChain
- crewai
- dotenv
- langchain_google_genai
- langchain_community

Ensure you have all dependencies installed as specified in the `requirements.txt` file.

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any inquiries or support, please contact [yourname@domain.com](mailto:yourname@domain.com).
