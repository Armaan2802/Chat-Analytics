import pandas as pd
from langchain.llms import OpenAI

df = pd.read_csv('your csv data file')
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo",
               openai_api_key=""),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.run("Add your query here")
