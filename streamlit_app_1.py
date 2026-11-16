from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st

# 1. Load environment variables (OpenAI + LangSmith keys)
load_dotenv()

# 2. Page Setup
st.set_page_config(page_title="Smart Study Buddy", page_icon="🎓")
st.title("🎓 Smart Study Buddy")
st.caption("Powered by LangChain, OpenAI, and LangSmith")


# 3. Cache the chain so Streamlit doesn't rebuild it on every button click
@st.cache_resource
def get_chain():
  prompt = ChatPromptTemplate.from_messages([
      (
          "system",
          (
              "You are a helpful study buddy. Explain complex topics clearly"
              " using simple analogies tailored to the user's target audience."
          ),
      ),
      ("user", "Explain {topic} to a {audience} in 3 short sentences."),
  ])
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  output_parser = StrOutputParser()

  return prompt | llm | output_parser


chain = get_chain()

# 4. Build the User Interface (Inputs)
topic = st.text_input("What topic do you want to learn?", placeholder="e.g., Quantum Computing, Inflation, Photosynthesis")
audience = st.text_input("Who is this explanation for?", placeholder="e.g., 10-year-old, chef, gamer")

# 5. Trigger the Chain and Stream Output to the Web Page
if st.button("Explain It!", type="primary"):
  if not topic or not audience:
    st.warning("Please fill in both the topic and the target audience.")
  else:
    st.subheader(f"Explaining {topic} to a {audience}:")
    # st.write_stream() connects directly to LangChain's .stream() generator!
    st.write_stream(chain.stream({"topic": topic, "audience": audience}))