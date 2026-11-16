from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. Build the components
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful study buddy. Explain complex topics clearly using"
            " simple analogies tailored to the user's target audience."
        ),
    ),
    ("user", "Explain {topic} to a {audience} in 3 short sentences."),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
output_parser = StrOutputParser()

# 2. Assemble the chain
chain = prompt | llm | output_parser

# 3. Interactive CLI Loop with Streaming
print("🎓 Smart Study Buddy is ready! (Type 'quit' to exit)\n")

while True:
  topic = input("Enter a topic to learn: ")
  if topic.lower() == "quit":
    break

  audience = input("Who is this explanation for? (e.g., 10-year-old, chef): ")
  if audience.lower() == "quit":
    break

  print("\nBuddy: ", end="")

  # Use .stream() instead of .invoke() to print chunks in real time!
  for chunk in chain.stream({"topic": topic, "audience": audience}):
    print(chunk, end="", flush=True)

  print("\n" + "-" * 40 + "\n")