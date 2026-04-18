# chat.py
from agent import build_agent, ask

WELCOME = """
==================================================
  Economic Indicators AI Agent
  Ask anything about US economic data 2015-2024
  Type 'quit' to exit
==================================================

Example questions you can ask:
  - What was the unemployment rate in 2020?
  - When did inflation peak and what was the value?
  - What happened to GDP during COVID?
  - Compare federal funds rate vs inflation in 2022
  - Which year had the highest housing starts?
"""

if __name__ == "__main__":
    print(WELCOME)
    print("Loading agent...")
    agent = build_agent()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "bye", "goodbye"):
            print("\nAssistant: Goodbye! Feel free to come back anytime.\n")
            break
        ask(agent, question)