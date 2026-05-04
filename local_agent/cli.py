from local_agent.runtime import load_chat_agent


def main() -> None:
    agent, _settings = load_chat_agent()

    while True:
        try:
            query = input("Ask: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        response = agent.run(query)
        print("\nAnswer:", response)

