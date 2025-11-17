# rag_cli.py

from rag_core import rag_query, rag_query_with_history

def main():
    print("Reinforcement Learning LLM Assistant (CLI)")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        answer = rag_query_with_history(q, history)
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
