from src.rag.rag import answer_question, build_rag


def main():
    vector_store = build_rag()

    answer = answer_question(
        "Che cosa raccoglie il portale?",
        vector_store,
    )

    print("\\n--- RISPOSTA RAG ---")
    print(answer)


if __name__ == "__main__":
    main()
