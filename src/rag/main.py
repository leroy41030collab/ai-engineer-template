from src.rag.rag import answer_question, build_rag


def main():
    vector_store = build_rag(force_rebuild=True)

    result = answer_question(
        "Che cosa raccoglie il portale?",
        vector_store,
    )

    print("\n--- RISPOSTA RAG ---")
    print(result["answer"])

    print("\n--- FONTI RECUPERATE ---")
    for source in result["sources"]:
        print(
            f"- {source['filename']} "
            f"(chunk: {source['chunk_index']}, "
            f"distanza: {source['score']})"
        )


if __name__ == "__main__":
    main()