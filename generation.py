import os
from openai import OpenAI


def answer_with_llm(question, retrieved_chunks, model=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it before using --llm, "
            "or run retrieval-only mode without --llm."
        )

    client = OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    evidence = "\n\n".join(
        f"[Chunk {item['chunk_id']} | similarity {item['score']:.3f}]\n{item['text']}"
        for item in retrieved_chunks
    )

    response = client.responses.create(
        model=model,
        instructions=(
            "You are a careful question-answering assistant for a RAG evaluation. "
            "Answer only using the supplied reference passages. Do not add medical "
            "facts from memory. If the passages do not contain the answer, say: "
            "'The supplied reference does not provide enough information to answer this.' "
            "Keep the answer concise and mention uncertainty when needed."
        ),
        input=f"Question: {question}\n\nRetrieved reference passages:\n{evidence}",
    )
    return response.output_text.strip()
