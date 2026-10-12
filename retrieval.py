import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if not chunks:
            raise ValueError("No chunks were created.")
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.vectors = self.model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, question, top_k=3):
        question_vector = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.vectors @ question_vector
        best = np.argsort(scores)[::-1][:top_k]
        return [
            {"rank": rank, "chunk_id": int(idx), "score": float(scores[idx]),
             "text": self.chunks[idx]}
            for rank, idx in enumerate(best, start=1)
        ]
