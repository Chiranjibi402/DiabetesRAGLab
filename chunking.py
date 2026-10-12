import re


def naive_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Cut the document into fixed-size windows with a small overlap."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Use chunk_size > 0 and 0 <= overlap < chunk_size.")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def structure_aware_chunk(text: str, max_size: int = 1000) -> list[str]:
    """Keep Markdown headings and nearby paragraphs together where possible."""
    if max_size <= 0:
        raise ValueError("max_size must be positive.")

    sections = re.split(r"(?m)(?=^#{1,6}\s)", text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        heading_match = re.match(r"(?m)^#{1,6}\s+.*$", section)
        heading = heading_match.group(0).strip() if heading_match else ""
        body = section[heading_match.end():].strip() if heading_match else section
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        current = [heading] if heading else []

        for paragraph in paragraphs:
            current_text = "\n\n".join(current).strip()
            proposed = "\n\n".join(p for p in (current_text, paragraph) if p).strip()

            if len(proposed) <= max_size:
                current.append(paragraph)
                continue

            has_body = any(part != heading for part in current)
            if has_body:
                chunks.append("\n\n".join(current).strip())
                current = [heading] if heading else []

            prefix = f"{heading}\n\n" if heading else ""
            room = max(1, max_size - len(prefix))
            if len(paragraph) <= room:
                current.append(paragraph)
            else:
                # Fallback for a long paragraph or table.
                for start in range(0, len(paragraph), room):
                    piece = paragraph[start:start + room].strip()
                    if piece:
                        chunks.append(f"{prefix}{piece}".strip())
                current = [heading] if heading else []

        if any(part != heading for part in current):
            chunks.append("\n\n".join(current).strip())

    return chunks
