from typing import List
import logfire


def _split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split one oversized block of text into pieces of at most `chunk_size` chars.
    Prefers to cut at a line break, then a sentence end, then a space, so we
    avoid slicing words in half. Consecutive pieces share `overlap` chars.
    """
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Look for a natural boundary in the last 30% of the window
            window_start = start + int(chunk_size * 0.7)
            for sep in ("\n", ". ", " "):
                cut = text.rfind(sep, window_start, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        # Step forward, keeping a small overlap (always make progress)
        start = max(end - overlap, start + 1)
    return pieces


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
    """
    Paragraph-aware chunker.
    - Packs paragraphs together up to `chunk_size` characters.
    - FIX: a paragraph longer than `chunk_size` (e.g. HTML/PDF text that has no
      blank lines) is now split further instead of becoming one giant chunk.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        # Normalise Windows line endings so "\r\n\r\n" paragraph breaks work
        paragraphs = text.replace("\r\n", "\n").split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # Oversized paragraph: flush what we have, then split it on its own
            if len(p) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(_split_long_text(p, chunk_size, overlap))
                continue

            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
