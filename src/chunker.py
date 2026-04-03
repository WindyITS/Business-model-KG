import re
from typing import Callable, Iterable, List

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency is optional at runtime.
    tiktoken = None


ITEM_HEADER_RE = re.compile(r"^(ITEM\s+\d+[A-Z]?\.?|PART\s+[IVX]+)$", re.IGNORECASE)
HEADING_TOKEN_LIMIT = 12


def _normalize_filing_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"(?i)(PART\s+[IVX]+)\s*(Item\s+\d+[A-Z]?)", r"\1\n\2", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_heading(line: str) -> bool:
    cleaned = line.strip(" :\t")
    if not cleaned:
        return False
    if ITEM_HEADER_RE.match(cleaned):
        return True
    if cleaned.startswith("•"):
        return False

    words = cleaned.split()
    if len(words) > HEADING_TOKEN_LIMIT:
        return False
    if cleaned.endswith((".", ";", ":")):
        return False

    letters = [ch for ch in cleaned if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.7:
        return True

    title_like = sum(word[:1].isupper() for word in words if word)
    return title_like >= max(1, len(words) - 1)


def _paragraphs_from_text(text: str) -> List[str]:
    normalized = _normalize_filing_text(text)
    paragraphs: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        if buffer:
            paragraphs.append(" ".join(buffer).strip())
            buffer.clear()

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            flush_buffer()
            continue
        if _looks_like_heading(line):
            flush_buffer()
            paragraphs.append(line)
            continue
        if line.startswith("•"):
            flush_buffer()
            paragraphs.append(line)
            continue
        buffer.append(line)

    flush_buffer()
    return paragraphs


def _token_counter() -> Callable[[str], int]:
    if tiktoken is None:
        return lambda text: len(text.split())

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        encoding.encode("tokenizer warmup")
    except Exception:
        return lambda text: len(text.split())

    return lambda text: len(encoding.encode(text))


def _semantic_blocks(paragraphs: Iterable[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []

    for paragraph in paragraphs:
        if _looks_like_heading(paragraph):
            if current:
                blocks.append(current)
            current = [paragraph]
            continue
        if not current:
            current = [paragraph]
            continue
        current.append(paragraph)

    if current:
        blocks.append(current)
    return blocks


def _split_large_block(block: List[str], max_tokens: int, count_tokens: Callable[[str], int]) -> List[List[str]]:
    if count_tokens("\n\n".join(block)) <= max_tokens:
        return [block]

    chunks: List[List[str]] = []
    current: List[str] = []
    for paragraph in block:
        candidate = current + [paragraph]
        if current and count_tokens("\n\n".join(candidate)) > max_tokens:
            chunks.append(current)
            current = [paragraph]
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def _overlap_tail(paragraphs: List[str], overlap_tokens: int, count_tokens: Callable[[str], int]) -> List[str]:
    tail: List[str] = []
    for paragraph in reversed(paragraphs):
        candidate = [paragraph] + tail
        if tail and count_tokens("\n\n".join(candidate)) > overlap_tokens:
            break
        tail = candidate
        if count_tokens("\n\n".join(tail)) >= overlap_tokens:
            break
    return tail


def chunk_text_semantic(text: str, max_tokens: int = 650, overlap_tokens: int = 100) -> List[str]:
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be strictly less than max_tokens")

    paragraphs = _paragraphs_from_text(text)
    if not paragraphs:
        return []

    count_tokens = _token_counter()
    blocks: List[List[str]] = []
    for block in _semantic_blocks(paragraphs):
        blocks.extend(_split_large_block(block, max_tokens=max_tokens, count_tokens=count_tokens))

    chunks: List[str] = []
    current: List[str] = []
    for block in blocks:
        candidate = current + block
        if current and count_tokens("\n\n".join(candidate)) > max_tokens:
            chunks.append("\n\n".join(current).strip())
            current = _overlap_tail(current, overlap_tokens=overlap_tokens, count_tokens=count_tokens) + block
            if count_tokens("\n\n".join(current)) > max_tokens:
                chunks.append("\n\n".join(block).strip())
                current = []
        else:
            current = candidate

    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def read_and_chunk_file(file_path: str, max_tokens: int = 650, overlap_tokens: int = 100) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file_handle:
        text = file_handle.read()
    return chunk_text_semantic(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
