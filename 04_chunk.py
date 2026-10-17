def chunk_text(text, chunk_size = 3, overlap = 2):
    chunk = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk.append(text[start:end])
        start += chunk_size - overlap
    return chunk
text = "ABCDEFGHIJ"
print(chunk_text(text, chunk_size=3, overlap=2))