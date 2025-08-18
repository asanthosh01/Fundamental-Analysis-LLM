from bs4 import BeautifulSoup
import re

def clean_text(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    # Extract visible text with newlines separating blocks
    text = soup.get_text(separator="\n")

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)

    # Strip leading/trailing whitespace
    return text.strip()

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def chunk_text_by_paragraph(text, max_paragraphs=5):
    paragraphs = text.split('. ')  # roughly split by sentence end
    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs):
        chunk = '. '.join(paragraphs[i:i+max_paragraphs])
        chunks.append(chunk)
    return chunks

def main():
    with open("apple_10k_20240928.txt", "r", encoding="utf-8") as f:
        raw_html = f.read()

    cleaned_text = clean_text(raw_html)
    chunks = chunk_text(cleaned_text)

    print(f"Total chunks created: {len(chunks)}")
    print("Example chunk preview:", chunks[0][:500])

if __name__ == "__main__":
    main()
