import subprocess
import os

# Paths
MODEL_PATH = r"C:\Users\asant\Desktop\models\mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CLI_PATH = r"C:\Users\asant\Desktop\llama.cpp-master\build\bin\Release\llama-cli.exe"
INPUT_FILE = r"C:\Users\asant\Desktop\llama.cpp-master\ceera-llm\data\apple_mda_2024.txt"
OUTPUT_FILE = r"C:\Users\asant\Desktop\llama.cpp-master\ceera-llm\data\apple_mda_summary.txt"

# Query function
def query(prompt: str, n_predict=150, temp=0.7, threads=8):
    try:
        result = subprocess.run(
            [
                CLI_PATH,
                "-m", MODEL_PATH,
                "-p", prompt,
                "--n_predict", str(n_predict),
                "--temp", str(temp),
                "--threads", str(threads)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running llama-cli: {e}")
        return ""

# Chunk the text
def chunk_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        mda_text = f.read()

    chunks = chunk_text(mda_text, chunk_size=800)
    print(f"Total chunks to summarize: {len(chunks)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)} (length {len(chunk)} chars)...")
            prompt = f"Summarize this for me:\n{chunk}"
            reply = query(prompt)
            out_file.write(f"--- Chunk {i+1} ---\n{reply}\n\n")
            print(f"Chunk {i+1} summarized")

    print(f"\nSummarization complete! Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
