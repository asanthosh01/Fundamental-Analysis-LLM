import re

def extract_mda_section(text):
    # Normalize the text (case/spacing)
    lower_text = text.lower()

    # find start and end of MD&A section
    start_pattern = re.search(r'item\s+7\.*\s+management[’\'`]?s discussion and analysis', lower_text)
    end_pattern = re.search(r'item\s+8\.*\s+financial statements', lower_text)

    if start_pattern and end_pattern:
        start_idx = start_pattern.start()
        end_idx = end_pattern.start()

        mda_section = text[start_idx:end_idx]
        return mda_section.strip()
    else:
        print("Could not find MD&A section boundaries.")
        return None

def main():
    with open("apple_10k_20240928.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    mda_text = extract_mda_section(full_text)

    if mda_text:
        with open("apple_mda_2024.txt", "w", encoding="utf-8") as out:
            out.write(mda_text)
        print("Extracted MD&A section saved as 'apple_mda_2024.txt'")
        print("Preview:\n")
        print(mda_text[:1000])
    else:
        print("No MD&A section extracted.")

if __name__ == "__main__":
    main()
