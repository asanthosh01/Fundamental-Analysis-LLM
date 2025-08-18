import requests
from bs4 import BeautifulSoup

CIK = "0000320193"

headers = {
    "User-Agent": "Albin Santhosh (asanthosh2406@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

filing_url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/000032019324000123/aapl-20240928.htm"

filing_html = requests.get(filing_url, headers=headers).text

print("Filing URL:", filing_url)
print("Filing HTML content preview:", filing_html[:500])

soup = BeautifulSoup(filing_html, "html.parser")
text = soup.get_text()

with open("apple_10k_20240928.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Apple 10-K filing saved as 'apple_10k_20240928.txt'")
