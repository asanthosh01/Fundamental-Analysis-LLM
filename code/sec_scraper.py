"""Retrieve filing HTML using a contact User-Agent supplied by the runner."""
import argparse
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",required=True)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    from urllib.parse import urlparse
    if urlparse(args.url).hostname not in {"www.sec.gov","sec.gov"}:
        parser.error("Provide an SEC filing URL")
    agent=os.environ.get("SEC_USER_AGENT","")
    if "@" not in agent:
        parser.error("Set SEC_USER_AGENT to your name and real contact email")
    response=requests.get(args.url,headers={"User-Agent":agent},timeout=30)
    response.raise_for_status()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(BeautifulSoup(response.text,"html.parser").get_text(" ",strip=True),encoding="utf-8")
