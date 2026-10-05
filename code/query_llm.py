"""Compatibility entry point for the portable, evidence-validated LangChain pipeline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from financial_analysis.sentiment import main

if __name__ == "__main__":
    main()
