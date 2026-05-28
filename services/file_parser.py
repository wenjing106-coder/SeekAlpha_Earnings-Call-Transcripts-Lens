"""
services/file_parser.py
========================
[Purpose]
This module is responsible for parsing uploaded TXT files and extracting key metadata:
  - Company Name
  - Stock Ticker Symbol
  - Fiscal Year
  - Fiscal Quarter (Q1/Q2/Q3/Q4)

[Why do we need this module?]
Earnings call transcripts typically contain a header line with structured information,
for example: "21st Century Fox, Inc. (NASDAQ:FOX) Q4 2015 Earnings Call August 5, 2015"
We use regular expressions (regex) to automatically extract company info from this line,
so the user doesn't have to type it manually.

[Design Strategy]
1. Primary method: Parse the first line of the transcript using regex (Seeking Alpha format).
2. Fallback method: If the first line doesn't match, try to extract info from the filename.
3. Final fallback: If nothing works, return "Unknown" / "N/A" default values.

[Key Concepts for Beginners]
- dataclass: A Python 3.7+ decorator that auto-generates __init__, __repr__, etc.
- re.compile(): Pre-compiles a regex pattern for better performance on repeated matches.
- Optional[str]: A type hint meaning the value can be either a str or None.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptMeta:
    """
    Structured metadata extracted from a single transcript file.
    
    [Field Descriptions]
    - company_name: Full company name, e.g. "21st Century Fox, Inc."
    - ticker: Stock ticker symbol, e.g. "FOX"
    - year: Fiscal year, e.g. "2015"
    - quarter: Fiscal quarter, e.g. "Q4"
    - filename: Original uploaded filename (used for display and fallback parsing)
    - raw_text: Complete transcript text content (all subsequent analysis is based on this)
    """
    company_name: str
    ticker: str
    year: str
    quarter: str
    filename: str
    raw_text: str  # Full transcript content


# ---------------------------------------------------------------------------
# Regex Pattern Definitions
# ---------------------------------------------------------------------------

# [Pattern 1] Extract metadata from the first line of the transcript.
# Expected format: "Company Name (EXCHANGE:TICKER) QN YYYY Earnings Call ..."
# Example: "21st Century Fox, Inc. (NASDAQ:FOX) Q4 2015 Earnings Call August 5, 2015"
#
# Group breakdown:
#   (?P<company>.+?)     -> Non-greedy match for company name (stops at parenthesis)
#   \((?P<exchange>\w+):(?P<ticker>\w+)\)  -> Matches (EXCHANGE:TICKER) e.g. (NASDAQ:FOX)
#   Q(?P<quarter>[1-4])  -> Matches quarter Q1-Q4
#   (?P<year>\d{4})      -> Matches four-digit year
_HEADER_RE = re.compile(
    r"^(?P<company>.+?)\s*"           # Company name (non-greedy, stops at space+paren)
    r"\((?P<exchange>\w+):(?P<ticker>\w+)\)\s*"  # (EXCHANGE:TICKER)
    r"Q(?P<quarter>[1-4])\s+"         # Quarter: Q1-Q4
    r"(?P<year>\d{4})",               # Four-digit year
    re.IGNORECASE,                    # Case-insensitive matching
)

# [Pattern 2] Fallback: extract quarter and year from the filename.
# Example filename: "FOX Q4 2015 Results.txt" -> extracts Q4, 2015
_FILENAME_QY_RE = re.compile(r"Q([1-4])\s*(\d{4})", re.IGNORECASE)

# [Pattern 3] Fallback: extract company name and ticker from the filename.
# Example filename: "21st Century Fox (FOX) Q4 2015.txt" -> extracts "21st Century Fox", "FOX"
_FILENAME_COMPANY_RE = re.compile(
    r"^(?P<company>.+?)\s*\((?P<ticker>\w+)\)", re.IGNORECASE
)


def parse_transcript(raw_bytes: bytes, filename: str) -> TranscriptMeta:
    """
    Parse an uploaded TXT file and return structured metadata.

    [Parameters]
    raw_bytes : bytes
        Raw byte content of the uploaded file.
        Streamlit's file_uploader returns bytes, so we accept bytes here.
    filename : str
        Original filename. Used as a fallback source when the first line cannot be parsed.

    [Returns]
    TranscriptMeta object containing all extracted metadata.

    [Usage Example]
    >>> with open("transcript.txt", "rb") as f:
    ...     meta = parse_transcript(f.read(), "transcript.txt")
    >>> print(meta.company_name, meta.ticker, meta.quarter, meta.year)
    """
    # Step 1: Decode bytes to string.
    # Files from different sources may use different encodings (UTF-8 is most common).
    text = _decode_text(raw_bytes)

    # Step 2: Find the first non-empty line (skip any blank lines at the top).
    first_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    # Step 3: Try to extract metadata from the first line using the primary regex.
    company, ticker, year, quarter = _extract_meta_from_line(first_line)

    # Step 4: If quarter/year not found in first line, try extracting from the filename.
    if not quarter or not year:
        year_fb, quarter_fb = _extract_qy_from_filename(filename)
        year = year or year_fb          # "or" logic: if year is empty string, use fallback
        quarter = quarter or quarter_fb

    # Step 5: If company/ticker not found either, try extracting from the filename.
    if not company or not ticker:
        company_fb, ticker_fb = _extract_company_from_filename(filename)
        company = company or company_fb
        ticker = ticker or ticker_fb

    # Step 6: Build and return the result object with defaults for anything still missing.
    return TranscriptMeta(
        company_name=company or "Unknown Company",  # Final fallback: default values
        ticker=ticker or "N/A",
        year=year or "N/A",
        quarter=quarter or "N/A",
        filename=filename,
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# Internal Helper Functions (prefixed with underscore = "module-private")
# ---------------------------------------------------------------------------

def _decode_text(raw: bytes) -> str:
    """
    Decode bytes to string, trying multiple encodings in priority order.
    
    [Why multiple encodings?]
    - UTF-8: Modern standard encoding, covers all global characters.
    - UTF-8-sig: UTF-8 with BOM (Byte Order Mark); commonly produced by Windows Notepad.
    - Latin-1: Western European encoding; serves as a safe fallback (never raises an error).
    
    [Parameters] raw: Raw byte data
    [Returns] Decoded string
    """
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # If all encodings fail (theoretically impossible since latin-1 decodes anything),
    # use errors="replace" to substitute un-decodable bytes with U+FFFD (replacement char).
    return raw.decode("utf-8", errors="replace")


def _extract_meta_from_line(line: str):
    """
    Extract metadata from the transcript's first line.
    
    [Parameters] line: The first non-empty line of the transcript.
    [Returns] Tuple (company, ticker, year, quarter); empty strings if parsing fails.
    
    [Regex Matching Example]
    Input:  "21st Century Fox, Inc. (NASDAQ:FOX) Q4 2015 Earnings Call August 5, 2015"
    Result:
      company  = "21st Century Fox, Inc."
      exchange = "NASDAQ"
      ticker   = "FOX"
      quarter  = "4"
      year     = "2015"
    """
    m = _HEADER_RE.match(line)  # .match() attempts to match from the start of the string
    if m:
        return (
            m.group("company").strip().rstrip(","),  # Remove trailing comma if present
            m.group("ticker").upper(),               # Normalize ticker to uppercase
            m.group("year"),
            f"Q{m.group('quarter')}",               # Prepend "Q" prefix, e.g. "Q4"
        )
    return ("", "", "", "")  # Match failed, return all empty


def _extract_qy_from_filename(filename: str):
    """
    Fallback method: extract quarter and year from the filename string.
    
    [Parameters] filename: The uploaded file's name.
    [Returns] Tuple (year, quarter).
    
    [Examples]
    "FOX Q4 2015 Results.txt" -> ("2015", "Q4")
    "random_file.txt"         -> ("", "")
    """
    m = _FILENAME_QY_RE.search(filename)  # .search() looks anywhere in the string
    if m:
        return m.group(2), f"Q{m.group(1)}"
    return ("", "")


def _extract_company_from_filename(filename: str):
    """
    Fallback method: extract company name and ticker from the filename string.
    
    [Parameters] filename: The uploaded file's name.
    [Returns] Tuple (company_name, ticker).
    
    [Examples]
    "21st Century Fox (FOX) Q4 2015.txt" -> ("21st Century Fox", "FOX")
    "unknown_file.txt"                   -> ("", "")
    """
    m = _FILENAME_COMPANY_RE.search(filename)
    if m:
        return m.group("company").strip(), m.group("ticker").upper()
    return ("", "")
