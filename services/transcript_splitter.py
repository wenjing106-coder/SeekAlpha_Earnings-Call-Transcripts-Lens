"""
services/transcript_splitter.py
================================
[Purpose]
Split a complete earnings-call transcript into two distinct sections:
  1. Prepared Remarks — Formal statements delivered by company executives.
  2. Q&A Session — Analyst questions and management's real-time answers.

[Why do we need to split?]
During an earnings call:
  - Prepared Remarks are carefully scripted, polished, and optimistic by design.
  - Q&A is spontaneous; analysts ask tough questions and management responds on the spot.
By analyzing sentiment separately for each section, we can detect potential
communication inconsistency (e.g., overly positive prepared remarks vs. cautious Q&A).

[Design Strategy]
- Pure rule-based approach: We scan for known "marker" phrases that indicate
  where the Q&A section begins.
- No machine learning needed because these markers are highly predictable
  and consistent across transcript providers (Seeking Alpha, etc.).
- If no marker is found, the entire text is treated as Prepared Remarks and
  Q&A is returned as an empty string.

[Marker Source]
Common markers from Seeking Alpha and similar financial transcript providers:
  - "Question-and-Answer Session" (most common explicit header)
  - "Q&A Session"
  - "Questions and Answers"
  - Moderator transition cues like "let's take questions"
  - Operator announcing "first question"

[Key Concepts for Beginners]
- re.compile(): Pre-compiles a regex pattern for reuse (better performance).
- re.MULTILINE: Makes ^ and $ match the start/end of EACH LINE (not just the whole string).
- re.DOTALL: Makes the . character also match newline characters (\\n).
- re.IGNORECASE: Makes matching case-insensitive ("Question" matches "question").
- dataclass: A decorator that auto-generates __init__ and other boilerplate methods.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SplitTranscript:
    """
    Container for the split transcript result.
    
    [Field Descriptions]
    - prepared_remarks: Full text of the Prepared Remarks section.
    - qa_section: Full text of the Q&A section (empty string "" if not found).
    - split_method: Which marker triggered the split (useful for debugging/display).
    """
    prepared_remarks: str
    qa_section: str
    split_method: str  # Which marker was used to split (for debugging)


# ---------------------------------------------------------------------------
# Q&A Boundary Marker List (ordered by priority)
# 
# How it works: We try each marker in order. The FIRST one that matches
# determines the split point. Each element is a tuple: (description, compiled regex).
# ---------------------------------------------------------------------------
_QA_MARKERS: List[Tuple[str, re.Pattern]] = [
    # ===== Priority 1: Explicit section headers =====
    # These are the most reliable markers — they appear as standalone lines.
    
    # Matches: "Question-and-Answer Session" (on its own line)
    # This is the most common format on Seeking Alpha transcripts.
    ("header: Question-and-Answer Session",
     re.compile(r"^\s*Question[\s-]*and[\s-]*Answer\s+Session\s*$",
                re.IGNORECASE | re.MULTILINE)),

    # Matches: "Q&A Session" (on its own line)
    ("header: Q&A Session",
     re.compile(r"^\s*Q\s*&\s*A\s+Session\s*$",
                re.IGNORECASE | re.MULTILINE)),

    # Matches: "Questions and Answers" or "Question and Answer" (on its own line)
    ("header: Questions and Answers",
     re.compile(r"^\s*Questions?\s+and\s+Answers?\s*$",
                re.IGNORECASE | re.MULTILINE)),

    # ===== Priority 2: Moderator transition cues =====
    # When there's no explicit header, moderators typically say something like
    # "let's turn it over to take questions" or "open it up for Q&A".
    
    # Matches phrases like: "turn it over ... take questions" or "open up for Q&A"
    ("cue: turn it over ... take questions",
     re.compile(
         r"(?:turn\s+it\s+over|open\s+(?:it\s+)?up|like\s+to)\s+"
         r".*?(?:take\s+questions|Q\s*&\s*A|question[\s-]*and[\s-]*answer)",
         re.IGNORECASE)),

    # ===== Priority 3: Operator announcing the first question =====
    # When the Operator says "Our first question will come from..." the Q&A has begun.
    
    # Matches: "Operator\n... first question" or "Operator\n... begin the question"
    ("cue: Operator first question",
     re.compile(
         r"Operator\s*\n"
         r".*(?:first\s+question|begin\s+the\s+question)",
         re.IGNORECASE | re.DOTALL)),
]

# ---------------------------------------------------------------------------
# Trailing Noise Cleanup Regex
# 
# Seeking Alpha transcripts often have leftover voting widget text at the bottom:
#   "1\n  V. Bearish\n2\n  Bearish\n3\n  Neutral\n..."
# This is useless for analysis and must be removed.
# ---------------------------------------------------------------------------
_TAIL_NOISE_RE = re.compile(
    r"(?:\n\s*\d+\s*\n\s*(?:V\.\s*)?(?:Bearish|Bullish|Neutral|Authors?).*)",
    re.IGNORECASE | re.DOTALL,
)


def split_transcript(raw_text: str) -> SplitTranscript:
    """
    Split a complete earnings-call transcript into Prepared Remarks and Q&A.

    [Parameters]
    raw_text : str
        The full transcript text content.

    [Returns]
    SplitTranscript object containing both sections and the split method used.

    [Processing Steps]
    1. Remove trailing web noise (voting widget remnants, etc.)
    2. Try each Q&A marker in priority order.
    3. First match found -> everything before = Prepared Remarks, everything after = Q&A.
    4. No marker found -> entire text is treated as Prepared Remarks.

    [Usage Example]
    >>> result = split_transcript(full_text)
    >>> print(len(result.prepared_remarks), len(result.qa_section))
    >>> print(result.split_method)  # Shows which marker triggered the split
    """
    # Step 1: Remove trailing noise.
    # sub() replaces matched text with empty string (effectively deleting it).
    cleaned = _TAIL_NOISE_RE.sub("", raw_text).strip()

    # Step 2: Try each marker in priority order.
    for label, pattern in _QA_MARKERS:
        m = pattern.search(cleaned)  # Search the entire text for this pattern
        if m:
            # Found a match! Use m.start() (the position where the match begins) to split.
            split_pos = m.start()
            prepared = cleaned[:split_pos].strip()  # Everything before = Prepared Remarks
            qa = cleaned[split_pos:].strip()        # Everything after (inclusive) = Q&A
            return SplitTranscript(
                prepared_remarks=prepared,
                qa_section=qa,
                split_method=label,  # Record which marker triggered the split
            )

    # Step 3: No Q&A marker found -> the whole text is Prepared Remarks.
    return SplitTranscript(
        prepared_remarks=cleaned,
        qa_section="",
        split_method="no marker found (whole text = Prepared Remarks)",
    )
