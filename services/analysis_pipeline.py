"""
services/analysis_pipeline.py — Orchestration Layer
=========================================================
[Purpose]
Single-function entry point for per-transcript analysis.
Summarization has been removed; focus is on:
  1. Transcript splitting (PR vs Q&A)
  2. Sentence-level sentiment analysis (NSS, t-test, hedge words, length)

Topic shift (BERTopic) runs cross-quarter in app.py after individual analyses.
"""

from dataclasses import dataclass

from transformers import BertTokenizer, BertForSequenceClassification

from services.transcript_splitter import split_transcript, SplitTranscript
from services.sentiment_engine import (
    compute_transcript_sentiment,
    TranscriptSentiment,
)


@dataclass
class AnalysisResult:
    """
    All analysis results for a single transcript.

    Fields:
    - company_name, ticker, quarter_label: Display metadata.
    - year_sort_key: Numeric key for chronological sorting.
    - split: SplitTranscript (PR + Q&A texts).
    - sentiment: TranscriptSentiment (NSS, t-test, hedge words, length).
    """
    company_name: str
    ticker: str
    quarter_label: str
    year_sort_key: int

    split: SplitTranscript
    sentiment: TranscriptSentiment


def run_full_analysis(
    company_name: str,
    ticker: str,
    year: str,
    quarter: str,
    raw_text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> AnalysisResult:
    """
    Execute analysis pipeline for one transcript:
    1. Split into PR + Q&A.
    2. Run sentiment analysis.
    """
    quarter_label = f"{quarter} {year}" if quarter != "N/A" else f"{year}"
    year_sort_key = _compute_sort_key(year, quarter)

    # Step 1: Split transcript
    split = split_transcript(raw_text)

    # Step 2: Sentiment analysis
    sentiment = compute_transcript_sentiment(
        prepared_text=split.prepared_remarks,
        qa_text=split.qa_section,
        tokenizer=tokenizer,
        model=model,
    )

    return AnalysisResult(
        company_name=company_name,
        ticker=ticker,
        quarter_label=quarter_label,
        year_sort_key=year_sort_key,
        split=split,
        sentiment=sentiment,
    )


def _compute_sort_key(year: str, quarter: str) -> int:
    """year * 10 + quarter_number for chronological sorting."""
    try:
        y = int(year)
    except (ValueError, TypeError):
        y = 9999
    q_num = 0
    if quarter and quarter.upper().startswith("Q"):
        try:
            q_num = int(quarter[1])
        except (ValueError, IndexError):
            q_num = 0
    return y * 10 + q_num
