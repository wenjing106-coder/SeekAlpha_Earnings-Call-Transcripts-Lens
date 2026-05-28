"""
services/sentiment_engine.py — Sentiment Analysis Engine
==============================================================
[Purpose]
Perform sentence-level sentiment analysis using FinBERT (yiyanghkust/finbert-tone),
then compute a Net Sentiment Score (NSS) per sentence, aggregate by section, and
run a Two-Sample t-Test to determine whether the sentiment shift between Prepared
Remarks and Q&A is statistically significant.

[Key Terms]
- Sentence-level NSS: NSS = P(Positive) - P(Negative) for every sentence.
- Section means: mu_PR = mean NSS of Prepared Remarks sentences,
                 mu_QA = mean NSS of Q&A sentences.
- Delta = mu_PR - mu_QA (positive means PR is more optimistic than QA).
- Two-sample t-test: scipy.stats.ttest_ind on the two NSS distributions.
  If p-value < 0.05, the shift is statistically significant.
- Hedge Words detection: Counts hedging language (might, could, perhaps, uncertain, etc.)
  separately in PR and QA; spikes indicate evasiveness.
- Length Mutation: Computes average response length in Q&A to flag anomalies.

[What is FinBERT?]
FinBERT is a BERT model fine-tuned on financial text. It classifies into:
  Negative / Neutral / Positive

[Core Algorithm]
1. Split section text into individual sentences (using NLTK).
2. For each sentence, compute NSS = P(pos) - P(neg) via FinBERT.
3. Aggregate: mean(NSS_sentences) for each section.
4. Delta = mean_PR - mean_QA.
5. t-test on the two distributions to get p-value.
6. If p < 0.05, raise significance flag.

[Hedge Words]
A curated list of hedging/uncertainty language commonly used by executives:
"might", "could", "perhaps", "uncertain", "possibly", "may", "approximate",
"unclear", "we believe", "potentially", "it is possible", etc.
A spike in these words in Q&A vs PR signals defensive or evasive communication.

[Length Mutation]
If Q&A answers are drastically shorter (evasiveness) or longer (obfuscation)
than a reasonable baseline, this is flagged as anomalous.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import re
import torch
import numpy as np
from scipy import stats

from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FinBERT (yiyanghkust/finbert-tone) actual label order:
# index 0=Neutral, 1=Positive, 2=Negative
# We dynamically read model.config.label2id at inference time for safety.
_LABEL_NAMES = ["Neutral", "Positive", "Negative"]

# Max tokens per chunk for FinBERT (512 limit, 450 safe with special tokens)
_CHUNK_MAX_TOKENS = 450
_CHUNK_OVERLAP = 50

# Statistical significance threshold
_SIGNIFICANCE_ALPHA = 0.05

# Hedge / uncertainty words list
_HEDGE_WORDS = [
    r"\bmight\b", r"\bcould\b", r"\bperhaps\b", r"\buncertain\b",
    r"\bpossibly\b", r"\bmay\b", r"\bapproximate(?:ly)?\b",
    r"\bunclear\b", r"\bpotentially\b", r"\bwe believe\b",
    r"\bit is possible\b", r"\bwe think\b", r"\bwe hope\b",
    r"\bestimate\b", r"\bexpect\b", r"\banticipate\b",
    r"\blikely\b", r"\bunlikely\b", r"\bvolatil(?:e|ity)\b",
    r"\brisk\b", r"\bcautious\b", r"\bconservative\b",
]

# Pre-compiled combined pattern for hedge words (case-insensitive)
_HEDGE_PATTERN = re.compile("|".join(_HEDGE_WORDS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SectionSentiment:
    """
    Aggregated sentiment for one section (Prepared Remarks OR Q&A).

    Fields:
    - mean_nss: Mean Net Sentiment Score across all sentences.
    - nss_values: List of per-sentence NSS values (for t-test).
    - n_sentences: Number of sentences analyzed.
    - overall_label: "Positive" if mean_nss >= 0, else "Negative".
    - prob_positive: Average P(positive) across sentences.
    - prob_neutral: Average P(neutral) across sentences.
    - prob_negative: Average P(negative) across sentences.
    - hedge_count: Total hedge word occurrences in this section.
    - hedge_density: Hedge words per 1000 words.
    - avg_sentence_length: Average number of words per sentence.
    """
    mean_nss: float
    nss_values: List[float]
    n_sentences: int
    overall_label: str
    prob_positive: float
    prob_neutral: float
    prob_negative: float
    hedge_count: int
    hedge_density: float
    avg_sentence_length: float


@dataclass
class TranscriptSentiment:
    """
    Complete sentiment analysis result for one transcript.

    Fields:
    - prepared: SectionSentiment for Prepared Remarks.
    - qa: SectionSentiment for Q&A (zeros if Q&A is empty).
    - delta: mu_PR - mu_QA (positive = PR more optimistic).
    - abs_delta: |delta| for magnitude comparison.
    - t_statistic: t-test statistic value.
    - p_value: Two-sample t-test p-value.
    - is_significant: True if p_value < 0.05.
    - risk_note: Human-readable assessment.
    - hedge_spike: True if QA hedge density > 1.5x PR hedge density.
    - length_anomaly: Description of length mutation if detected, else empty.
    """
    prepared: SectionSentiment
    qa: SectionSentiment
    delta: float
    abs_delta: float
    t_statistic: float
    p_value: float
    is_significant: bool
    risk_note: str
    hedge_spike: bool
    length_anomaly: str


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_finbert() -> Tuple[BertTokenizer, BertForSequenceClassification]:
    """
    Load FinBERT tokenizer and model.
    Returns (tokenizer, model) tuple.
    model.eval() switches to inference mode.
    Uses float16 precision to reduce memory footprint (~50% savings).
    """
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone",
        torch_dtype=torch.float16,
    )
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_transcript_sentiment(
    prepared_text: str,
    qa_text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> TranscriptSentiment:
    """
    Full sentiment analysis pipeline:
    1. Compute per-sentence NSS for Prepared Remarks.
    2. Compute per-sentence NSS for Q&A.
    3. Calculate delta = mean_PR - mean_QA.
    4. Run Two-Sample t-Test for statistical significance.
    5. Detect hedge word spikes and length anomalies.
    6. Generate risk assessment.

    Parameters:
    - prepared_text: Full Prepared Remarks text.
    - qa_text: Full Q&A text (can be empty).
    - tokenizer, model: Pre-loaded FinBERT components.

    Returns: TranscriptSentiment dataclass.
    """
    # Analyze Prepared Remarks
    prepared_result = _analyze_section(prepared_text, tokenizer, model)

    # Analyze Q&A
    if qa_text and qa_text.strip():
        qa_result = _analyze_section(qa_text, tokenizer, model)
    else:
        qa_result = SectionSentiment(
            mean_nss=0.0, nss_values=[], n_sentences=0,
            overall_label="N/A",
            prob_positive=0.0, prob_neutral=0.0, prob_negative=0.0,
            hedge_count=0, hedge_density=0.0, avg_sentence_length=0.0,
        )

    # Compute delta
    delta = prepared_result.mean_nss - qa_result.mean_nss
    abs_delta = abs(delta)

    # Two-sample t-test (only if both have enough data)
    t_stat = 0.0
    p_val = 1.0
    is_sig = False

    if prepared_result.n_sentences >= 3 and qa_result.n_sentences >= 3:
        # Welch's t-test (does not assume equal variances)
        t_result = stats.ttest_ind(
            prepared_result.nss_values,
            qa_result.nss_values,
            equal_var=False,
        )
        t_stat = float(t_result.statistic)
        p_val = float(t_result.pvalue)
        is_sig = p_val < _SIGNIFICANCE_ALPHA

    # Hedge word spike detection
    hedge_spike = False
    if prepared_result.hedge_density > 0 and qa_result.hedge_density > 0:
        hedge_spike = qa_result.hedge_density > (1.5 * prepared_result.hedge_density)
    elif qa_result.hedge_density > 15:  # Absolute threshold if PR has zero
        hedge_spike = True

    # Length anomaly detection
    length_anomaly = ""
    if (prepared_result.avg_sentence_length > 0 and
            qa_result.avg_sentence_length > 0 and
            qa_result.n_sentences >= 3):
        ratio = qa_result.avg_sentence_length / prepared_result.avg_sentence_length
        if ratio < 0.5:
            length_anomaly = (
                f"Q&A answers are unusually short (avg {qa_result.avg_sentence_length:.0f} words vs "
                f"{prepared_result.avg_sentence_length:.0f} in PR). May indicate evasiveness."
            )
        elif ratio > 2.0:
            length_anomaly = (
                f"Q&A answers are unusually long (avg {qa_result.avg_sentence_length:.0f} words vs "
                f"{prepared_result.avg_sentence_length:.0f} in PR). May indicate obfuscation."
            )

    # Generate risk note
    risk_note = _generate_risk_note(
        qa_result.n_sentences, delta, is_sig, p_val, hedge_spike, length_anomaly
    )

    return TranscriptSentiment(
        prepared=prepared_result,
        qa=qa_result,
        delta=round(delta, 4),
        abs_delta=round(abs_delta, 4),
        t_statistic=round(t_stat, 4),
        p_value=round(p_val, 6),
        is_significant=is_sig,
        risk_note=risk_note,
        hedge_spike=hedge_spike,
        length_anomaly=length_anomaly,
    )


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _analyze_section(
    text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> SectionSentiment:
    """
    Analyze a section at sentence level: compute NSS for each sentence,
    count hedge words, and measure sentence lengths.
    """
    if not text or not text.strip():
        return SectionSentiment(
            mean_nss=0.0, nss_values=[], n_sentences=0,
            overall_label="N/A",
            prob_positive=0.0, prob_neutral=0.0, prob_negative=0.0,
            hedge_count=0, hedge_density=0.0, avg_sentence_length=0.0,
        )

    # Split into sentences
    sentences = _split_sentences(text)
    if not sentences:
        return SectionSentiment(
            mean_nss=0.0, nss_values=[], n_sentences=0,
            overall_label="N/A",
            prob_positive=0.0, prob_neutral=0.0, prob_negative=0.0,
            hedge_count=0, hedge_density=0.0, avg_sentence_length=0.0,
        )

    # Compute NSS for each sentence
    nss_values = []
    all_probs = []  # To compute average probabilities

    # Dynamically resolve label indices from model config
    # FinBERT (yiyanghkust/finbert-tone): {0: Neutral, 1: Positive, 2: Negative}
    label2id = model.config.label2id
    pos_idx = label2id.get("Positive", label2id.get("positive", 1))
    neg_idx = label2id.get("Negative", label2id.get("negative", 2))
    neu_idx = label2id.get("Neutral", label2id.get("neutral", 0))

    for sent in sentences:
        probs = _infer_sentence(sent, tokenizer, model)
        # probs order matches model's id2label: [P(Neutral), P(Positive), P(Negative)]
        nss = float(probs[pos_idx] - probs[neg_idx])  # P(pos) - P(neg)
        nss_values.append(nss)
        all_probs.append(probs)

    # Aggregate
    mean_nss = float(np.mean(nss_values))
    avg_probs = np.mean(all_probs, axis=0)

    # Hedge words
    word_count = len(text.split())
    hedge_matches = _HEDGE_PATTERN.findall(text)
    hedge_count = len(hedge_matches)
    hedge_density = (hedge_count / max(word_count, 1)) * 1000

    # Average sentence length
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = float(np.mean(sent_lengths))

    return SectionSentiment(
        mean_nss=round(mean_nss, 4),
        nss_values=nss_values,
        n_sentences=len(sentences),
        overall_label="Positive" if mean_nss >= 0 else "Negative",
        prob_positive=round(float(avg_probs[pos_idx]), 4),
        prob_neutral=round(float(avg_probs[neu_idx]), 4),
        prob_negative=round(float(avg_probs[neg_idx]), 4),
        hedge_count=hedge_count,
        hedge_density=round(hedge_density, 2),
        avg_sentence_length=round(avg_sent_len, 1),
    )


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex-based rules.
    Filters out very short sentences (< 5 words) and speaker labels.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)

    filtered = []
    for s in raw_sentences:
        s = s.strip()
        if len(s.split()) < 5:
            continue
        # Skip speaker labels like "John Smith - CFO"
        if re.match(r"^[\w\s\.]+\s*[-–]\s*[\w\s,&]+$", s) and len(s.split()) < 10:
            continue
        # Skip standalone "Operator" lines
        if s.lower().startswith("operator"):
            continue
        filtered.append(s)

    return filtered


@torch.no_grad()
def _infer_sentence(
    sentence: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> np.ndarray:
    """
    Run FinBERT on a single sentence. If the sentence is too long (> 450 tokens),
    chunk it and average. Returns probability array in model's native label order:
    [P(Neutral), P(Positive), P(Negative)] for yiyanghkust/finbert-tone.
    """
    # Check token length
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)

    if len(token_ids) <= _CHUNK_MAX_TOKENS:
        # Single inference
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        return probs
    else:
        # Chunk and average (rare for single sentences, but possible for long paragraphs)
        chunks = _chunk_tokens(token_ids, tokenizer, _CHUNK_MAX_TOKENS, _CHUNK_OVERLAP)
        all_probs = []
        for chunk_text in chunks:
            inputs = tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)


def _chunk_tokens(
    token_ids: List[int],
    tokenizer: BertTokenizer,
    max_tokens: int,
    overlap: int,
) -> List[str]:
    """Split token IDs into overlapping chunks and decode back to text."""
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_str = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        if chunk_str.strip():
            chunks.append(chunk_str)
        start += max_tokens - overlap
    return chunks


def _generate_risk_note(
    qa_n_sentences: int,
    delta: float,
    is_significant: bool,
    p_value: float,
    hedge_spike: bool,
    length_anomaly: str,
) -> str:
    """Generate a comprehensive risk assessment note."""
    if qa_n_sentences == 0:
        return "Q&A section not detected; sentiment shift analysis unavailable."

    parts = []

    if is_significant:
        direction = "more optimistic" if delta > 0 else "more pessimistic"
        parts.append(
            f"Statistically significant sentiment shift detected (p={p_value:.4f}). "
            f"Prepared Remarks are {direction} than Q&A by {abs(delta):.3f} NSS points."
        )
    else:
        parts.append(
            f"No statistically significant sentiment shift (p={p_value:.4f}). "
            f"The tone difference between sections may be due to random variation."
        )

    if hedge_spike:
        parts.append(
            "Hedge word spike detected: Q&A contains significantly more uncertainty "
            "language than Prepared Remarks."
        )

    if length_anomaly:
        parts.append(length_anomaly)

    return " ".join(parts)
