# Earnings Call Lens- Powered by Deepseek V4-Flash

**Detect sentiment shifts, hidden signals & topic evolution across quarterly earnings calls.**

A Streamlit web application that analyzes earnings-call transcripts using FinBERT sentiment analysis, BERTopic-style topic modeling, and LLM-powered financial insights (DeepSeek V4-Flash).

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Sentiment Shift Detection** | Sentence-level FinBERT NSS scoring + Welch's t-test + hedge-word density + length-anomaly detection |
| **Topic Evolution** | BERTopic-style pipeline (PCA + sklearn HDBSCAN + c-TF-IDF + hierarchical topic reduction) |
| **Factor Risk Quadrant** | Bubble chart mapping Attention Delta vs. Q&A NSS with quadrant classification (Momentum / Stress / Recovery / Dormant) |
| **Sankey Network** | Multi-quarter step-indexed flow diagram showing top-5 macro-theme evolution |
| **LLM-Powered Insights** | DeepSeek V4-Flash generates professional financial conclusions and actionable investor recommendations |
| **Local Narrative Engine** | Data-driven fallback that produces unique, contextual analysis paragraphs when no API key is configured |
| **Tab-Based UI** | Three top-level tabs: Sentiment Analysis, Topic Shift Analysis, Raw Transcripts |

---

## How It Works

### Data Flow

```
User uploads TXT files (up to 4 quarterly transcripts)
       |
       v
file_parser.py --> TranscriptMeta (company, ticker, year, quarter, raw text)
       |
       v
analysis_pipeline.py (per-transcript orchestration)
       |--- transcript_splitter.py (split into Prepared Remarks + Q&A)
       |--- sentiment_engine.py (FinBERT NSS, t-test, hedge words, length)
       |
       v
topic_shift.py (cross-quarter BERTopic modeling)
       |
       v
llm_insights.py (AI-powered conclusions + actionable advice)
```

### Sentiment Analysis Pipeline

1. **Transcript Splitting** -- Rule-based detection of Q&A boundary markers (e.g., "Question-and-Answer Session") to separate Prepared Remarks from Q&A.
2. **Sentence-Level NSS** -- Each sentence scored by FinBERT (`yiyanghkust/finbert-tone`): NSS = P(Positive) - P(Negative).
3. **Section Aggregation** -- Mean NSS computed per section (PR and Q&A).
4. **Delta & t-Test** -- Delta = mu_PR - mu_QA; Welch's t-test determines statistical significance (p < 0.05).
5. **Hedge Words** -- Regex-based detection of uncertainty language; density measured per 1000 words.
6. **Length Mutation** -- Flags anomalously short (evasiveness) or long (obfuscation) Q&A responses.

### Topic Shift Pipeline (BERTopic-Style)

1. **Sentence Splitting** -- Transcripts split into individual sentences (not paragraphs).
2. **Embedding** -- `sentence-transformers/all-MiniLM-L6-v2` encodes all sentences.
3. **Dimensionality Reduction** -- PCA (15 components) via sklearn.
4. **Clustering** -- `sklearn.cluster.HDBSCAN` (avoids C-library dependencies on Streamlit Cloud).
5. **c-TF-IDF** -- Class-based TF-IDF with POS lexical pruning (retains only PROPN/NOUN tokens).
6. **Financial Stopwords** -- 9-category domain-specific dictionary (800+ terms including personal names).
7. **Hierarchical Reduction** -- Cosine-similarity merging of micro-topics down to max 12 macro-themes.
8. **QoQ Delta Proportions** -- Quarter-over-quarter attention shifts for anomaly detection.
9. **Factor Score** -- |Delta Proportion| x (1 - NSS) ranks topics by combined risk.

---

## LLM Integration

### Primary: DeepSeek V4-Flash

- **Model**: `deepseek-chat` (routes to DeepSeek-V4-Flash)
- **API**: OpenAI-compatible endpoint at `https://api.deepseek.com`
- **Mode**: Thinking mode explicitly **disabled** for reliable structured output
- **Parameters**: max_tokens=2048, temperature=0.4, top_p=0.9
- **Output Format**: ANALYSIS paragraph + 3 numbered RECOMMENDATIONS
- **Parsing**: Strict format parser with lenient fallback (handles non-standard responses)

### Fallback: Local Narrative Engine

When no API key is configured (or the API fails), a sophisticated rule-based engine generates unique, data-driven paragraphs based on actual analysis patterns. This engine:

- Produces contextual financial analysis (not generic templates)
- References specific data points (delta values, hedge ratios, quarter labels)
- Covers multiple scenarios: consistent sentiment, persistent divergence, worsening trends, improving trends
- Always succeeds -- no external dependencies required

### API Key Setup

Provide your DeepSeek API key via any of these methods (priority order):

1. **Sidebar input** (recommended) -- Paste directly into the "DeepSeek API Key" field
2. **Streamlit secrets** -- Add `DEEPSEEK_API_KEY = "sk-..."` to `.streamlit/secrets.toml`
3. **Environment variable** -- `export DEEPSEEK_API_KEY="sk-..."`

Get your key at: [platform.deepseek.com](https://platform.deepseek.com) ($2 top-up is enough for months of use).

---

## Project Structure

```
webapp/
|-- app.py                          Main Streamlit application (v7.0-deepseek)
|-- requirements.txt                Python dependencies
|-- README.md                       This file
|-- .gitignore                      Git ignore rules
|-- .streamlit/
|   |-- config.toml                 Theme + server configuration
|   |-- secrets.toml                (gitignored) API keys
|-- services/
|   |-- __init__.py                 Package docstring + architecture overview
|   |-- file_parser.py              Parse TXT uploads; extract metadata via regex
|   |-- transcript_splitter.py      Split into Prepared Remarks + Q&A sections
|   |-- sentiment_engine.py         FinBERT NSS scoring, t-test, hedge words, length
|   |-- topic_shift.py              BERTopic pipeline (PCA + HDBSCAN + c-TF-IDF)
|   |-- financial_stopwords.py      Domain-specific stopword dictionary (9 categories)
|   |-- analysis_pipeline.py        Orchestration layer (split + sentiment per transcript)
|   |-- llm_insights.py             DeepSeek LLM insights + local narrative engine fallback
```

---

## Installation & Running

### Prerequisites

- Python 3.9+
- ~2 GB disk space (for model downloads on first run)

## Quick Start

Follow these steps to get the application up and running locally:

```bash
# 1. Clone the repository and navigate into the project directory
git clone [https://github.com/wenjing106-coder/SeekAlpha_Earnings-Call-Transcripts-Lens.git](https://github.com/wenjing106-coder/SeekAlpha_Earnings-Call-Transcripts-Lens.git)
cd SeekAlpha_Earnings-Call-Transcripts-Lens

# 2. Install dependencies (uses CPU-only PyTorch to save memory)
pip install -r requirements.txt

# 3. Run the Streamlit application
streamlit run app.py
```

### Pro-Tip Additions (Optional but Recommended)

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Requirements

```
streamlit>=1.28.0
transformers>=4.35.0
torch>=2.0.0           (CPU-only via --extra-index-url)
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.18.0
requests>=2.28.0
```

---

## Memory Management

The application is designed to run in constrained environments (~1 GB RAM):

- **Sequential Model Loading** -- FinBERT (~440 MB) and the embedding model (~90 MB) are never held in memory simultaneously.
- **Explicit Garbage Collection** -- Models are deleted and `gc.collect()` is called between pipeline stages.
- **Float16 Precision** -- FinBERT loads in half-precision (`torch_dtype=torch.float16`) for ~50% memory savings.
- **Batch Processing** -- Sentence embeddings processed in batches of 64.

---

## Visualizations

### Tab 1: Sentiment Analysis

| Component | Description |
|-----------|-------------|
| **Sentiment Cards** | Per-quarter color-coded cards (Positive/Negative/Neutral) with NSS values |
| **Consolidated Chart** | 3-row subplot: Delta NSS with significance markers, QA hedge density, QA avg sentence length |
| **Per-Quarter Implications** | Contextual insight boxes ("Sentiment Collapse", "Defensive Opener", etc.) |
| **LLM Conclusion** | AI-generated analysis paragraph + 3 forward-looking recommendations |
| **Detailed Metrics Table** | Expandable raw data: PR/QA NSS, delta, t-stat, p-value, hedge density, avg length |

### Tab 2: Topic Shift Analysis

| Component | Description |
|-----------|-------------|
| **Factor Risk Quadrant** | Bubble chart: X=Attention Delta%, Y=QA NSS, Size=Factor Score, Color=Quadrant |
| **Sankey Network** | Multi-quarter step-indexed flow (top 5 themes); ribbon color = expansion/contraction/stable |
| **LLM Conclusion** | AI-generated topic evolution analysis + 3 forward-looking recommendations |
| **Topic Details Table** | Expandable: all discovered topics with POS-pruned keywords and document counts |

### Tab 3: Raw Transcripts

- Full transcript preview per quarter with sub-tabs: Prepared Remarks, Q&A Session, Full Transcript.

---

## Technical Design Decisions

| Decision | Rationale |
|----------|-----------|
| PCA + sklearn HDBSCAN (not UMAP + hdbscan) | Avoids C-library compilation issues on Streamlit Cloud |
| Sentence-level granularity for topics | Prevents one mega-topic from absorbing all text |
| POS lexical pruning (PROPN/NOUN only) | Eliminates filler verbs/adverbs from c-TF-IDF keywords |
| Heuristic POS fallback | Works when spaCy model is unavailable (Streamlit Cloud free tier) |
| Financial stopwords (9 categories) | Suppresses generic corporate vocabulary + executive names |
| Hierarchical topic reduction | Condenses 60+ micro-topics into 10-15 actionable macro-themes |
| DeepSeek thinking mode disabled | Prevents empty `content` responses; gives predictable structured output |
| `@st.cache_resource` for analysis | Avoids pickle serialization errors with nested dataclass objects |
| Sequential model loading with GC | Fits within 1 GB RAM constraint without swap-thrashing |

---

## Version History

| Version | Changes |
|---------|---------|
| **7.0-deepseek** | Migrated LLM backend from Mistral-7B/HF to DeepSeek V4-Flash; added local narrative engine fallback; DeepSeek API key validation in sidebar |
| **6.0** | BERTopic pipeline with POS lexical pruning; Sankey Network; Factor Risk Quadrant; hierarchical topic reduction |
| **5.0** | BERTopic-style topic modeling (PCA + HDBSCAN + c-TF-IDF); financial stopwords |
| **4.0** | Representative sentences per topic; QoQ delta proportions |
| **3.0** | Tab-based UI; sentiment cards; consolidated multi-metric chart |
| **2.0** | Sentence-level NSS; t-test; hedge words; length anomaly detection |
| **1.0** | Initial release; basic FinBERT sentiment analysis |

---

## License

This project is for educational and research purposes.
