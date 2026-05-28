"""
services/__init__.py
====================
Package initializer for the ``services`` package.

Architecture Overview (v7.0-deepseek)
--------------------------------------

Modules:
  file_parser.py           Parse uploaded TXT files; extract company/ticker/year/quarter.
  transcript_splitter.py   Split transcript into "Prepared Remarks" and "Q&A" sections.
  sentiment_engine.py      Sentence-level FinBERT NSS scoring, t-test, hedge words, length.
  topic_shift.py           BERTopic-style topic modeling (PCA + sklearn HDBSCAN + c-TF-IDF).
  financial_stopwords.py   Domain-specific stopword dictionary (9 categories + personal names).
  analysis_pipeline.py     Orchestrate split + sentiment into a single analysis entry point.
  llm_insights.py          LLM-powered financial conclusions & advice (DeepSeek V4-Flash API).

Data Flow:
  User uploads TXT files
       |
       v
  file_parser.py -> TranscriptMeta (metadata + raw text)
       |
       v
  analysis_pipeline.py (orchestration)
       |--- transcript_splitter.py (split into PR + Q&A)
       |--- sentiment_engine.py (NSS, t-test, hedge words, length anomaly)
       |
       v
  topic_shift.py (cross-quarter BERTopic modeling)
       (called directly by app.py after all individual analyses complete)
       |
       v
  llm_insights.py (generate AI-powered conclusions + actionable advice)
       (called by app.py separately for sentiment and topic results;
        gracefully falls back to local narrative engine if no DEEPSEEK_API_KEY configured)
"""
