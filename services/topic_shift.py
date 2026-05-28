"""
services/topic_shift.py — BERTopic-Style Topic Shift Detection
====================================================================
[Purpose]
Discover topics discussed across multiple quarters and visualize how the
topic distribution shifts over time using the BERTopic methodology.

[Algorithm — BERTopic pipeline implemented manually]
1. Split transcripts into individual SENTENCES (not paragraphs/chunks).
2. Embed sentences using sentence-transformers (all-MiniLM-L6-v2).
3. Reduce dimensionality with PCA (sklearn, 15 components).
4. Cluster sentences with sklearn.cluster.HDBSCAN.
5. Extract topic representations via class-based TF-IDF (c-TF-IDF)
   with a custom financial stopword dictionary and POS lexical pruning.
6. Hierarchical topic reduction — merge similar topics via cosine
   similarity of c-TF-IDF vectors down to max_topics macro themes.
7. Store representative sentences per topic per quarter.

[Key Design Decisions]
- Sentence-level granularity prevents one mega-topic from absorbing all text.
- Financial stopwords suppress generic corporate vocabulary, executive names,
  filler words, and procedural language that dominates earnings calls.
- PCA + sklearn HDBSCAN avoids bertopic/hdbscan/umap compilation issues
  on Streamlit Cloud (no C-library dependencies required).
- POS Lexical Pruning retains only PROPN/NOUN tokens for keyword extraction,
  using spaCy when available or a comprehensive heuristic fallback.
- Hierarchical reduction condenses 60+ micro-topics into 10-15 macro themes.
- Raw sentence counts per topic per quarter enable Sankey flow visualization.

[Output]
- Per-quarter topic distribution (% of sentences devoted to each topic).
- Topic labels (auto-generated keyword representations).
- Representative sentences per topic per quarter (top 5).
- QoQ delta proportions for signal detection.
- Raw sentence counts per (topic, quarter) for Sankey ribbon widths.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re

import numpy as np
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from services.financial_stopwords import FINANCIAL_STOPWORDS


# ---------------------------------------------------------------------------
# Heuristic POS Filter — Module-Level Constant
# ---------------------------------------------------------------------------
# This set serves as POS-like filtering. It contains common verbs,
# adverbs, adjectives, function words, and other non-noun tokens that should
# be excluded from c-TF-IDF keyword extraction.
#
# Using a frozenset at module level ensures:
# 1. O(1) lookup during token filtering (vs O(n) with a list).
# 2. Single allocation on import — not rebuilt per function call.
# 3. Immutability — cannot be accidentally modified at runtime.
# ---------------------------------------------------------------------------
_HEURISTIC_FILTER_OUT = frozenset({
    # --- Common verbs (all tenses/forms) ---
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "must", "need", "dare", "ought",
    "say", "said", "says", "saying",
    "get", "gets", "got", "getting", "gotten",
    "make", "makes", "made", "making",
    "go", "goes", "went", "going", "gone",
    "take", "takes", "took", "taking", "taken",
    "come", "comes", "came", "coming",
    "give", "gives", "gave", "giving", "given",
    "know", "knows", "knew", "knowing", "known",
    "think", "thinks", "thought", "thinking",
    "see", "sees", "saw", "seeing", "seen",
    "want", "wants", "wanted", "wanting",
    "look", "looks", "looked", "looking",
    "use", "uses", "used", "using",
    "find", "finds", "found", "finding",
    "tell", "tells", "told", "telling",
    "ask", "asks", "asked", "asking",
    "work", "works", "worked", "working",
    "seem", "seems", "seemed", "seeming",
    "feel", "feels", "felt", "feeling",
    "try", "tries", "tried", "trying",
    "leave", "leaves", "left", "leaving",
    "call", "calls", "called", "calling",
    "keep", "keeps", "kept", "keeping",
    "let", "lets", "letting",
    "begin", "begins", "began", "beginning",
    "show", "shows", "showed", "showing", "shown",
    "hear", "hears", "heard", "hearing",
    "play", "plays", "played", "playing",
    "run", "runs", "ran", "running",
    "move", "moves", "moved", "moving",
    "live", "lives", "lived", "living",
    "believe", "believes", "believed", "believing",
    "bring", "brings", "brought", "bringing",
    "happen", "happens", "happened", "happening",
    "write", "writes", "wrote", "writing", "written",
    "provide", "provides", "provided", "providing",
    "sit", "sits", "sat", "sitting",
    "stand", "stands", "stood", "standing",
    "lose", "loses", "lost", "losing",
    "pay", "pays", "paid", "paying",
    "meet", "meets", "met", "meeting",
    "include", "includes", "included", "including",
    "continue", "continues", "continued", "continuing",
    "set", "sets", "setting",
    "learn", "learns", "learned", "learning",
    "change", "changes", "changed", "changing",
    "lead", "leads", "led", "leading",
    "understand", "understands", "understood", "understanding",
    "watch", "watches", "watched", "watching",
    "follow", "follows", "followed", "following",
    "stop", "stops", "stopped", "stopping",
    "create", "creates", "created", "creating",
    "speak", "speaks", "spoke", "speaking", "spoken",
    "read", "reads", "reading",
    "allow", "allows", "allowed", "allowing",
    "add", "adds", "added", "adding",
    "spend", "spends", "spent", "spending",
    "grow", "grows", "grew", "growing", "grown",
    "open", "opens", "opened", "opening",
    "walk", "walks", "walked", "walking",
    "win", "wins", "won", "winning",
    "offer", "offers", "offered", "offering",
    "remember", "remembers", "remembered", "remembering",
    "consider", "considers", "considered", "considering",
    "appear", "appears", "appeared", "appearing",
    "buy", "buys", "bought", "buying",
    "serve", "serves", "served", "serving",
    "die", "dies", "died", "dying",
    "send", "sends", "sent", "sending",
    "expect", "expects", "expected", "expecting",
    "build", "builds", "built", "building",
    "stay", "stays", "stayed", "staying",
    "fall", "falls", "fell", "falling", "fallen",
    "reach", "reaches", "reached", "reaching",
    "kill", "kills", "killed", "killing",
    "remain", "remains", "remained", "remaining",
    "suggest", "suggests", "suggested", "suggesting",
    "raise", "raises", "raised", "raising",
    "pass", "passes", "passed", "passing",
    "sell", "sells", "sold", "selling",
    "require", "requires", "required", "requiring",
    "report", "reports", "reported", "reporting",
    "decide", "decides", "decided", "deciding",
    "pull", "pulls", "pulled", "pulling",
    "drive", "drives", "drove", "driving", "driven",
    "deliver", "delivers", "delivered", "delivering",
    "achieve", "achieves", "achieved", "achieving",
    "focus", "focuses", "focused", "focusing",
    # --- Adverbs ---
    "very", "really", "just", "also", "well", "still", "already",
    "even", "quite", "almost", "enough", "probably", "certainly",
    "actually", "usually", "always", "never", "often", "sometimes",
    "soon", "yet", "perhaps", "maybe", "indeed", "however",
    "therefore", "thus", "hence", "moreover", "furthermore",
    "nevertheless", "nonetheless", "meanwhile", "otherwise",
    "basically", "essentially", "generally", "specifically",
    "particularly", "especially", "approximately", "significantly",
    "relatively", "absolutely", "definitely", "clearly",
    "obviously", "apparently", "frankly", "honestly",
    "simply", "merely", "hardly", "barely", "slightly",
    "extremely", "incredibly", "tremendously",
    # --- Adjectives (common, non-topical) ---
    "good", "great", "big", "small", "large", "long", "short",
    "high", "low", "new", "old", "young", "important", "different",
    "same", "other", "next", "last", "few", "many", "much",
    "little", "own", "right", "able", "possible", "likely",
    "certain", "sure", "true", "real", "clear", "full",
    "strong", "better", "best", "worse", "worst",
    "higher", "highest", "lower", "lowest",
    "larger", "largest", "smaller", "smallest",
    "significant", "meaningful", "substantial",
    # --- Function words / determiners / prepositions / conjunctions ---
    "the", "and", "that", "this", "with", "for", "from",
    "but", "not", "are", "all", "can", "had", "her",
    "was", "one", "our", "out", "you", "his", "has",
    "its", "they", "been", "have", "more", "when",
    "who", "than", "them", "some", "what", "there",
    "then", "into", "only", "very", "about", "over",
    "such", "after", "most", "also", "these", "those",
    "through", "between", "each", "which", "their",
    "where", "here", "both", "under", "around",
    "while", "before", "during", "without", "within",
    "along", "across", "behind", "beyond", "against",
    # --- Pronouns ---
    "him", "her", "she", "his", "hers", "himself", "herself",
    "they", "them", "their", "theirs", "themselves",
    "myself", "yourself", "itself", "ourselves", "yourselves",
    "who", "whom", "whose", "which", "what",
    # --- Common earnings-call filler / procedural ---
    "thank", "thanks", "okay", "yes", "yeah", "sure", "right",
    "question", "questions", "answer", "answers",
    "comment", "comments", "remark", "remarks",
    "please", "morning", "afternoon", "everyone", "everybody",
    "operator", "thank", "welcome", "hello", "goodbye",
})


# ---------------------------------------------------------------------------
# POS Lexical Pruning — Only retain PROPN and NOUN tokens
# ---------------------------------------------------------------------------

def _load_spacy_model():
    """
    Load spaCy model for POS tagging.
    Returns None if spaCy is not available (will use heuristic fallback).
    """
    if not _SPACY_AVAILABLE:
        return None

    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    except OSError:
        pass

    # Try to download the model
    try:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True, capture_output=True, timeout=120
        )
        return spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass

    # Last resort: use blank model
    nlp = spacy.blank("en")
    return nlp


# Lazy-loaded spaCy model (None means not loaded yet, False means unavailable)
_NLP_MODEL = None


def _get_nlp():
    """Get or load the spaCy NLP model (singleton). Returns None if unavailable."""
    global _NLP_MODEL
    if _NLP_MODEL is None:
        _NLP_MODEL = _load_spacy_model()
        if _NLP_MODEL is None:
            _NLP_MODEL = False  # Mark as unavailable
    return _NLP_MODEL if _NLP_MODEL is not False else None


def _pos_filter_documents(documents: List[str]) -> List[str]:
    """
    Apply POS tagging and retain only PROPN (Proper Nouns) and NOUN tokens.
    This eliminates filler verbs, adverbs, and adjectives from c-TF-IDF.

    If POS tagging is unavailable (blank model fallback), uses a heuristic:
    - Keeps tokens that are alpha, >2 chars, and not in _HEURISTIC_FILTER_OUT.

    Returns: List of POS-filtered documents (same length, nouns-only text).
    """
    nlp = _get_nlp()
    
    # Check if the model has a trained tagger
    has_tagger = nlp is not None and hasattr(nlp, 'pipe_names') and (
        "tagger" in nlp.pipe_names or "tok2vec" in nlp.pipe_names
    )
    
    if has_tagger:
        # Use proper POS tagging
        filtered = []
        for doc in nlp.pipe(documents, batch_size=256, n_process=1):
            noun_tokens = [token.text.lower() for token in doc
                           if token.pos_ in ("PROPN", "NOUN")
                           and len(token.text) > 1
                           and token.is_alpha]
            filtered.append(" ".join(noun_tokens))
        return filtered
    else:
        # Fallback: heuristic noun extraction (no trained model available)
        # Uses module-level _HEURISTIC_FILTER_OUT constant for performance
        filtered = []
        for doc_text in documents:
            tokens = doc_text.lower().split()
            noun_tokens = [
                t for t in tokens
                if t.isalpha() and len(t) > 2 and t not in _HEURISTIC_FILTER_OUT
            ]
            filtered.append(" ".join(noun_tokens))
        return filtered


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TopicInfo:
    """Information about a single discovered topic."""
    topic_id: int
    label: str           # e.g., "cloud_subscription_arr"
    keywords: List[str]  # Top representative keywords
    count: int           # Total number of sentences in this topic


@dataclass
class QuarterTopicDistribution:
    """Topic distribution for one quarter."""
    quarter_label: str
    topic_proportions: Dict[int, float]  # topic_id -> proportion (0-1)
    dominant_topic: int
    dominant_topic_label: str


@dataclass
class TopicShiftResult:
    """Complete topic analysis result."""
    topics: List[TopicInfo]                          # All discovered topics
    quarter_distributions: List[QuarterTopicDistribution]  # Per-quarter breakdown
    topics_over_time_df: Optional[object]            # Reserved for future use
    n_documents: int                                 # Total sentences processed
    n_topics: int                                    # Topics discovered (excl. outlier -1)
    # NEW in v4: representative sentences per (topic_id, quarter_label) -> top 5 sentences
    representative_sentences: Dict[str, List[str]] = field(default_factory=dict)
    # NEW in v4: QoQ delta proportions per topic
    qoq_deltas: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)
    # NEW: Raw sentence counts per topic per quarter (for Sankey flow widths)
    topic_sentence_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_embedding_model() -> SentenceTransformer:
    """Load sentence-transformer for embeddings."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_topic_shift(
    prepared_texts: Dict[str, str],
    qa_texts: Dict[str, str],
    embed_model: SentenceTransformer,
    min_topic_size: int = 5,
    max_topics: int = 12,
) -> TopicShiftResult:
    """
    Run BERTopic-style pipeline across all quarters to discover topics.

    Parameters:
    - prepared_texts: {quarter_label: PR text}
    - qa_texts: {quarter_label: QA text}
    - embed_model: Pre-loaded sentence-transformer.
    - min_topic_size: Minimum sentences to form a topic cluster.
    - max_topics: Maximum macro-topics after hierarchical reduction.

    Returns: TopicShiftResult with topics, distributions, rep sentences, and metadata.
    """
    labels = list(prepared_texts.keys())

    # Step 1: Split all transcripts into INDIVIDUAL SENTENCES
    documents = []
    timestamps = []  # Quarter label for each sentence

    for lbl in labels:
        pr_sentences = _split_into_sentences(prepared_texts.get(lbl, ""))
        documents.extend(pr_sentences)
        timestamps.extend([lbl] * len(pr_sentences))

        qa_sentences = _split_into_sentences(qa_texts.get(lbl, ""))
        documents.extend(qa_sentences)
        timestamps.extend([lbl] * len(qa_sentences))

    if len(documents) < 20:
        return TopicShiftResult(
            topics=[], quarter_distributions=[], topics_over_time_df=None,
            n_documents=len(documents), n_topics=0,
            representative_sentences={}, qoq_deltas={},
        )

    # Step 2: Generate embeddings for all sentences
    embeddings = embed_model.encode(documents, show_progress_bar=False, batch_size=64)

    # Step 3: Dimensionality reduction with PCA
    n_components = min(15, len(documents) - 1, embeddings.shape[1])
    pca_model = PCA(n_components=n_components, random_state=42)
    reduced_embeddings = pca_model.fit_transform(embeddings)

    # Step 4: HDBSCAN clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(min_topic_size, 5),
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = hdbscan_model.fit_predict(reduced_embeddings)

    # Step 5: Extract topic representations using c-TF-IDF
    topic_infos, ctfidf_matrix, feature_names = _extract_topic_representations_with_matrix(
        documents, cluster_labels
    )

    if not topic_infos:
        return TopicShiftResult(
            topics=[], quarter_distributions=[], topics_over_time_df=None,
            n_documents=len(documents), n_topics=0,
            representative_sentences={}, qoq_deltas={},
        )

    # Step 6: Hierarchical Topic Reduction
    # Merge similar topics if we have more than max_topics
    if len(topic_infos) > max_topics and ctfidf_matrix is not None:
        cluster_labels, topic_infos = _hierarchical_topic_reduction(
            documents=documents,
            cluster_labels=cluster_labels,
            topic_infos=topic_infos,
            ctfidf_matrix=ctfidf_matrix,
            max_topics=max_topics,
        )

    # Step 7: Compute per-quarter topic distributions + representative sentences
    quarter_distributions = []
    representative_sentences: Dict[str, List[str]] = {}
    topic_sentence_counts: Dict[str, int] = {}  # "tid|quarter" -> raw count

    for lbl in labels:
        indices = [i for i, t in enumerate(timestamps) if t == lbl]
        if not indices:
            continue

        quarter_topics = [cluster_labels[i] for i in indices]
        topic_counts = Counter(quarter_topics)
        total = sum(c for t, c in topic_counts.items() if t != -1)

        proportions = {}
        for tid, count in topic_counts.items():
            if tid == -1:
                continue
            proportions[tid] = count / max(total, 1)
            # Store raw sentence count for Sankey flow widths
            topic_sentence_counts[f"{tid}|{lbl}"] = count

        # Find dominant topic
        dominant = max(proportions.items(), key=lambda x: x[1]) if proportions else (0, 0)
        dominant_label = ""
        for ti in topic_infos:
            if ti.topic_id == dominant[0]:
                dominant_label = ti.label
                break

        quarter_distributions.append(QuarterTopicDistribution(
            quarter_label=lbl,
            topic_proportions=proportions,
            dominant_topic=dominant[0],
            dominant_topic_label=dominant_label,
        ))

        # Collect representative sentences per topic for this quarter
        # Use embeddings similarity to topic centroid to rank sentences
        for tid in proportions.keys():
            key = f"{tid}|{lbl}"
            # Get indices of sentences in this topic+quarter
            topic_quarter_indices = [
                i for i in indices if cluster_labels[i] == tid
            ]
            if not topic_quarter_indices:
                continue

            # Select top 5 representative sentences
            # Use distance to centroid in embedding space
            topic_embeddings = embeddings[topic_quarter_indices]
            centroid = topic_embeddings.mean(axis=0)
            distances = np.linalg.norm(topic_embeddings - centroid, axis=1)
            # Closest to centroid = most representative
            top_indices = distances.argsort()[:5]
            representative_sentences[key] = [
                documents[topic_quarter_indices[i]] for i in top_indices
            ]

    # Step 8: Calculate QoQ Delta Proportions
    qoq_deltas = _calculate_qoq_deltas(quarter_distributions)

    return TopicShiftResult(
        topics=topic_infos,
        quarter_distributions=quarter_distributions,
        topics_over_time_df=None,
        n_documents=len(documents),
        n_topics=len(topic_infos),
        representative_sentences=representative_sentences,
        qoq_deltas=qoq_deltas,
        topic_sentence_counts=topic_sentence_counts,
    )


# ---------------------------------------------------------------------------
# Internal: Hierarchical Topic Reduction
# ---------------------------------------------------------------------------

def _hierarchical_topic_reduction(
    documents: List[str],
    cluster_labels: np.ndarray,
    topic_infos: List[TopicInfo],
    ctfidf_matrix: np.ndarray,
    max_topics: int = 12,
) -> Tuple[np.ndarray, List[TopicInfo]]:
    """
    Merge similar topics via cosine similarity of c-TF-IDF vectors
    until we reach max_topics.

    Strategy: Iteratively merge the two most similar topics (highest cosine sim).
    After merging, reassign cluster labels and recompute topic info.
    """
    # Build mapping: topic_id -> row index in ctfidf_matrix
    current_topic_ids = [ti.topic_id for ti in topic_infos]
    id_to_idx = {tid: idx for idx, tid in enumerate(current_topic_ids)}

    # Work with dense matrix for similarity computation
    if hasattr(ctfidf_matrix, 'toarray'):
        dense_matrix = ctfidf_matrix.toarray()
    else:
        dense_matrix = np.array(ctfidf_matrix)

    # Build merge map: original_topic_id -> merged_topic_id
    merge_map = {tid: tid for tid in current_topic_ids}
    active_topics = list(current_topic_ids)

    while len(active_topics) > max_topics:
        if len(active_topics) <= 2:
            break

        # Compute pairwise cosine similarity between active topics
        active_indices = [id_to_idx[tid] for tid in active_topics]
        active_vectors = dense_matrix[active_indices]

        sim_matrix = cosine_similarity(active_vectors)
        # Zero out diagonal
        np.fill_diagonal(sim_matrix, -1)

        # Find most similar pair
        flat_idx = sim_matrix.argmax()
        i, j = divmod(flat_idx, sim_matrix.shape[1])

        # Merge topic j into topic i (keep i, remove j)
        keep_tid = active_topics[i]
        merge_tid = active_topics[j]

        # Update merge map: all topics that were mapped to merge_tid now map to keep_tid
        for orig_tid, mapped_tid in merge_map.items():
            if mapped_tid == merge_tid:
                merge_map[orig_tid] = keep_tid

        # Update the vector for the merged topic (average)
        keep_idx = id_to_idx[keep_tid]
        merge_idx = id_to_idx[merge_tid]
        dense_matrix[keep_idx] = (dense_matrix[keep_idx] + dense_matrix[merge_idx]) / 2.0

        # Remove merged topic from active list
        active_topics.remove(merge_tid)

    # Apply merge map to cluster labels
    new_cluster_labels = cluster_labels.copy()
    for orig_tid, new_tid in merge_map.items():
        if orig_tid != new_tid:
            new_cluster_labels[cluster_labels == orig_tid] = new_tid

    # Recompute topic info for merged topics
    merged_topic_infos = _extract_topic_representations(documents, new_cluster_labels)

    return new_cluster_labels, merged_topic_infos


# ---------------------------------------------------------------------------
# Internal: QoQ Delta Calculation
# ---------------------------------------------------------------------------

def _calculate_qoq_deltas(
    quarter_distributions: List[QuarterTopicDistribution],
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Calculate quarter-over-quarter delta proportion for each topic.

    Returns: {topic_id: [(quarter_label, delta_pct), ...]}
    where delta_pct = (proportion_Q[n] - proportion_Q[n-1]) / proportion_Q[n-1]
    expressed as percentage change (e.g., 3.0 = +300%).
    """
    if len(quarter_distributions) < 2:
        return {}

    # Collect all topic IDs
    all_topic_ids = set()
    for qd in quarter_distributions:
        all_topic_ids.update(qd.topic_proportions.keys())

    qoq_deltas: Dict[int, List[Tuple[str, float]]] = {}

    for tid in all_topic_ids:
        deltas = []
        for i in range(1, len(quarter_distributions)):
            prev_qd = quarter_distributions[i - 1]
            curr_qd = quarter_distributions[i]

            prev_prop = prev_qd.topic_proportions.get(tid, 0.0)
            curr_prop = curr_qd.topic_proportions.get(tid, 0.0)

            if prev_prop > 0.005:  # Only calculate if previous had meaningful presence
                delta_pct = (curr_prop - prev_prop) / prev_prop
            elif curr_prop > 0.01:
                # Topic appeared from near-zero -> treat as "new topic spike"
                delta_pct = 9.99  # Cap at 999%
            else:
                delta_pct = 0.0

            deltas.append((curr_qd.quarter_label, delta_pct))

        qoq_deltas[tid] = deltas

    return qoq_deltas


# ---------------------------------------------------------------------------
# Internal: c-TF-IDF Topic Extraction (with financial stopwords)
# ---------------------------------------------------------------------------

def _extract_topic_representations_with_matrix(
    documents: List[str],
    cluster_labels: np.ndarray,
    top_n_words: int = 8,
) -> Tuple[List[TopicInfo], Optional[object], Optional[object]]:
    """
    Extract topic keywords using c-TF-IDF and also return the matrix
    for hierarchical reduction.
    
    POS Lexical Pruning: Only PROPN and NOUN tokens are retained for
    keyword extraction, eliminating filler verbs/adverbs/adjectives.

    Returns: (topic_infos, ctfidf_matrix, feature_names)
    """
    unique_labels = sorted(set(cluster_labels))
    topic_ids = [l for l in unique_labels if l != -1]

    if not topic_ids:
        return [], None, None

    # Create per-cluster concatenated documents
    cluster_docs = []
    for tid in topic_ids:
        indices = [i for i, l in enumerate(cluster_labels) if l == tid]
        merged = " ".join(documents[i] for i in indices)
        cluster_docs.append(merged)

    # Apply POS filtering: retain only PROPN and NOUN tokens
    cluster_docs_filtered = _pos_filter_documents(cluster_docs)

    vectorizer = CountVectorizer(
        stop_words=FINANCIAL_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=8000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    try:
        count_matrix = vectorizer.fit_transform(cluster_docs_filtered)
    except ValueError:
        return [], None, None

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf.fit_transform(count_matrix)

    feature_names = vectorizer.get_feature_names_out()

    topic_infos = []
    for idx, tid in enumerate(topic_ids):
        scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_n_words]
        keywords = [feature_names[i] for i in top_indices if scores[i] > 0]

        label = "_".join(keywords[:4]) if keywords else f"Topic_{tid}"
        count = int(np.sum(cluster_labels == tid))

        topic_infos.append(TopicInfo(
            topic_id=tid,
            label=label,
            keywords=keywords,
            count=count,
        ))

    return topic_infos, tfidf_matrix, feature_names


def _extract_topic_representations(
    documents: List[str],
    cluster_labels: np.ndarray,
    top_n_words: int = 8,
) -> List[TopicInfo]:
    """
    Extract topic keywords using class-based TF-IDF (c-TF-IDF).
    Simplified version that only returns TopicInfo (used after merging).
    
    POS Lexical Pruning: Only PROPN and NOUN tokens are retained.
    """
    unique_labels = sorted(set(cluster_labels))
    topic_ids = [l for l in unique_labels if l != -1]

    if not topic_ids:
        return []

    cluster_docs = []
    for tid in topic_ids:
        indices = [i for i, l in enumerate(cluster_labels) if l == tid]
        merged = " ".join(documents[i] for i in indices)
        cluster_docs.append(merged)

    # Apply POS filtering: retain only PROPN and NOUN tokens
    cluster_docs_filtered = _pos_filter_documents(cluster_docs)

    vectorizer = CountVectorizer(
        stop_words=FINANCIAL_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=8000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    try:
        count_matrix = vectorizer.fit_transform(cluster_docs_filtered)
    except ValueError:
        return []

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf.fit_transform(count_matrix)

    feature_names = vectorizer.get_feature_names_out()

    topic_infos = []
    for idx, tid in enumerate(topic_ids):
        scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_n_words]
        keywords = [feature_names[i] for i in top_indices if scores[i] > 0]

        label = "_".join(keywords[:4]) if keywords else f"Topic_{tid}"
        count = int(np.sum(cluster_labels == tid))

        topic_infos.append(TopicInfo(
            topic_id=tid,
            label=label,
            keywords=keywords,
            count=count,
        ))

    return topic_infos


# ---------------------------------------------------------------------------
# Internal: Sentence Splitting
# ---------------------------------------------------------------------------

# Abbreviations that should NOT trigger sentence boundaries
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "inc", "ltd", "co",
    "corp", "vs", "etc", "approx", "dept", "est", "vol", "fig",
    "e.g", "i.e", "st", "ave", "blvd",
}

# Compiled regex for sentence boundary detection
_SENT_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'
)

# Pattern to detect speaker labels (e.g., "John Smith - CEO:" or "Operator:")
_SPEAKER_LABEL = re.compile(
    r'^[A-Z][a-zA-Z\s\-\.]+(?:\s*[-\u2013\u2014]\s*[A-Za-z\s,]+)?:\s*'
)


def _split_into_sentences(text: str) -> List[str]:
    """
    Split a transcript section into individual sentences.

    Strategy:
    1. Clean the text (remove speaker labels, normalize whitespace).
    2. Split on sentence boundaries (. ! ? followed by uppercase).
    3. Filter out sentences that are too short (< 6 words) or too long (> 60 words).
    4. Remove sentences that are purely procedural/moderator language.

    Returns: List of clean sentences suitable for embedding + clustering.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace and line breaks
    text = re.sub(r'\r\n', '\n', text)

    # Process line-by-line to handle speaker labels
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove speaker labels (e.g., "Shantanu Narayen - CEO:")
        line = _SPEAKER_LABEL.sub('', line)
        if line:
            clean_lines.append(line)

    # Rejoin into one text block
    full_text = ' '.join(clean_lines)

    # Normalize multiple spaces
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    if not full_text:
        return []

    # Split into sentences
    raw_sentences = _SENT_BOUNDARY.split(full_text)

    # Filter and clean sentences
    sentences = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Count words
        words = sent.split()
        word_count = len(words)

        # Skip too-short sentences (likely fragments)
        if word_count < 6:
            continue

        # Skip excessively long sentences (likely parsing errors)
        if word_count > 60:
            sub_parts = re.split(r'[;]\s+|(?<=\w)\s+[-\u2013\u2014]\s+(?=[A-Z])', sent)
            for part in sub_parts:
                part = part.strip()
                if len(part.split()) >= 6:
                    sentences.append(part)
        else:
            sentences.append(sent)

    return sentences
