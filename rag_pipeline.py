#!/usr/bin/env python
# =================================================================================================
# MODULE: RAG PIPELINE BENCHMARK (BASE-RETRIEVAL VS RE-RANKING)
# =================================================================================================
"""
Dual-Stage Retrieval-Augmented Generation (RAG) System over The Constitution of India.
=====================================================================================

This module implements, benchmarks, and analyzes two distinct RAG architectures:
  1. Pipeline 1 (Baseline Vector Retrieval):
     - Uses sentence-transformers/all-MiniLM-L6-v2 (Bi-Encoder, 384 dimensions)
     - Exact inner-product search on unit-normalized embeddings via FAISS IndexFlatIP
     - Retrieves Top-k candidates (default k=10) and supplies Top-n (default n=3) directly to LLM.
  2. Pipeline 2 (Two-Stage Re-Ranking Architecture):
     - Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (Cross-Encoder, 22.7M parameters)
     - Performs joint token-level cross-attention scoring across the Top-10 retrieved candidate chunks
     - Re-ranks candidates descending by relevance logits and supplies Top-3 to the local LLM.

Generation:
  - Local LLM: llama3.2:3b via Ollama API
  - Controlled generation: temperature = 0, fixed random seed (42), strict context-only legal prompting.

Corpus & Preprocessing:
  - Source: The Constitution of India (official PDF)
  - Operative text: Pages 32 through 381 (1-indexed, inclusive)
  - Layout-preserving extraction, footnote sanitization, semantic clause chunking (900 chars, 150 overlap).

Deliverables & Artifacts Generated:
  - results/run_config.json          : Comprehensive system configuration, environment, and dependencies
  - results/queries.json             : Structured benchmark query specifications and ground truth
  - results/cache_manifest.json      : SHA-256 cryptographic cache manifest for chunking & embeddings
  - results/rank_shift.csv           : Per-candidate rank shifts between baseline and re-ranking
  - results/metrics_summary.csv      : High-level retrieval accuracy metrics with 95% bootstrap CI
  - results/metrics_by_category.csv  : Performance breakdown across 8 distinct legal query categories
  - results/status_counts.csv        : Outcome distribution (Fixed, Degraded, Both OK, Both Fail)
  - results/answers_comparison.csv   : Direct textual and factual comparison of Answer 1 vs Answer 2
  - results/latency.csv              : Microsecond-precision stage-wise latency profiles per query
  - results/latency_summary.csv      : Median and mean latency breakdown across all pipeline stages
  - results/latency_scaling.csv      : Empirical CPU latency scaling measurements across corpus subsets
  - results/latency_projection.csv   : Theoretical linear latency extrapolation up to 10M chunks
  - results/candidate_pool_sweep.csv : Sensitivity sweep of candidate pool size k in {3, 5, 10, 20}
  - results/report_tables.md         : Publication-ready Markdown tables and qualitative failure studies
  - results/figures/rank_shift.png   : Visualization of candidate rank promotions into the LLM window
  - results/figures/latency.png      : Visual comparison of baseline vs re-ranking retrieval latencies
  - results/figures/scaling.png      : Log-log scaling plot comparing vector search vs exhaustive CE
"""

# %% SECTION 1: GLOBAL IMPORTS & EXPERIMENTAL CONFIGURATION
# =================================================================================================
# SECTION 1: GLOBAL IMPORTS & EXPERIMENTAL CONFIGURATION
# =================================================================================================
# Import standard library utilities for operating system interactions, cryptographic hashing,
# mathematical statistics, process management, and JSON manipulation.
import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Import scientific computing and numerical data structures.
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------
# 1.1 Model Checkpoint Identifiers & Architectural Parameters
# -------------------------------------------------------------------------------------------------
# Bi-Encoder: Pre-trained SentenceTransformer that maps arbitrary text into a 384-dimensional
# dense vector space. Trained via contrastive loss on 1B+ sentence pairs for symmetric semantic similarity.
BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-Encoder: Pre-trained transformer classifier that accepts concatenated (query, document) pairs
# and performs full cross-attention across all token pairs, directly predicting relevance logits.
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Downstream Generative Large Language Model executed locally via Ollama server.
LLM_MODEL = "llama3.2:3b"

# Local path to the primary PDF document source.
PDF_PATH = "constitution_of_india.pdf"

# -------------------------------------------------------------------------------------------------
# 1.2 Corpus Page Bounds & Text Segmentation Hyperparameters
# -------------------------------------------------------------------------------------------------
# Pages 1-31 contain front matter (Preamble, detailed Table of Contents, Amendment history tables).
# Pages 382+ contain appendix tables, state lists, and historical indexes.
# Restricting to pages 32-381 isolates substantive constitutional Articles and Schedules.
FIRST_PAGE, LAST_PAGE = 32, 381  # 1-indexed, inclusive bounds

# Target chunk length in characters (~180-220 tokens) and overlap to preserve clause boundary coherence.
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

# Global deterministic pseudo-random seed to ensure 100% reproducible sampling and bootstrap runs.
SEED = 42

# -------------------------------------------------------------------------------------------------
# 1.3 System Prompt with Strict Negative Constraints
# -------------------------------------------------------------------------------------------------
# The system prompt enforces:
#  1. Strict evidence grounding: Answer ONLY from provided context chunks.
#  2. Verbatim abstention syntax: Triggers explicit abstention when context lacks proof, eliminating hallucination.
#  3. Citation requirement: Explicitly demands Article numbers whenever present in the text.
#  4. Conciseness budget: Upper bound of 4 sentences to prevent verbose rambling.
SYSTEM_PROMPT = (
    "You are a careful legal assistant for the Constitution of India. Answer the question "
    "using ONLY the context passages provided. If the context does not contain the answer, "
    "reply exactly: 'The provided context does not contain the answer.' "
    "Be concise (at most 4 sentences) and cite Article numbers when the context states them."
)


def build_user_prompt(query, chunks):
    """
    Constructs an identical, standardized user prompt across both Pipeline 1 and Pipeline 2.

    Formatting the context consistently ensures that differences in LLM answer quality
    between Pipeline 1 (Baseline) and Pipeline 2 (Re-Ranked) are strictly attributable to
    the retrieval quality and chunk ordering, rather than prompt variations.

    Args:
        query (str): The natural language constitutional question.
        chunks (list[dict]): The Top-n retrieved chunk dictionaries (each containing 'text',
                             'chunk_id', and 'page_start').

    Returns:
        str: Standardized context prompt with labeled passage blocks followed by the question.
    """
    # Format each context passage with its 1-indexed relative rank, absolute chunk ID, and source page
    context_blocks = []
    for rank_idx, chunk_data in enumerate(chunks, 1):
        block_header = f"[Passage {rank_idx} | chunk {chunk_data['chunk_id']} | page {chunk_data['page_start']}]"
        context_blocks.append(f"{block_header}\n{chunk_data['text']}")

    formatted_context = "\n\n".join(context_blocks)
    return f"Context:\n{formatted_context}\n\nQuestion: {query}\nAnswer:"


# %% SECTION 2: BENCHMARK QUERY TAXONOMY & TEST SUITE
# =================================================================================================
# SECTION 2: BENCHMARK QUERY TAXONOMY & TEST SUITE
# =================================================================================================
# The benchmark comprises 56 manually verified queries balanced across eight distinct categories:
#   1. Control (C1-C7)       : Direct single-article recall queries; tests baseline retrieval capacity.
#   2. Keyword Trap (K1-K7)  : Lexical divergence; queries sharing keywords with deceptive non-gold articles.
#   3. Exception (E1-E7)     : General constitutional rules overridden by specific subordinate clauses.
#   4. Negation (N1-N7)      : Prohibitions, bars on jurisdiction, and negative statutory disqualifications.
#   5. Role Limit (R1-R7)    : Specific numeric thresholds, age qualifications, and institutional term limits.
#   6. Procedure (P1-P7)     : Multi-step voting thresholds, impeachment mechanisms, and legislative passage.
#   7. Cross-Article (X1-X7) : Synthesis requiring discrimination between contrasting constitutional powers.
#   8. Unanswerable (U1-U7)  : Out-of-corpus queries testing rigorous LLM abstention and non-hallucination.

CURATED_QUERIES = [
    # =========================================================================
    # 1. CONTROL (Direct factual queries; straightforward single-article search)
    # =========================================================================
    dict(
        id="C1", category="Control",
        query="What is the minimum age to be eligible for election as President of India?",
        expected_answer="Under Article 58(1)(b), a person must have completed the age of thirty-five (35) years to be eligible for election as President of India.",
        gold=r"completed the age of thirty-five years",
        gold_sufficiency="any", source_citation="Article 58(1)(b), p. 56", is_unanswerable=False,
        required_facts=[dict(claim="35 years of age", pattern=r"35|thirty-five"),
                        dict(claim="Article 58 citation", pattern=r"58")]
    ),
    dict(
        id="C2", category="Control",
        query="Which article protects life and personal liberty and what exception does it allow?",
        expected_answer="Article 21 protects life and personal liberty, providing that no person shall be deprived of life or personal liberty except according to procedure established by law.",
        gold=r"deprived of his life or personal liberty except according to procedure established by law",
        gold_sufficiency="any", source_citation="Article 21, p. 42", is_unanswerable=False,
        required_facts=[dict(claim="Article 21", pattern=r"21"),
                        dict(claim="Procedure established by law", pattern=r"procedure established by law")]
    ),
    dict(
        id="C3", category="Control",
        query="How long can an arrested person be detained before being produced before a magistrate, and what journey-time exception applies?",
        expected_answer="Under Article 22(2), every person arrested and detained in custody must be produced before the nearest magistrate within twenty-four hours of arrest, excluding the time necessary for the journey from the place of arrest to the court of the magistrate.",
        gold=r"within a period of twenty-four hours of such arrest excluding the time necessary for the journey",
        gold_sufficiency="any", source_citation="Article 22(2), p. 42-43", is_unanswerable=False,
        required_facts=[dict(claim="24 hours", pattern=r"24|twenty-four"),
                        dict(claim="Excluding journey time", pattern=r"journey|travel|excluding"),
                        dict(claim="Nearest magistrate", pattern=r"magistrate"),
                        dict(claim="Article 22", pattern=r"22")]
    ),
    dict(
        id="C4", category="Control",
        query="What is the ordinary duration of the House of the People, and under what condition and for how long can it be extended?",
        expected_answer="Under Article 83(2), the House of the People continues for five years from the date appointed for its first meeting. While a Proclamation of Emergency is in operation, Parliament may by law extend this period for not exceeding one year at a time, and not extending beyond six months after the proclamation ceases to operate.",
        gold=r"continue for five years from the date appointed for its first meeting",
        gold_sufficiency="any", source_citation="Article 83(2), p. 70", is_unanswerable=False,
        required_facts=[dict(claim="Ordinary term 5 years", pattern=r"5|five\s*years"),
                        dict(claim="Emergency extension up to 1 year at a time", pattern=r"one year|1 year|emergency"),
                        dict(claim="Article 83", pattern=r"83")]
    ),
    dict(
        id="C5", category="Control",
        query="Who acts as the ex-officio Chairman of the Council of States?",
        expected_answer="Under Articles 64 and 89(1), the Vice-President of India shall be ex-officio Chairman of the Council of States.",
        gold=r"ex[-\s]*officio Chairman of the Council",
        gold_sufficiency="any", source_citation="Article 64 / Article 89(1), p. 61, 72", is_unanswerable=False,
        required_facts=[dict(claim="Vice-President", pattern=r"vice-president|vice president"),
                        dict(claim="Article 64 or 89", pattern=r"64|89")]
    ),
    dict(
        id="C6", category="Control",
        query="To whom is the Council of Ministers collectively responsible?",
        expected_answer="Under Article 75(3), the Council of Ministers shall be collectively responsible to the House of the People.",
        gold=r"Council of Ministers shall be collectively responsible to the House of the People",
        gold_sufficiency="any", source_citation="Article 75(3), p. 66", is_unanswerable=False,
        required_facts=[dict(claim="House of the People / Lok Sabha", pattern=r"house of the people|lok sabha"),
                        dict(claim="Article 75", pattern=r"75")]
    ),
    dict(
        id="C7", category="Control",
        query="In which House of Parliament can a Money Bill NOT be introduced?",
        expected_answer="Under Article 109(1), a Money Bill shall not be introduced in the Council of States (Rajya Sabha).",
        gold=r"Money Bill shall not be introduced in the Council of States",
        gold_sufficiency="any", source_citation="Article 109(1), p. 80", is_unanswerable=False,
        required_facts=[dict(claim="Council of States / Rajya Sabha", pattern=r"council of states|rajya sabha"),
                        dict(claim="Cannot be introduced", pattern=r"not|cannot"),
                        dict(claim="Article 109", pattern=r"109")]
    ),

    # =========================================================================
    # 2. KEYWORD TRAP / DISTRACTOR (Lexical divergence, distinct terminology)
    # =========================================================================
    dict(
        id="K1", category="Keyword trap",
        query="Is an Indian citizen allowed to accept a knighthood or honour from a foreign government?",
        expected_answer="Under Article 18(2), no citizen of India shall accept any title from any foreign State.",
        gold=r"No citizen of India shall accept any title from any foreign State",
        gold_sufficiency="any", source_citation="Article 18(2), p. 40", is_unanswerable=False,
        required_facts=[dict(claim="No citizen shall accept", pattern=r"no|not|cannot|prohibit"),
                        dict(claim="Title from foreign State", pattern=r"title"),
                        dict(claim="Article 18", pattern=r"18")]
    ),
    dict(
        id="K2", category="Keyword trap",
        query="Who has the power to pardon a person sentenced to death?",
        expected_answer="Under Article 72(1)(c), the President has the power to grant pardons, reprieves, respites or remissions of punishment or to suspend, remit or commute the sentence of any person in all cases where the sentence is a sentence of death.",
        gold=r"in all cases where the sentence is a sentence of death",
        gold_sufficiency="any", source_citation="Article 72(1)(c), p. 64", is_unanswerable=False,
        required_facts=[dict(claim="President", pattern=r"president"),
                        dict(claim="Article 72", pattern=r"72")]
    ),
    dict(
        id="K3", category="Keyword trap",
        query="How can the head of the Indian state be thrown out of office before the term ends?",
        expected_answer="Under Article 61, the President (head of state) can be removed from office before term expiration through impeachment for violation of the Constitution, upon a resolution passed by a two-thirds majority of the total membership of each House.",
        gold=r"Procedure for impeachment of the President",
        gold_sufficiency="any", source_citation="Article 61, p. 60", is_unanswerable=False,
        required_facts=[dict(claim="Impeachment procedure", pattern=r"impeach"),
                        dict(claim="Violation of Constitution", pattern=r"violation of the constitution"),
                        dict(claim="Article 61", pattern=r"61")]
    ),
    dict(
        id="K4", category="Keyword trap",
        query="Who picks the top judge of the apex court?",
        expected_answer="Under Article 124(2), every Judge of the Supreme Court (including the Chief Justice of India) shall be appointed by the President by warrant under his hand and seal.",
        gold=r"Judge of the Supreme Court shall be appointed by the President by warrant under his hand and seal",
        gold_sufficiency="any", source_citation="Article 124(2), p. 88", is_unanswerable=False,
        required_facts=[dict(claim="President", pattern=r"president"),
                        dict(claim="Article 124", pattern=r"124")]
    ),
    dict(
        id="K5", category="Keyword trap",
        query="Can the police force of one State investigate or exercise jurisdiction in another State without permission?",
        expected_answer="Under the Seventh Schedule (Union List, Entry 80), powers and jurisdiction of a State police force cannot be extended to another State so as to enable police to exercise powers without the consent of the Government of that State.",
        gold=r"without the consent of the Government of the State in which such area is situated",
        gold_sufficiency="any", source_citation="Seventh Schedule, List I, Entry 80, p. 346", is_unanswerable=False,
        required_facts=[dict(claim="Requires consent of State government", pattern=r"consent"),
                        dict(claim="Cannot exercise without consent", pattern=r"without|not|consent")]
    ),
    dict(
        id="K6", category="Keyword trap",
        query="Is private property completely immune from government acquisition?",
        expected_answer="Under Article 300A, property is not immune: no person shall be deprived of his property save by authority of law.",
        gold=r"No person shall be deprived of his property save by authority of law",
        gold_sufficiency="any", source_citation="Article 300A, p. 201-202", is_unanswerable=False,
        required_facts=[dict(claim="Authority of law allows deprivation", pattern=r"authority of law"),
                        dict(claim="Article 300A", pattern=r"300A")]
    ),
    dict(
        id="K7", category="Keyword trap",
        query="What happens when the government of a province or region breaks down and cannot function under constitutional rules?",
        expected_answer="Under Article 356, if the President is satisfied that a situation has arisen where the Government of the State cannot be carried on in accordance with the Constitution, the President may issue a Proclamation assuming state powers (President's Rule).",
        gold=r"Government of the State cannot be carried on in accordance with the provisions of this Constitution",
        gold_sufficiency="any", source_citation="Article 356(1), p. 242", is_unanswerable=False,
        required_facts=[dict(claim="President's proclamation / rule", pattern=r"president|proclamation"),
                        dict(claim="Article 356", pattern=r"356")]
    ),

    # =========================================================================
    # 3. EXCEPTIONS & CORNER CASES (General rule overridden by specific provisos)
    # =========================================================================
    dict(
        id="E1", category="Exception",
        query="Are there any persons to whom the protections against arrest and detention in Article 22 do not apply?",
        expected_answer="Under Article 22(3), the arrest protections do not apply to: (a) any person who for the time being is an enemy alien, or (b) any person arrested or detained under any law providing for preventive detention.",
        gold=r"enemy alien",
        gold_sufficiency="any", source_citation="Article 22(3), p. 43", is_unanswerable=False,
        required_facts=[dict(claim="Enemy alien", pattern=r"enemy alien"),
                        dict(claim="Preventive detention", pattern=r"preventive detention"),
                        dict(claim="Article 22", pattern=r"22")]
    ),
    dict(
        id="E2", category="Exception",
        query="Despite equality of opportunity in public employment, can the State reserve posts for some groups?",
        expected_answer="Under Article 16(4), nothing shall prevent the State from making any provision for the reservation of appointments or posts in favour of any backward class of citizens not adequately represented in State services.",
        gold=r"reservation of appointments or posts in favour of any backward class",
        gold_sufficiency="any", source_citation="Article 16(4), p. 39", is_unanswerable=False,
        required_facts=[dict(claim="Backward class reservation", pattern=r"backward class|reservation"),
                        dict(claim="Article 16", pattern=r"16")]
    ),
    dict(
        id="E3", category="Exception",
        query="What is the longest period preventive detention can last without an Advisory Board review?",
        expected_answer="Under Article 22(4), no law providing for preventive detention shall authorise detention for a longer period than three months unless an Advisory Board has reported sufficient cause before the expiration of three months.",
        gold=r"longer period than three months unless",
        gold_sufficiency="any", source_citation="Article 22(4), p. 43", is_unanswerable=False,
        required_facts=[dict(claim="Three months maximum", pattern=r"three months|3 months"),
                        dict(claim="Advisory Board requirement", pattern=r"advisory board"),
                        dict(claim="Article 22", pattern=r"22")]
    ),
    dict(
        id="E4", category="Exception",
        query="Can the right to move any court for the enforcement of fundamental rights be suspended during an emergency, and which rights are exempt from suspension?",
        expected_answer="Under Article 359(1), the President may by order suspend the right to move any court for enforcement of fundamental rights during an emergency, except the rights conferred by Articles 20 and 21, which can never be suspended.",
        gold=r"right to move any court for the enforcement of such of the rights conferred by Part III \(except articles 20 and 21\)",
        gold_sufficiency="any", source_citation="Article 359(1), p. 247", is_unanswerable=False,
        required_facts=[dict(claim="President may suspend by order", pattern=r"president|order|suspend"),
                        dict(claim="Articles 20 and 21 exempt", pattern=r"20.*21|21.*20|articles?\s*20\s*and\s*21"),
                        dict(claim="Article 359", pattern=r"359")]
    ),
    dict(
        id="E5", category="Exception",
        query="Under what circumstances can Parliament legislate on a subject enumerated in the State List in the national interest?",
        expected_answer="Under Article 249(1), Parliament can legislate on a State List subject if the Council of States declares by resolution supported by not less than two-thirds of the members present and voting that it is necessary or expedient in the national interest.",
        gold=r"necessary or expedient in the national interest that Parliament should make laws with respect to",
        gold_sufficiency="any", source_citation="Article 249(1), p. 177", is_unanswerable=False,
        required_facts=[dict(claim="Council of States / Rajya Sabha resolution", pattern=r"council of states|rajya sabha"),
                        dict(claim="Two-thirds present and voting", pattern=r"two-thirds|2/3"),
                        dict(claim="National interest", pattern=r"national interest"),
                        dict(claim="Article 249", pattern=r"249")]
    ),
    dict(
        id="E6", category="Exception",
        query="What grounds of reasonable restrictions can the State impose on freedom of speech and expression?",
        expected_answer="Under Article 19(2), the State may impose reasonable restrictions on freedom of speech in the interests of the sovereignty and integrity of India, security of the State, friendly relations with foreign States, public order, decency or morality, or in relation to contempt of court, defamation or incitement to an offence.",
        gold=r"friendly relations with foreign States, public order, decency or morality",
        gold_sufficiency="any", source_citation="Article 19(2), p. 41", is_unanswerable=False,
        required_facts=[dict(claim="Public order or security", pattern=r"public order|security"),
                        dict(claim="Sovereignty and integrity", pattern=r"sovereignty|integrity"),
                        dict(claim="Article 19", pattern=r"19")]
    ),
    dict(
        id="E7", category="Exception",
        query="Under what circumstances can the President promulgate an Ordinance, and what is its maximum lifespan after Parliament reassembles?",
        expected_answer="Under Article 123, when both Houses of Parliament are not in session, the President may promulgate Ordinances if satisfied immediate action is necessary. An Ordinance ceases to operate at the expiration of six weeks from the reassembly of Parliament, unless disapproved earlier.",
        gold=r"shall cease to operate at the expiration of six weeks from the reassembly of Parliament",
        gold_sufficiency="any", source_citation="Article 123(2)(a), p. 87-88", is_unanswerable=False,
        required_facts=[dict(claim="When Parliament not in session", pattern=r"not in session|recess"),
                        dict(claim="Six weeks from reassembly", pattern=r"six weeks|6 weeks"),
                        dict(claim="Article 123", pattern=r"123")]
    ),

    # =========================================================================
    # 4. NEGATION & CONSTRAINTS (Explicit prohibitions, 'no person shall', disqualifications)
    # =========================================================================
    dict(
        id="N1", category="Negation",
        query="Is forced labour allowed in India?",
        expected_answer="Under Article 23(1), traffic in human beings and begar and other similar forms of forced labour are prohibited and any contravention is an offence punishable in accordance with law.",
        gold=r"begar and other similar forms of forced labour",
        gold_sufficiency="any", source_citation="Article 23(1), p. 43", is_unanswerable=False,
        required_facts=[dict(claim="Prohibited", pattern=r"prohibit|not|forbid|unlawful"),
                        dict(claim="Begar / forced labour", pattern=r"forced labour|begar"),
                        dict(claim="Article 23", pattern=r"23")]
    ),
    dict(
        id="N2", category="Negation",
        query="Can a child below fourteen be employed in a factory?",
        expected_answer="Under Article 24, no child below the age of fourteen years shall be employed to work in any factory or mine or engaged in any other hazardous employment.",
        gold=r"No child below the age of fourteen years shall be employed to work in any factory",
        gold_sufficiency="any", source_citation="Article 24, p. 44", is_unanswerable=False,
        required_facts=[dict(claim="Prohibited / no child", pattern=r"not|no|prohibit"),
                        dict(claim="Below fourteen years", pattern=r"fourteen|14"),
                        dict(claim="Article 24", pattern=r"24")]
    ),
    dict(
        id="N3", category="Negation",
        query="Can a Governor also be a Member of Parliament?",
        expected_answer="Under Article 158(1), the Governor shall not be a member of either House of Parliament or of a House of the Legislature of any State.",
        gold=r"The Governor shall not be a member of either House",
        gold_sufficiency="any", source_citation="Article 158(1), p. 98", is_unanswerable=False,
        required_facts=[dict(claim="Cannot be a member", pattern=r"not|cannot|no"),
                        dict(claim="Article 158", pattern=r"158")]
    ),
    dict(
        id="N4", category="Negation",
        query="Can educational institutions fully funded by the State provide religious instruction?",
        expected_answer="Under Article 28(1), no religious instruction shall be provided in any educational institution wholly maintained out of State funds.",
        gold=r"No religious instruction shall be provided in any educational institution wholly maintained out of State funds",
        gold_sufficiency="any", source_citation="Article 28(1), p. 45", is_unanswerable=False,
        required_facts=[dict(claim="No religious instruction allowed", pattern=r"not|no|cannot|prohibit"),
                        dict(claim="Wholly maintained out of State funds", pattern=r"state funds"),
                        dict(claim="Article 28", pattern=r"28")]
    ),
    dict(
        id="N5", category="Negation",
        query="What are the specific constitutional grounds for disqualification from being chosen as or being a member of either House of Parliament?",
        expected_answer="Under Article 102(1), grounds include: holding an office of profit under the Government (other than declared exempt by Parliament), being of unsound mind declared by a competent court, being an undischarged insolvent, not being a citizen of India (or voluntary allegiance to a foreign State), or being disqualified under any law made by Parliament.",
        gold=r"disqualified for being chosen as, and for being, a member of either House of Parliament",
        gold_sufficiency="any", source_citation="Article 102(1), p. 76-77", is_unanswerable=False,
        required_facts=[dict(claim="Office of profit", pattern=r"office of profit"),
                        dict(claim="Unsound mind or insolvent", pattern=r"unsound mind|insolvent"),
                        dict(claim="Citizenship / allegiance to foreign state", pattern=r"citizen|foreign state"),
                        dict(claim="Article 102", pattern=r"102")]
    ),
    dict(
        id="N6", category="Negation",
        query="Can a person be prosecuted and punished more than once for the same offence?",
        expected_answer="Under Article 20(2), no person shall be prosecuted and punished for the same offence more than once (protection against double jeopardy).",
        gold=r"No person shall be prosecuted and punished for the same offence more than once",
        gold_sufficiency="any", source_citation="Article 20(2), p. 41", is_unanswerable=False,
        required_facts=[dict(claim="No person prosecuted more than once", pattern=r"no|not|cannot|prohibit"),
                        dict(claim="Same offence / double jeopardy", pattern=r"same offence|double jeopardy"),
                        dict(claim="Article 20", pattern=r"20")]
    ),
    dict(
        id="N7", category="Negation",
        query="Can an accused person be compelled to be a witness against himself in a criminal proceeding?",
        expected_answer="Under Article 20(3), no person accused of any offence shall be compelled to be a witness against himself (protection against self-incrimination).",
        gold=r"No person accused of any offence shall be compelled to be a witness against himself",
        gold_sufficiency="any", source_citation="Article 20(3), p. 41", is_unanswerable=False,
        required_facts=[dict(claim="Cannot be compelled", pattern=r"no|not|cannot|compelled"),
                        dict(claim="Witness against himself", pattern=r"witness against himself|self-incrimination"),
                        dict(claim="Article 20", pattern=r"20")]
    ),

    # =========================================================================
    # 5. ROLE & NUMERIC LIMITS (Ages, quorums, fractions, tenures, time spans)
    # =========================================================================
    dict(
        id="R1", category="Role limit",
        query="At what age must a Supreme Court judge retire?",
        expected_answer="Under Article 124(2), a Judge of the Supreme Court shall hold office until he attains the age of sixty-five (65) years.",
        gold=r"hold office until he attains the age of sixty-five years",
        gold_sufficiency="any", source_citation="Article 124(2), p. 88", is_unanswerable=False,
        required_facts=[dict(claim="65 years of age", pattern=r"65|sixty-five"),
                        dict(claim="Article 124", pattern=r"124")]
    ),
    dict(
        id="R2", category="Role limit",
        query="What is the minimum age to become a member of the Council of States?",
        expected_answer="Under Article 84(b), a person must be not less than thirty (30) years of age to fill a seat in the Council of States (Rajya Sabha).",
        gold=r"in the case of a seat in the Council of States, not less than thirty years of age",
        gold_sufficiency="any", source_citation="Article 84(b), p. 70", is_unanswerable=False,
        required_facts=[dict(claim="30 years of age", pattern=r"30|thirty"),
                        dict(claim="Article 84", pattern=r"84")]
    ),
    dict(
        id="R3", category="Role limit",
        query="How many members can the President nominate to the Council of States?",
        expected_answer="Under Article 80(1)(a), the Council of States consists of twelve (12) members to be nominated by the President having special knowledge in literature, science, art, and social service.",
        gold=r"twelve members to be nominated by the President",
        gold_sufficiency="any", source_citation="Article 80(1)(a), p. 69", is_unanswerable=False,
        required_facts=[dict(claim="12 members", pattern=r"12|twelve"),
                        dict(claim="Article 80", pattern=r"80")]
    ),
    dict(
        id="R4", category="Role limit",
        query="What is the maximum gap allowed between two sessions of Parliament?",
        expected_answer="Under Article 85(1), six months shall not intervene between its last sitting in one session and the date appointed for its first sitting in the next session.",
        gold=r"six months shall not intervene between its last sitting in one session and the date appointed for its first sitting in the next session",
        gold_sufficiency="any", source_citation="Article 85(1), p. 70", is_unanswerable=False,
        required_facts=[dict(claim="Six months", pattern=r"six months|6 months"),
                        dict(claim="Article 85", pattern=r"85")]
    ),
    dict(
        id="R5", category="Role limit",
        query="What qualifications are required to be appointed as a Judge of the Supreme Court of India?",
        expected_answer="Under Article 124(3), a person must be a citizen of India and has been for at least 5 years a Judge of a High Court, or for at least 10 years an advocate of a High Court, or is in the opinion of the President a distinguished jurist.",
        gold=r"at least five years a Judge of a High Court",
        gold_sufficiency="any", source_citation="Article 124(3), p. 89", is_unanswerable=False,
        required_facts=[dict(claim="Citizen of India", pattern=r"citizen"),
                        dict(claim="5 years judge or 10 years advocate", pattern=r"five years|5 years|ten years|10 years"),
                        dict(claim="Distinguished jurist", pattern=r"distinguished jurist|president"),
                        dict(claim="Article 124", pattern=r"124")]
    ),
    dict(
        id="R6", category="Role limit",
        query="What fraction of a House's members forms the quorum to constitute a meeting of Parliament?",
        expected_answer="Under Article 100(3), the quorum to constitute a meeting of either House of Parliament shall be one-tenth of the total number of members of the House.",
        gold=r"one-tenth of the total number of members of the House",
        gold_sufficiency="any", source_citation="Article 100(3), p. 75", is_unanswerable=False,
        required_facts=[dict(claim="One-tenth", pattern=r"one-tenth|1/10|ten"),
                        dict(claim="Article 100", pattern=r"100")]
    ),
    dict(
        id="R7", category="Role limit",
        query="Until what age can a High Court judge hold office?",
        expected_answer="Under Article 217(1), a Judge of a High Court shall hold office until he attains the age of sixty-two (62) years.",
        gold=r"attains the age of sixty-two years",
        gold_sufficiency="any", source_citation="Article 217(1), p. 131", is_unanswerable=False,
        required_facts=[dict(claim="62 years of age", pattern=r"62|sixty-two"),
                        dict(claim="Article 217", pattern=r"217")]
    ),

    # =========================================================================
    # 6. PROCEDURAL & MULTI-CONDITION (Complex constitutional mechanisms)
    # =========================================================================
    dict(
        id="P1", category="Procedure",
        query="What majority is required in Parliament to pass a constitutional amendment under Article 368?",
        expected_answer="Under Article 368(2), an amendment Bill must be passed in each House by a majority of the total membership of that House AND by a majority of not less than two-thirds of the members of that House present and voting.",
        gold=r"majority of the total membership of that House and by a majority of not less than two-thirds of the members of that House present and voting",
        gold_sufficiency="any", source_citation="Article 368(2), p. 259", is_unanswerable=False,
        required_facts=[dict(claim="Majority of total membership", pattern=r"total membership"),
                        dict(claim="Two-thirds present and voting", pattern=r"two-thirds|2/3"),
                        dict(claim="Article 368", pattern=r"368")]
    ),
    dict(
        id="P2", category="Procedure",
        query="What is the procedure and required notice period to move a resolution for the impeachment of the President?",
        expected_answer="Under Article 61(2), the proposal must be contained in a resolution moved after at least fourteen days' notice in writing signed by not less than one-fourth of the total number of members of the House, and passed by a majority of not less than two-thirds of the total membership of the House.",
        gold=r"fourteen days' notice in writing signed by not less than one-fourth of the total number of members",
        gold_sufficiency="any", source_citation="Article 61(2), p. 60", is_unanswerable=False,
        required_facts=[dict(claim="14 days notice", pattern=r"fourteen|14"),
                        dict(claim="One-fourth members sign", pattern=r"one-fourth|1/4"),
                        dict(claim="Two-thirds total membership", pattern=r"two-thirds|2/3"),
                        dict(claim="Article 61", pattern=r"61")]
    ),
    dict(
        id="P3", category="Procedure",
        query="How long can the Council of States delay a Money Bill before it is deemed passed by both Houses?",
        expected_answer="Under Article 109(5), if a Money Bill is not returned to the House of the People within fourteen days, it shall be deemed to have been passed by both Houses at the expiration of the said period in the form in which it was passed by the House of the People.",
        gold=r"within a period of fourteen days from the date of its receipt of the Bill",
        gold_sufficiency="any", source_citation="Article 109(5), p. 80", is_unanswerable=False,
        required_facts=[dict(claim="14 days limit", pattern=r"fourteen|14"),
                        dict(claim="Deemed passed", pattern=r"deemed.*passed|passed"),
                        dict(claim="Article 109", pattern=r"109")]
    ),
    dict(
        id="P4", category="Procedure",
        query="How long can a Proclamation of Emergency continue without approval by resolutions of both Houses of Parliament?",
        expected_answer="Under Article 352(4), every Proclamation of Emergency shall cease to operate at the expiration of one month unless before that expiration it has been approved by resolutions of both Houses of Parliament.",
        gold=r"cease to operate at the expiration of one month",
        gold_sufficiency="any", source_citation="Article 352(4), p. 240", is_unanswerable=False,
        required_facts=[dict(claim="One month", pattern=r"one month|1 month"),
                        dict(claim="Approval of both Houses", pattern=r"both houses|resolutions"),
                        dict(claim="Article 352", pattern=r"352")]
    ),
    dict(
        id="P5", category="Procedure",
        query="Under what conditions can the President declare a Financial Emergency under Article 360?",
        expected_answer="Under Article 360(1), if the President is satisfied that a situation has arisen whereby the financial stability or credit of India or of any part of the territory thereof is threatened, he may declare a Financial Emergency by Proclamation.",
        gold=r"financial stability or credit of India or of any part of the territory thereof is threatened",
        gold_sufficiency="any", source_citation="Article 360(1), p. 248", is_unanswerable=False,
        required_facts=[dict(claim="Financial stability or credit threatened", pattern=r"financial stability|credit"),
                        dict(claim="President satisfaction", pattern=r"president"),
                        dict(claim="Article 360", pattern=r"360")]
    ),
    dict(
        id="P6", category="Procedure",
        query="Under what conditions can the President summon a joint sitting of both Houses of Parliament?",
        expected_answer="Under Article 108(1), if after a Bill has been passed by one House and transmitted to the other: (a) the Bill is rejected, (b) the Houses finally disagree as to amendments, or (c) more than six months elapse without being passed, the President may summon a joint sitting (except for a Money Bill).",
        gold=r"Joint sitting of both Houses in certain cases",
        gold_sufficiency="any", source_citation="Article 108(1), p. 79-80", is_unanswerable=False,
        required_facts=[dict(claim="Deadlock conditions / six months", pattern=r"rejected|disagree|six months"),
                        dict(claim="President summons", pattern=r"president"),
                        dict(claim="Article 108", pattern=r"108")]
    ),
    dict(
        id="P7", category="Procedure",
        query="What is the official language of the Union and what script must be used?",
        expected_answer="Under Article 343(1), the official language of the Union shall be Hindi in Devanagari script.",
        gold=r"official language of the Union shall be Hindi in Devanagari script",
        gold_sufficiency="any", source_citation="Article 343(1), p. 235", is_unanswerable=False,
        required_facts=[dict(claim="Hindi", pattern=r"hindi"),
                        dict(claim="Devanagari script", pattern=r"devanagari"),
                        dict(claim="Article 343", pattern=r"343")]
    ),

    # =========================================================================
    # 7. CROSS-ARTICLE & COMPARATIVE (Contrasting dual provisions across corpus)
    # =========================================================================
    dict(
        id="X1", category="Cross-article",
        query="How does the pardoning power of the President compare with that of a State Governor regarding death sentences?",
        expected_answer="Under Article 72(1)(c), the President has exclusive power to pardon, commute, or remit sentences of death in all cases. Under Article 161, a Governor has power to grant pardons for offences against State laws, but cannot pardon death sentences except to suspend, remit, or commute where provided by law.",
        gold=r"in all cases where the sentence is a sentence of death",
        gold_sufficiency="any", source_citation="Article 72 vs Article 161, p. 64, 100", is_unanswerable=False,
        required_facts=[dict(claim="President handles death sentences", pattern=r"president"),
                        dict(claim="Governor power under state law", pattern=r"governor"),
                        dict(claim="Article 72 or 161", pattern=r"72|161")]
    ),
    dict(
        id="X2", category="Cross-article",
        query="What happens when a law made by a State Legislature is repugnant to a law made by Parliament on a Concurrent List subject?",
        expected_answer="Under Article 254(1), the law made by Parliament shall prevail and the law made by the Legislature of the State shall, to the extent of the repugnancy, be void (subject to Article 254(2) where Presidential assent was obtained).",
        gold=r"law made by Parliament.*shall prevail and the law made by the Legislature of the State shall, to the extent of the repugnancy, be void",
        gold_sufficiency="any", source_citation="Article 254, p. 179", is_unanswerable=False,
        required_facts=[dict(claim="Parliament law prevails", pattern=r"parliament.*prevail|prevail"),
                        dict(claim="State law void to extent of repugnancy", pattern=r"void|repugnan"),
                        dict(claim="Article 254", pattern=r"254")]
    ),
    dict(
        id="X3", category="Cross-article",
        query="To whom does the Speaker of the House of the People address a resignation letter, and to whom does the Deputy Speaker address theirs?",
        expected_answer="Under Article 94(b), the Speaker addresses their resignation under hand to the Deputy Speaker, and the Deputy Speaker addresses their resignation to the Speaker.",
        gold=r"addressed, if such member is the Speaker, to the Deputy Speaker, and if such member is the Deputy Speaker, to the Speaker",
        gold_sufficiency="any", source_citation="Article 94(b), p. 73", is_unanswerable=False,
        required_facts=[dict(claim="Speaker to Deputy Speaker", pattern=r"deputy speaker"),
                        dict(claim="Deputy Speaker to Speaker", pattern=r"speaker"),
                        dict(claim="Article 94", pattern=r"94")]
    ),
    dict(
        id="X4", category="Cross-article",
        query="How do the qualifications for appointment as a High Court judge differ from those of a Supreme Court judge?",
        expected_answer="Under Article 124(3), a Supreme Court judge requires citizenship plus at least 5 years as High Court judge, or 10 years as High Court advocate, or being a distinguished jurist. Under Article 217(2), a High Court judge requires citizenship plus at least 10 years holding judicial office or 10 years as High Court advocate (no distinguished jurist category).",
        gold=r"at least five years a Judge of a High Court",
        gold_sufficiency="any", source_citation="Article 124(3) vs Article 217(2), p. 89, 131", is_unanswerable=False,
        required_facts=[dict(claim="5 years vs 10 years requirements", pattern=r"five years|5 years|ten years|10 years"),
                        dict(claim="Article 124 or 217", pattern=r"124|217")]
    ),
    dict(
        id="X5", category="Cross-article",
        query="How does the retirement age of a Supreme Court judge compare to that of a High Court judge?",
        expected_answer="Under Article 124(2), a Supreme Court judge retires at the age of sixty-five (65) years, whereas under Article 217(1), a High Court judge retires at the age of sixty-two (62) years.",
        gold=r"hold office until he attains the age of sixty-five years",
        gold_sufficiency="any", source_citation="Article 124(2) vs Article 217(1), p. 88, 131", is_unanswerable=False,
        required_facts=[dict(claim="Supreme Court 65 years", pattern=r"65|sixty-five"),
                        dict(claim="High Court 62 years", pattern=r"62|sixty-two"),
                        dict(claim="Article 124 and 217", pattern=r"124|217")]
    ),
    dict(
        id="X6", category="Cross-article",
        query="What is the constitutional provision for equal pay for equal work, and is it an enforceable fundamental right or a Directive Principle?",
        expected_answer="Under Article 39(d), equal pay for equal work for both men and women is a Directive Principle of State Policy (Part IV), which guides governance but is non-justiciable under Article 37.",
        gold=r"that there is equal pay for equal work for both men and women",
        gold_sufficiency="any", source_citation="Article 39(d), p. 52", is_unanswerable=False,
        required_facts=[dict(claim="Equal pay for equal work", pattern=r"equal pay"),
                        dict(claim="Directive Principle / Part IV", pattern=r"directive principle|policy|part iv"),
                        dict(claim="Article 39", pattern=r"39")]
    ),
    dict(
        id="X7", category="Cross-article",
        query="How does the procedure for introducing a Money Bill in a State Legislature compare to Parliament?",
        expected_answer="Under Articles 109 and 198, in both Parliament and State Legislatures, a Money Bill cannot be introduced in the upper house (Council of States / Legislative Council). In both, the upper house has 14 days to make recommendations.",
        gold=r"Special procedure in respect of Money Bills",
        gold_sufficiency="any", source_citation="Article 109 vs Article 198, p. 80, 117", is_unanswerable=False,
        required_facts=[dict(claim="Cannot be introduced in upper house", pattern=r"not.*introduced|council"),
                        dict(claim="14 days recommendation window", pattern=r"fourteen|14"),
                        dict(claim="Article 109 or 198", pattern=r"109|198")]
    ),

    # =========================================================================
    # 8. UNANSWERABLE & HARD DISTRACTORS (Testing model abstention & hallucination)
    # =========================================================================
    dict(
        id="U1", category="Unanswerable",
        query="What is the mandatory retirement age for the President of India?",
        expected_answer="The provided context does not contain the answer. (The Constitution prescribes no maximum retirement age for the President).",
        gold="", gold_sufficiency="any", source_citation="N/A (unanswerable from Constitution)", is_unanswerable=True,
        required_facts=[dict(claim="Abstention", pattern=r"does not contain the answer")]
    ),
    dict(
        id="U2", category="Unanswerable",
        query="What is the standard rate of Goods and Services Tax (GST) applicable to legal software services in India?",
        expected_answer="The provided context does not contain the answer. (GST rates are prescribed by statutory council notifications, not in the Constitution text).",
        gold="", gold_sufficiency="any", source_citation="N/A (unanswerable from Constitution)", is_unanswerable=True,
        required_facts=[dict(claim="Abstention", pattern=r"does not contain the answer")]
    ),
    dict(
        id="U3", category="Unanswerable",
        query="How many judges serve on the United States Supreme Court according to the Constitution?",
        expected_answer="The provided context does not contain the answer. (The Constitution of India does not govern foreign courts).",
        gold="", gold_sufficiency="any", source_citation="N/A (unanswerable from Constitution)", is_unanswerable=True,
        required_facts=[dict(claim="Abstention", pattern=r"does not contain the answer")]
    ),
    dict(
        id="U4", category="Unanswerable",
        query="What is the maximum legal speed limit on national expressways according to the Constitution of India?",
        expected_answer="The provided context does not contain the answer. (Speed limits are governed by the Motor Vehicles Act, not the Constitution).",
        gold="", gold_sufficiency="any", source_citation="N/A (unanswerable from Constitution)", is_unanswerable=True,
        required_facts=[dict(claim="Abstention", pattern=r"does not contain the answer")]
    ),
    dict(
        id="U5", category="Unanswerable",
        query="What is the minimum educational degree required to become the Prime Minister of India?",
        expected_answer="The provided context does not contain the answer. (The Constitution prescribes no educational qualifications for the Prime Minister or MPs).",
        gold="", gold_sufficiency="any", source_citation="N/A (unanswerable from Constitution)", is_unanswerable=True,
        required_facts=[dict(claim="Abstention", pattern=r"does not contain the answer")]
    ),
    dict(
        id="U6", category="Unanswerable",
        query="What is the statutory monetary fine for copyright infringement of digital media under the Constitution of India?",
        expected_answer="The provided context does not contain the answer. (Copyright penalties are statutory under the Copyright Act, not in the Constitution).",
        gold="", gold_sufficiency="any", source_citation="N/A (unanswerable from Constitution)", is_unanswerable=True,
        required_facts=[dict(claim="Abstention", pattern=r"does not contain the answer")]
    ),
    dict(
        id="U7", category="Unanswerable",
        query="How many non-permanent members serve on the United Nations Security Council according to the Constitution?",
        expected_answer="The provided context does not contain the answer. (UN Security Council composition is determined by the UN Charter, not the Constitution of India).",
        gold="", gold_sufficiency="any", source_citation="N/A (unanswerable from Constitution)", is_unanswerable=True,
        required_facts=[dict(claim="Abstention", pattern=r"does not contain the answer")]
    ),
]


# %% SECTION 3: PDF INGESTION, LAYOUT PRESERVATION & FOOTNOTE SANITIZATION
# =================================================================================================
# SECTION 3: PDF INGESTION, LAYOUT PRESERVATION & FOOTNOTE SANITIZATION
# =================================================================================================
# Regular expressions designed to detect layout structures in the Indian Constitution PDF:

# 3.1 Footnote Start Pattern:
# Official PDF pages contain dense legislative footnotes (e.g., '1. Subs. by the Constitution (Forty-fourth
# Amendment) Act, 1978, s. 2...'). Ingesting footnotes introduces catastrophic noise into embeddings.
# This regex matches standard footnote prefixes at the bottom of the page layout.
FOOTNOTE_START = re.compile(
    r"^\s{0,1}\d{1,3}\s*\.?\s*(Ins\.|Subs\.|Rep\.|Sub-|Omitted|Added|The words|The Constitution|Cl\.|Art\.|"
    r"Para|Entry|Entries|Sections?\b|S\.|See|Renumbered|Re-numbered|Clause|Substituted|Inserted|Repealed|"
    r"Expression|Existing|These|Provisions?|Words|Item|Sub-clause|Certain|Now|Vide|This|For |Part|Chapter|"
    r"The (bracket|brackets|figure|word|letter|entry|entries|proviso|marginal|Explanation|sub|portion|full|article|clause|section))"
)

# 3.2 Running Header Pattern:
# Matches running page headers (e.g., 'THE CONSTITUTION OF INDIA') appearing at top of pages.
HEADER = re.compile(r"^\s*\d*\s*THE CONSTITUTION OF INDIA\s*\d*\s*$")

# 3.3 Paragraph Indentation Pattern:
# Detects legal paragraph beginnings, sub-clause enumerations '(1)', '(a)', 'Provided that', etc.
PARA_START = re.compile(r"^\s{3,}(\[?\(?\d+[A-Z]*\)|\[?\d+[A-Z]*\.\s+\S|\(\w{1,4}\)\s|Provided|Explanation|\*)")

# 3.4 Article Header Pattern:
# Matches article heading titles (e.g., '21. Protection of life and personal liberty.—No person shall...')
ARTICLE_HEAD = re.compile(r"^\s*\[?(\d{1,3}[A-Z]{0,2})\.\s+[A-Z].{0,160}?\.\s*[—⎯\-]")


def extract_pages(pdf_path):
    """
    Extracts raw text page by page from the official Constitution of India PDF.

    Prioritizes the external CLI tool `pdftotext -layout` because it accurately preserves
    column indentation, statutory sub-clauses, and margin whitespace required for paragraph
    boundary detection. Falls back gracefully to PyMuPDF (fitz) if pdftotext is not installed.

    Args:
        pdf_path (str or Path): Path to the source PDF file.

    Returns:
        list[str]: Array of extracted page text strings (one per page).
    """
    # Prefer pdftotext with layout preservation
    if shutil.which("pdftotext"):
        command = ["pdftotext", "-layout", str(pdf_path), "-"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Form feed character () separates pages in pdftotext stdout
        return result.stdout.split("\f")

    # Fallback to PyMuPDF if system pdftotext utility is absent
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    return [page.get_text() for page in doc]


def clean_page(text):
    """
    Cleans raw page text by stripping running headers, amendment numbers, and bottom footnotes.

    Safety Guard:
      Footnotes are located strictly at the bottom of the page in the official publication layout.
      To prevent accidental truncation of valid legal text that happens to match footnote keywords,
      the footnote stripping rule is applied only in the lower 55% of each page (line index >= 0.45 * total).

    Args:
        text (str): Raw extracted single-page text string.

    Returns:
        str: Sanitized page text with footnotes and header artifacts removed.
    """
    raw_lines = text.split("\n")
    # Strip running headers matching 'THE CONSTITUTION OF INDIA'
    filtered_lines = [line for line in raw_lines if not HEADER.match(line)]
    total_lines = len(filtered_lines)

    # Scan bottom 55% of the page for footnote section start
    for idx, line in enumerate(filtered_lines):
        if idx >= int(total_lines * 0.45) and FOOTNOTE_START.match(line):
            # Truncate all subsequent lines since footnotes continue to page bottom
            filtered_lines = filtered_lines[:idx]
            break

    cleaned_text = "\n".join(filtered_lines)
    # Strip inline footnote markers such as '1***' or '2[' indicating historical amendments
    cleaned_text = re.sub(r"(?:(?<=\s)|^)\d{1,3}\*{3}", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"(?:(?<=\s)|^)\d{1,3}\[", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = cleaned_text.replace("]", "").replace("\u2003", " ")
    return cleaned_text


def split_units(para):
    """
    Recursively segments oversized statutory paragraphs into clean semantic units under CHUNK_CHARS.

    Splits progressively along punctuation clause boundaries (semicolons, periods, colons),
    falling back to word boundaries if an individual clause exceeds the character threshold.

    Args:
        para (str): A single extracted legal paragraph text.

    Returns:
        list[str]: Segmented text units, each constrained to at most CHUNK_CHARS length.
    """
    if len(para) <= CHUNK_CHARS:
        return [para]

    segmented_units, current_accumulator = [], ""
    # Split paragraph on sentence and clause boundaries (semicolons are heavily used in legal drafting)
    for clause in re.split(r"(?<=[.;:])\s+", para):
        # If an individual clause is larger than CHUNK_CHARS, hard-split on word boundaries
        while len(clause) > CHUNK_CHARS:
            space_cut = clause.rfind(" ", 0, CHUNK_CHARS)
            cut_idx = space_cut if space_cut > 0 else CHUNK_CHARS
            segmented_units.append(clause[:cut_idx])
            clause = clause[cut_idx:].strip()

        # Check if adding this clause exceeds the target chunk size
        if len(current_accumulator) + len(clause) + 1 > CHUNK_CHARS and current_accumulator:
            segmented_units.append(current_accumulator)
            current_accumulator = clause
        else:
            current_accumulator = f"{current_accumulator} {clause}".strip()

    if current_accumulator:
        segmented_units.append(current_accumulator)

    return segmented_units


# %% SECTION 4: SEMANTIC UNIT CHUNKING & REPRODUCIBILITY MANIFEST
# =================================================================================================
# SECTION 4: SEMANTIC UNIT CHUNKING & REPRODUCIBILITY MANIFEST
# =================================================================================================
def build_chunks(pdf_path):
    """
    Parses the Constitution PDF into clean, metadata-enriched semantic text chunks.

    Pipeline Overview:
      1. Extract and clean pages between FIRST_PAGE (32) and LAST_PAGE (381).
      2. Identify structural paragraph boundaries and track the active constitutional Article number.
      3. Assemble paragraphs into chunks of ~CHUNK_CHARS (900) characters with CHUNK_OVERLAP (150) overlap.
      4. Discard malformed fragments or numeric tables (filtering on length and alphabetic character ratio).

    Args:
        pdf_path (str or Path): Path to the Constitution PDF file.

    Returns:
        list[dict]: Array of verified semantic chunks with chunk_id, text, article, and page metadata.
    """
    pages = extract_pages(pdf_path)
    has_layout = shutil.which("pdftotext") is not None
    semantic_units = []
    active_article = None
    current_paragraph_lines = []
    paragraph_start_page = None

    def flush_paragraph():
        """Flushes accumulated lines of a statutory paragraph into segmented semantic units."""
        nonlocal current_paragraph_lines, paragraph_start_page, active_article
        if not current_paragraph_lines:
            return
        combined_text = re.sub(r"\s+", " ", " ".join(current_paragraph_lines)).strip()
        current_paragraph_lines = []
        if not combined_text:
            return

        # Check if paragraph begins with an Article title (e.g., '19. Protection of certain rights...')
        match_article = ARTICLE_HEAD.match(combined_text)
        if match_article:
            active_article = match_article.group(1)

        for unit in split_units(combined_text):
            semantic_units.append((unit, paragraph_start_page, active_article))

    # Iterate strictly over operative constitutional pages (pages 32 to 381)
    for page_zero_idx in range(FIRST_PAGE - 1, LAST_PAGE):
        page_num = page_zero_idx + 1
        page_text = clean_page(pages[page_zero_idx])

        for line in page_text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Detect beginning of a new constitutional clause or paragraph
            is_para_boundary = (PARA_START.match(line) if has_layout
                                else re.match(r"^\s*(\(\w{1,4}\)|\d+[A-Z]*\.)\s", line))

            if is_para_boundary:
                flush_paragraph()
                paragraph_start_page = page_num
            elif paragraph_start_page is None:
                paragraph_start_page = page_num

            current_paragraph_lines.append(line_stripped)

    # Flush any remaining paragraph content at end of pages
    flush_paragraph()

    # Step 2: Assemble semantic units into sliding-window overlapping chunks
    assembled_chunks = []
    current_chunk_units = []
    overlap_tail = ""

    def emit_chunk():
        """Emits a completed chunk and computes sliding-window overlap tail for continuity."""
        nonlocal current_chunk_units, overlap_tail
        if not current_chunk_units:
            return
        body_text = " ".join(u[0] for u in current_chunk_units)
        full_text = f"{overlap_tail} {body_text}".strip() if overlap_tail else body_text

        # Collect all constitutional articles referenced within this chunk
        referenced_articles = sorted(
            {u[2] for u in current_chunk_units if u[2]},
            key=lambda art_str: (len(art_str), art_str)
        )

        chunk_record = dict(
            chunk_id=len(assembled_chunks),
            text=full_text,
            page_start=current_chunk_units[0][1],
            page_end=current_chunk_units[-1][1],
            article=current_chunk_units[0][2],
            articles=referenced_articles,
            n_chars=len(full_text)
        )
        assembled_chunks.append(chunk_record)

        # Compute overlap tail from end of body to prepend to the next chunk
        trailing_text = body_text[-CHUNK_OVERLAP:]
        space_idx = trailing_text.find(" ")
        overlap_tail = trailing_text[space_idx + 1:] if space_idx >= 0 and len(body_text) > CHUNK_OVERLAP else ""
        current_chunk_units = []

    for unit_entry in semantic_units:
        # Check if adding the new unit exceeds CHUNK_CHARS
        projected_size = sum(len(x[0]) + 1 for x in current_chunk_units) + len(unit_entry[0])
        if current_chunk_units and projected_size > CHUNK_CHARS:
            emit_chunk()
        current_chunk_units.append(unit_entry)

    emit_chunk()

    # Step 3: Filter out numerical table artifacts or tiny fragments (< 80 chars or low alphabetic content)
    valid_chunks = [
        c for c in assembled_chunks
        if c["n_chars"] >= 80 and (sum(ch.isalpha() for ch in c["text"]) / c["n_chars"] > 0.60)
    ]

    # Re-index chunk IDs consecutively
    for new_id, chunk_item in enumerate(valid_chunks):
        chunk_item["chunk_id"] = new_id

    return valid_chunks


def verify_chunks(chunks):
    """
    Automated sanity assertions verifying chunk volume, character distributions, and Article metadata.

    Guarantees:
      - Total chunks >= 500 (standard parsing yields ~880 chunks).
      - No tiny corrupted chunks (< 70 characters).
      - Broad Article metadata coverage across the statutory corpus.

    Args:
        chunks (list[dict]): Extracted chunks array.

    Returns:
        bool: True if all sanity assertions pass.
    """
    char_lengths = [c["n_chars"] for c in chunks]
    article_tagged_count = sum(bool(c["article"]) for c in chunks)
    mean_length = float(np.mean(char_lengths))

    print(f"[chunk_sanity] Total chunks: {len(chunks)}, mean chars: {mean_length:.0f}, "
          f"min: {min(char_lengths)}, max: {max(char_lengths)}, "
          f"chunks carrying Article metadata: {article_tagged_count}/{len(chunks)}")

    assert len(chunks) >= 500, f"Suspiciously few chunks: {len(chunks)}"
    assert min(char_lengths) >= 70, f"Found chunk with suspiciously few chars: {min(char_lengths)}"
    return True


def compute_manifest(pdf_path, chunks, bi_name, ce_name):
    """
    Calculates a cryptographic SHA-256 fingerprint of the raw PDF and chunk corpus.

    Ensures that any change in source document, page boundaries, chunking hyperparameters,
    or embedding model names triggers an automatic cache invalidation.

    Args:
        pdf_path (str or Path): Path to the source PDF.
        chunks (list[dict]): Assembled chunks array.
        bi_name (str): Identifier of the Bi-Encoder model.
        ce_name (str): Identifier of the Cross-Encoder model.

    Returns:
        dict: Cryptographic manifest with SHA-256 hashes and configuration metadata.
    """
    raw_pdf_bytes = Path(pdf_path).read_bytes()
    pdf_sha256 = hashlib.sha256(raw_pdf_bytes).hexdigest()

    concatenated_chunk_text = "".join(c["text"] for c in chunks)
    chunks_sha256 = hashlib.sha256(concatenated_chunk_text.encode("utf-8")).hexdigest()

    return dict(
        pdf_path=str(pdf_path),
        pdf_sha256=pdf_sha256,
        first_page=FIRST_PAGE,
        last_page=LAST_PAGE,
        chunk_chars=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        n_chunks=len(chunks),
        chunks_sha256=chunks_sha256,
        bi_encoder=bi_name,
        cross_encoder=ce_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )


def is_manifest_valid(manifest_path, current_manifest):
    """
    Verifies existing on-disk cache manifest against the current execution environment.

    Args:
        manifest_path (Path): Path to the cached cache_manifest.json file.
        current_manifest (dict): Newly computed manifest for current environment.

    Returns:
        bool: True if all 9 cryptographic and algorithmic parameters match exactly.
    """
    if not manifest_path.exists():
        return False
    try:
        saved_manifest = json.loads(manifest_path.read_text())
        invariant_keys = [
            "pdf_sha256", "first_page", "last_page", "chunk_chars", "chunk_overlap",
            "n_chunks", "chunks_sha256", "bi_encoder", "cross_encoder"
        ]
        return all(saved_manifest.get(k) == current_manifest.get(k) for k in invariant_keys)
    except Exception:
        return False


# %% SECTION 5: DUAL-STAGE RETRIEVER ARCHITECTURE (FAISS IP + CROSS-ENCODER)
# =================================================================================================
# SECTION 5: DUAL-STAGE RETRIEVER ARCHITECTURE (FAISS IP + CROSS-ENCODER)
# =================================================================================================
class Retriever:
    """
    Two-Stage Hybrid Neural Retrieval Engine:
      - Stage 1: Fast Dense Bi-Encoder + FAISS Exact Inner-Product Index (sub-millisecond retrieval).
      - Stage 2: High-Precision Cross-Encoder Re-Ranking via joint token-level cross-attention.

    Mathematical Representation:
      Bi-Encoder    : s(q, d) = u_q^T v_d  (dot product over L2-normalized 384-d vectors == cosine)
      Cross-Encoder : s(q, d) = W * Transformer([CLS] o q o [SEP] o d o [SEP])
    """

    def __init__(self, chunks, out_dir):
        """
        Initializes the retriever, loads transformer checkpoints, and constructs the FAISS index.

        Args:
            chunks (list[dict]): Preprocessed corpus chunk dictionaries.
            out_dir (str or Path): Output directory for caching embeddings and index data.
        """
        import faiss
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.faiss = faiss
        self.chunks = chunks
        self.out_dir = Path(out_dir)

        # Initialize neural encoder checkpoints
        self.bi = SentenceTransformer(BI_ENCODER)
        self.ce = CrossEncoder(CROSS_ENCODER)

        manifest_file = self.out_dir / "cache_manifest.json"
        current_manifest = compute_manifest(PDF_PATH, chunks, BI_ENCODER, CROSS_ENCODER)
        embeddings_cache_file = self.out_dir / "embeddings.npy"

        # Check if precomputed embedding cache is valid
        if embeddings_cache_file.exists() and is_manifest_valid(manifest_file, current_manifest):
            print("[cache] Valid cache manifest found. Loading precomputed dense embeddings from disk.")
            self.emb = np.load(embeddings_cache_file)
        else:
            print("[cache] Invalidation or missing cache. Computing fresh bi-encoder embeddings offline...")
            # Compute dense embeddings in batches; normalize to unit length for inner-product equivalence
            self.emb = self.bi.encode(
                [c["text"] for c in chunks],
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True
            ).astype("float32")
            # Persist embeddings and cryptographic manifest to disk
            np.save(embeddings_cache_file, self.emb)
            manifest_file.write_text(json.dumps(current_manifest, indent=2))

        # Construct FAISS flat inner-product index (O(N) exact dot-product scan)
        embedding_dim = self.emb.shape[1]  # 384 dimensions
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.emb)

    def embed_query(self, query_str):
        """
        Encodes an incoming natural language query into a unit-normalized 384-dimensional vector.

        Args:
            query_str (str): Input query text.

        Returns:
            np.ndarray: 2D float32 array of shape (1, 384).
        """
        return self.bi.encode([query_str], normalize_embeddings=True).astype("float32")

    def search(self, query_vector, k):
        """
        Performs exact inner product similarity search over the indexed corpus.

        Because vectors are L2-normalized, inner product equals cosine similarity:
            s(q, d) = u_q^T v_d = ||u_q|| * ||v_d|| * cos(theta) = cos(theta)

        Args:
            query_vector (np.ndarray): 384-d normalized float32 query vector.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            tuple[list[int], list[float]]: (retrieved_chunk_ids, cosine_similarity_scores)
        """
        effective_k = min(k, len(self.chunks))
        scores, indices = self.index.search(query_vector, effective_k)
        return indices[0].tolist(), scores[0].tolist()

    def rerank(self, query_str, candidate_ids):
        """
        Executes second-stage re-ranking by scoring (query, passage) pairs with the Cross-Encoder.

        Passes each pair jointly through all 6 transformer cross-attention layers, capturing
        rich query-document token interactions that bi-encoders inherently miss.

        Args:
            query_str (str): The natural language query.
            candidate_ids (list[int]): Chunk IDs retrieved by Stage 1.

        Returns:
            list[float]: Unconstrained scalar relevance logits for each candidate chunk.
        """
        if not candidate_ids:
            return []
        input_pairs = [(query_str, self.chunks[cid]["text"]) for cid in candidate_ids]
        relevance_logits = self.ce.predict(input_pairs, batch_size=len(candidate_ids), show_progress_bar=False)
        return [float(score) for score in relevance_logits]


# %% SECTION 6: LOCAL LLM CLIENT & INFERENCE CACHE (OLLAMA)
# =================================================================================================
# SECTION 6: LOCAL LLM CLIENT & INFERENCE CACHE (OLLAMA)
# =================================================================================================
class LLM:
    """
    Wrapper for local LLM inference via Ollama with deterministic parameters and on-disk caching.

    Key Features:
      - Deterministic execution: Temperature = 0, fixed random seed (42).
      - SHA-256 prompt hashing cache: Avoids duplicate inference during repeated benchmark runs.
      - Latency & token tracking: Records wall-clock generation seconds and token counts.
    """

    def __init__(self, model_name, cache_file_path):
        """
        Initializes the LLM interface and loads persistent generation cache.

        Args:
            model_name (str): Ollama model tag (e.g., 'llama3.2:3b').
            cache_file_path (str or Path): Path to local JSON response cache.
        """
        self.model = model_name
        self.cache_path = Path(cache_file_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.client = None

    def _client(self):
        """Lazily connects to the local Ollama daemon and validates model availability."""
        if self.client is None:
            import ollama
            self.client = ollama.Client()
            available_models = [m.model for m in self.client.list().models]
            # Ensure requested model exists locally
            is_present = any(
                n == self.model or n.startswith(self.model + ":") or self.model.startswith(n)
                for n in available_models
            )
            if not is_present:
                sys.exit(f"Ollama model '{self.model}' not found. Run: ollama pull {self.model} (have: {available_models})")
        return self.client

    def chat(self, system_prompt, user_prompt, max_tokens=300):
        """
        Sends chat completion request to the local LLM with prompt caching.

        Args:
            system_prompt (str): Strict instructions to ground responses in context.
            user_prompt (str): Labeled context passages and question.
            max_tokens (int): Maximum output tokens budget.

        Returns:
            dict: Structured response containing 'text', 'seconds', 'prompt_tokens',
                  'output_tokens', and 'cached' boolean flag.
        """
        # Unique cache key covering model identity, prompts, and deterministic seed
        cache_key = hashlib.sha256(
            json.dumps([self.model, system_prompt, user_prompt, max_tokens, SEED]).encode()
        ).hexdigest()

        if cache_key in self.cache:
            return {**self.cache[cache_key], "cached": True}

        start_time = time.perf_counter()
        # Budget context window based on model architecture
        is_reasoning_model = any(tag in self.model.lower() for tag in ("deepseek", "qwq"))
        token_budget = max_tokens * 8 if is_reasoning_model else max_tokens
        context_window = 6144 if is_reasoning_model else 4096

        response = self._client().chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options=dict(
                temperature=0,
                seed=SEED,
                num_ctx=context_window,
                num_predict=token_budget
            )
        )

        raw_output = response["message"]["content"]
        # Strip chain-of-thought tags if running a hybrid reasoning checkpoint
        cleaned_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
        cleaned_output = cleaned_output.split("</think>")[-1].strip()

        elapsed_seconds = time.perf_counter() - start_time
        result_payload = dict(
            text=cleaned_output,
            seconds=elapsed_seconds,
            prompt_tokens=response.get("prompt_eval_count"),
            output_tokens=response.get("eval_count")
        )

        # Store to in-memory dictionary and persist to disk
        self.cache[cache_key] = result_payload
        self.cache_path.write_text(json.dumps(self.cache))
        return {**result_payload, "cached": False}


# %% SECTION 7: MULTI-FACETED LEGAL ANSWER EVALUATION RUBRIC
# =================================================================================================
# SECTION 7: MULTI-FACETED LEGAL ANSWER EVALUATION RUBRIC
# =================================================================================================
def evaluate_answer(answer_text, spec, context_chunks):
    """
    Evaluates generated legal answers across five orthogonal evaluation criteria:

      1. Abstention Score [0.0 or 1.0]:
         - Rewards appropriate refusal ('The provided context does not contain the answer')
           when query is unanswerable or gold evidence was absent from retrieved context.
         - Penalizes false refusal when gold evidence was present in the prompt context.

      2. Completeness Score [0.0 - 1.0]:
         - Measures the fraction of required atomic facts identified via regex match.

      3. Citation Score [0.0 or 1.0]:
         - Verifies whether the expected constitutional Article number is explicitly cited.

      4. Grounding Score [0.0 - 1.0]:
         - Hallucination detector: Confirms that Article numbers cited in the answer
           actually appear in the supplied context passages.

      5. Correctness Score [0.0 - 1.0]:
         - Composite substantive legal score:
           Correctness = 0.6 * Completeness + 0.2 * Citation + 0.2 * Grounding

    Args:
        answer_text (str): The raw answer generated by the LLM.
        spec (dict): Benchmark query specification containing expected answers and required facts.
        context_chunks (list[dict]): The Top-n context passages provided to the LLM.

    Returns:
        dict: Scores for correctness, completeness, citation, grounding, and abstention.
    """
    text_clean = answer_text.strip()
    text_lower = text_clean.lower()
    is_unanswerable_query = spec.get("is_unanswerable", False)

    # Detect if LLM issued the required abstention response
    did_abstain = (
        "does not contain the answer" in text_lower or
        "provided context does not contain" in text_lower
    )

    # ---------------------------------------------------------------------------------------------
    # 7.1 Abstention Evaluation
    # ---------------------------------------------------------------------------------------------
    if is_unanswerable_query:
        # For unanswerable queries, abstention is the only acceptable correct behavior
        abstention_score = 1.0 if did_abstain else 0.0
    else:
        # Check if the retrieved context actually contained any gold evidence chunk
        has_gold_chunk_in_context = any(
            c["chunk_id"] in set(spec.get("gold_ids", [])) for c in context_chunks
        )
        if not has_gold_chunk_in_context:
            # Gold chunk was missing: abstaining is legally sound and prevents hallucination
            abstention_score = 1.0 if did_abstain else 0.5
        else:
            # Gold chunk was present: abstaining represents a false refusal
            abstention_score = 0.0 if did_abstain else 1.0

    # ---------------------------------------------------------------------------------------------
    # 7.2 Completeness Evaluation (Atomic Fact Recall)
    # ---------------------------------------------------------------------------------------------
    required_facts = spec.get("required_facts", [])
    if is_unanswerable_query:
        completeness = 1.0 if did_abstain else 0.0
    elif did_abstain:
        completeness = 0.0
    elif required_facts:
        fact_hits = sum(bool(re.search(f["pattern"], text_clean, re.IGNORECASE)) for f in required_facts)
        completeness = round(fact_hits / len(required_facts), 3)
    else:
        completeness = 1.0

    # ---------------------------------------------------------------------------------------------
    # 7.3 Statutory Citation Evaluation
    # ---------------------------------------------------------------------------------------------
    expected_citation = spec.get("source_citation", "")
    match_article = re.search(r"article\s+(\d+[a-z]*)", expected_citation, re.IGNORECASE)
    if is_unanswerable_query or not match_article:
        citation_score = 1.0 if did_abstain or not match_article else 0.0
    else:
        expected_art_num = match_article.group(1).lower()
        has_cited_article = bool(re.search(rf"\barticle\s+{expected_art_num}\b", text_lower))
        citation_score = 1.0 if has_cited_article else 0.0

    # ---------------------------------------------------------------------------------------------
    # 7.4 Context Grounding & Hallucination Suppression
    # ---------------------------------------------------------------------------------------------
    if is_unanswerable_query and did_abstain:
        grounding_score = 1.0
    elif did_abstain:
        grounding_score = 1.0
    else:
        combined_context_text = " ".join(c["text"].lower() for c in context_chunks)
        cited_articles_in_answer = re.findall(r"\barticle\s+(\d+[a-z]*)", text_lower)
        if cited_articles_in_answer:
            # Verify each cited article appears in the context
            grounded_citations = [art for art in cited_articles_in_answer if art in combined_context_text]
            grounding_score = round(len(grounded_citations) / len(cited_articles_in_answer), 2)
        else:
            grounding_score = 0.8  # No ungrounded articles cited

    # ---------------------------------------------------------------------------------------------
    # 7.5 Composite Correctness Metric
    # ---------------------------------------------------------------------------------------------
    if is_unanswerable_query:
        composite_correctness = 1.0 if did_abstain else 0.0
    elif did_abstain:
        composite_correctness = 0.0
    else:
        composite_correctness = round(
            0.60 * completeness + 0.20 * citation_score + 0.20 * grounding_score, 3
        )

    return dict(
        correctness=composite_correctness,
        completeness=completeness,
        citation=citation_score,
        grounding=grounding_score,
        abstention=abstention_score
    )


# %% SECTION 8: RANKING METRICS, RANK SHIFT DYNAMICS & STAGE EXECUTION
# =================================================================================================
# SECTION 8: RANKING METRICS, RANK SHIFT DYNAMICS & STAGE EXECUTION
# =================================================================================================
def norm(text_str):
    """Normalizes arbitrary whitespace sequences into single spaces."""
    return re.sub(r"\s+", " ", text_str)


def calc_ndcg(retrieved_ids, gold_set, k=3):
    """
    Computes Normalized Discounted Cumulative Gain at rank k (nDCG@k).

    Formula:
        DCG@k = sum_{i=1}^k [ rel(i) / log2(i + 1) ]
        IDCG@k = sum_{i=1}^{min(k, |Gold|)} [ 1 / log2(i + 1) ]
        nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids (list[int]): Ordered sequence of retrieved chunk IDs.
        gold_set (set[int]): Set of ground-truth relevant chunk IDs.
        k (int): Evaluation cutoff rank (default: 3).

    Returns:
        float: Normalized gain in [0.0, 1.0].
    """
    discounted_gain = 0.0
    for rank_idx, chunk_id in enumerate(retrieved_ids[:k], 1):
        if chunk_id in gold_set:
            discounted_gain += 1.0 / np.log2(rank_idx + 1)

    ideal_gain = sum(1.0 / np.log2(idx + 1) for idx in range(1, min(k, len(gold_set)) + 1))
    return float(discounted_gain / ideal_gain) if ideal_gain > 0.0 else 0.0


def run_query(spec, R, llm, top_k=10, top_n=3):
    """
    Executes a complete evaluation trial for a single query across both pipelines.

    Execution Flow:
      1. Stage 1 Retrieval: Bi-Encoder embeds query -> FAISS IndexFlatIP searches Top-k candidates.
      2. Stage 2 Re-Ranking: Cross-Encoder joint-scores candidate pairs -> re-orders descending.
      3. Rank Shift Calculation: Logs rank displacement (shift = base_rank - ce_rank).
      4. Outcome Classification: Assigns status (FIXED_BY_RERANK, DEGRADED_BY_RERANK, BOTH_OK, etc.).
      5. LLM Answer Generation (optional): Generates Answer 1 (Baseline) and Answer 2 (Re-Ranked).
      6. Evaluation Rubric: Evaluates completeness, citation, grounding, and correctness.

    Args:
        spec (dict): Benchmark query specification.
        R (Retriever): Active two-stage retriever instance.
        llm (LLM or None): Local LLM instance (None if skipping LLM generation).
        top_k (int): Candidate pool size retrieved by Stage 1 (default: 10).
        top_n (int): Number of context chunks passed to LLM (default: 3).

    Returns:
        dict: Complete per-query result record containing retrieval metrics and generated answers.
    """
    query_text = spec["query"]
    is_unans = spec.get("is_unanswerable", False)
    gold_chunk_ids = set(spec.get("gold_ids", []))

    # ---------------------------------------------------------------------------------------------
    # Step 1: Baseline Bi-Encoder Search (Stage 1)
    # ---------------------------------------------------------------------------------------------
    query_vector = R.embed_query(query_text)
    base_candidate_ids, base_similarity_scores = R.search(query_vector, top_k)
    effective_candidate_count = len(base_candidate_ids)

    # ---------------------------------------------------------------------------------------------
    # Step 2: Cross-Encoder Re-Ranking (Stage 2)
    # ---------------------------------------------------------------------------------------------
    cross_encoder_logits = R.rerank(query_text, base_candidate_ids)
    # Sort candidates descending by Cross-Encoder relevance score
    rerank_permutation = sorted(range(effective_candidate_count), key=lambda idx: -cross_encoder_logits[idx])
    ce_rank_mapping = {base_candidate_ids[idx]: rank_pos + 1 for rank_pos, idx in enumerate(rerank_permutation)}
    reranked_candidate_ids = [base_candidate_ids[idx] for idx in rerank_permutation]

    # Build detailed rank-shift audit trail for top-10 candidates
    candidate_shift_records = []
    for base_rank, (cid, b_score, ce_score) in enumerate(
        zip(base_candidate_ids, base_similarity_scores, cross_encoder_logits), 1
    ):
        chunk_obj = R.chunks[cid]
        new_rank = ce_rank_mapping[cid]
        candidate_shift_records.append(dict(
            query_id=spec["id"],
            chunk_id=cid,
            base_rank=base_rank,
            base_score=round(float(b_score), 4),
            ce_score=round(float(ce_score), 4),
            ce_rank=new_rank,
            rank_shift=base_rank - new_rank,  # Positive shift denotes upward promotion
            is_gold=cid in gold_chunk_ids,
            page=chunk_obj["page_start"],
            article=chunk_obj["article"],
            preview=chunk_obj["text"][:160]
        ))

    # Helper function to find best rank among gold chunks
    def get_best_gold_rank(rank_key):
        ranks = [rec[rank_key] for rec in candidate_shift_records if rec["is_gold"]]
        return min(ranks) if ranks else None

    gold_rank_baseline = get_best_gold_rank("base_rank")
    gold_rank_reranked = get_best_gold_rank("ce_rank")
    candidate_recall_at_10 = float(bool(gold_chunk_ids & set(base_candidate_ids))) if not is_unans else None

    # ---------------------------------------------------------------------------------------------
    # Step 3: Classify Retrieval Outcome Status
    # ---------------------------------------------------------------------------------------------
    if is_unans:
        outcome_status = "UNANSWERABLE"
    elif gold_rank_baseline is None:
        outcome_status = "GOLD_NOT_IN_TOP10"
    elif gold_rank_baseline > top_n and gold_rank_reranked <= top_n:
        outcome_status = "FIXED_BY_RERANK"      # Rescued from ranks 4-10 into Top-3 LLM context
    elif gold_rank_baseline <= top_n and gold_rank_reranked > top_n:
        outcome_status = "DEGRADED_BY_RERANK"   # Regressed out of Top-3 LLM context
    elif gold_rank_baseline <= top_n:
        outcome_status = "BOTH_OK"              # Maintained in Top-3 under both pipelines
    else:
        outcome_status = "BOTH_FAIL"            # Present in candidate pool but missed Top-3 in both

    baseline_top_context = base_candidate_ids[:top_n]
    reranked_top_context = reranked_candidate_ids[:top_n]

    # Compute ranking quality metrics
    ndcg_base = calc_ndcg(base_candidate_ids, gold_chunk_ids, top_n) if not is_unans else None
    ndcg_rerank = calc_ndcg(reranked_candidate_ids, gold_chunk_ids, top_n) if not is_unans else None

    query_summary_record = dict(
        id=spec["id"],
        category=spec["category"],
        query=query_text,
        expected_answer=spec.get("expected_answer", ""),
        source_citation=spec.get("source_citation", ""),
        is_unanswerable=is_unans,
        n_gold_chunks=len(gold_chunk_ids),
        candidate_recall10=candidate_recall_at_10,
        gold_rank_base=gold_rank_baseline,
        gold_rank_rerank=gold_rank_reranked,
        ndcg3_base=ndcg_base,
        ndcg3_rerank=ndcg_rerank,
        status=outcome_status,
        base_top3=baseline_top_context,
        rerank_top3=reranked_top_context,
        same_prompt_chunks=set(baseline_top_context) == set(reranked_top_context),
        same_prompt_order=baseline_top_context == reranked_top_context,
        top10=candidate_shift_records,
        facts=spec.get("facts", [])
    )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Downstream Generation & Evaluation (Answer 1 vs Answer 2)
    # ---------------------------------------------------------------------------------------------
    if llm is not None:
        for pipeline_tag, context_ids in (("1", baseline_top_context), ("2", reranked_top_context)):
            context_chunk_objs = [R.chunks[cid] for cid in context_ids]
            prompt_str = build_user_prompt(query_text, context_chunk_objs)
            llm_response = llm.chat(SYSTEM_PROMPT, prompt_str)

            query_summary_record[f"answer{pipeline_tag}"] = llm_response["text"]
            query_summary_record[f"llm_seconds{pipeline_tag}"] = round(llm_response["seconds"], 2)
            query_summary_record[f"prompt_tokens{pipeline_tag}"] = llm_response["prompt_tokens"]
            query_summary_record[f"llm_cached{pipeline_tag}"] = llm_response["cached"]

            # Evaluate generated answer using legal evaluation rubric
            evaluation_results = evaluate_answer(llm_response["text"], spec, context_chunk_objs)
            for eval_metric_name, eval_metric_val in evaluation_results.items():
                query_summary_record[f"{eval_metric_name}{pipeline_tag}"] = eval_metric_val

    return query_summary_record


# %% SECTION 9: ATOMIC RESUMABLE BENCHMARK RUNNER
# =================================================================================================
# SECTION 9: ATOMIC RESUMABLE BENCHMARK RUNNER
# =================================================================================================
def run_benchmark(queries, R, llm, out_dir, top_k=10, top_n=3):
    """
    Executes benchmark queries with atomic file-swap checkpointing.

    Prevents data loss during interrupted runs by serializing state after each query to a
    temporary file and atomically replacing the target checkpoint via os.replace.

    Args:
        queries (list[dict]): Benchmark query specifications.
        R (Retriever): Initialized retriever instance.
        llm (LLM or None): LLM instance.
        out_dir (str or Path): Results directory.
        top_k (int): Candidates retrieved by Bi-Encoder.
        top_n (int): Candidates supplied to LLM context.

    Returns:
        list[dict]: Array of evaluated per-query result records.
    """
    output_dir = Path(out_dir)
    checkpoint_file = output_dir / "checkpoint_results.json"
    temp_checkpoint_file = output_dir / "checkpoint_results.tmp"

    completed_query_map = {}
    if checkpoint_file.exists():
        try:
            loaded_checkpoint = json.loads(checkpoint_file.read_text())
            completed_query_map = {rec["id"]: rec for rec in loaded_checkpoint}
            print(f"[resume] Loaded {len(completed_query_map)} queries from existing checkpoint.")
        except Exception as err:
            print(f"[resume] Could not parse existing checkpoint ({err}). Starting fresh execution.")

    all_results = []
    total_query_count = len(queries)

    for query_idx, query_spec in enumerate(queries, 1):
        query_id = query_spec["id"]

        # Check if query was previously completed in an earlier interrupted run
        if query_id in completed_query_map:
            cached_result = completed_query_map[query_id]
            all_results.append(cached_result)
            print(f"[{query_idx}/{total_query_count}] {query_id:>3} [CACHED] {cached_result['status']:<20} "
                  f"gold rank {cached_result.get('gold_rank_base')} -> {cached_result.get('gold_rank_rerank')}")
            continue

        query_result = run_query(query_spec, R, llm, top_k, top_n)
        all_results.append(query_result)
        completed_query_map[query_id] = query_result

        print(f"[{query_idx}/{total_query_count}] {query_id:>3} {query_result['status']:<20} "
              f"gold rank {query_result.get('gold_rank_base')} -> {query_result.get('gold_rank_rerank')} | "
              f"{query_spec['query'][:55]}")

        # Atomic checkpoint save: Write to temporary file, then rename atomically
        temp_checkpoint_file.write_text(json.dumps(list(completed_query_map.values()), indent=1))
        temp_checkpoint_file.replace(checkpoint_file)

    return all_results


# %% SECTION 10: UNBIASED ADVERSARIAL QUERY AUTO-MINER
# =================================================================================================
# SECTION 10: UNBIASED ADVERSARIAL QUERY AUTO-MINER
# =================================================================================================
def mine_queries(R, llm, n, out_dir, keep=5):
    """
    Auto-mines queries specifically targeting passages where the Bi-Encoder fails.

    Preserves scientific objectivity by evaluating all generated queries without selection bias,
    logging the full outcome distribution before selecting candidate failure studies.

    Args:
        R (Retriever): Initialized retriever instance.
        llm (LLM): Local LLM instance.
        n (int): Target number of candidate queries to synthesize.
        out_dir (str or Path): Results directory.
        keep (int): Maximum number of promoted failure cases to return.

    Returns:
        list[dict]: Selected query specifications that exhibit FIXED_BY_RERANK behavior.
    """
    candidates_cache_path = Path(out_dir) / "mined_candidates.json"
    prng = random.Random(SEED)

    # Filter chunks with substantial text content and minimal table noise
    eligible_chunk_pool = [
        c for c in R.chunks
        if c["n_chars"] > 600 and c["article"] and (sum(ch.isdigit() for ch in c["text"]) / c["n_chars"] < 0.05)
    ]

    candidate_records = json.loads(candidates_cache_path.read_text()) if candidates_cache_path.exists() else []
    seen_chunk_ids = {item["gold_chunk"] for item in candidate_records}

    miner_system_prompt = (
        "Write ONE specific question that can be answered only from the given passage. "
        "Do not copy long phrases, do not mention 'the passage' or 'the text', "
        "and include enough detail to be unambiguous. Output only the question."
    )

    # Sample chunks and generate synthetic queries
    sampled_chunks = prng.sample(eligible_chunk_pool, min(len(eligible_chunk_pool), n * 2))
    for chunk_item in sampled_chunks:
        if len(candidate_records) >= n:
            break
        if chunk_item["chunk_id"] in seen_chunk_ids:
            continue

        raw_question = llm.chat(miner_system_prompt, chunk_item["text"], max_tokens=60)["text"].strip()
        first_line_q = raw_question.split("\n")[0].strip()

        # Quality filters for valid questions
        if len(first_line_q) < 20 or "passage" in first_line_q.lower() or not first_line_q.endswith("?"):
            continue

        candidate_records.append(dict(gold_chunk=chunk_item["chunk_id"], query=first_line_q))
        seen_chunk_ids.add(chunk_item["chunk_id"])
        candidates_cache_path.write_text(json.dumps(candidate_records, indent=1))

    # Evaluate all mined candidates across retrieval pipelines
    rescued_queries = []
    for idx, cand_item in enumerate(candidate_records):
        eval_spec = dict(
            id=f"M{idx+1}",
            category="Buried (auto-mined)",
            query=cand_item["query"],
            gold_ids=[cand_item["gold_chunk"]],
            facts=[],
            is_unanswerable=False
        )
        eval_result = run_query(eval_spec, R, None)
        cand_item.update(
            status=eval_result["status"],
            base_rank=eval_result["gold_rank_base"],
            rerank_rank=eval_result["gold_rank_rerank"]
        )
        if eval_result["status"] == "FIXED_BY_RERANK":
            rescued_queries.append(eval_spec)

    candidates_cache_path.write_text(json.dumps(candidate_records, indent=1))
    outcome_distribution = pd.Series([c["status"] for c in candidate_records]).value_counts().to_dict()
    print(f"[mine] {len(candidate_records)} candidates generated -> Outcome distribution: {outcome_distribution}")
    return rescued_queries[:keep]


# %% SECTION 11: FAIR LATENCY PROFILING & HARDWARE TIMING
# =================================================================================================
# SECTION 11: FAIR LATENCY PROFILING & HARDWARE TIMING
# =================================================================================================
def profile_retrieval(queries, R, repeats=5):
    """
    Profiles microsecond-level retrieval latencies with warm-up cycles and stage separation.

    Methodological Controls:
      - Pre-warms both neural models over 3 dry runs to eliminate PyTorch and CUDA JIT cold-start spikes.
      - Measures Query Embedding, FAISS Index Search, and Cross-Encoder Re-Ranking independently.
      - Repeats measurements (default: 5 times) and computes robust medians and standard deviations.

    Args:
        queries (list[dict]): Benchmark queries.
        R (Retriever): Retriever instance.
        repeats (int): Measurement repetitions per query.

    Returns:
        pd.DataFrame: Table of per-query latency metrics across all pipeline stages.
    """
    # Pre-warm models to eliminate cold-start JIT and PyTorch overhead
    for _ in range(3):
        R.rerank("warm up query for timing", list(range(min(10, len(R.chunks)))))
        R.embed_query("warm up query for timing")

    profile_records = []
    for query_spec in queries:
        query_text = query_spec["query"]
        embedding_durations, faiss_durations, cross_encoder_durations = [], [], []

        for _ in range(repeats):
            # Stage a: Bi-Encoder Query Embedding
            t_start_emb = time.perf_counter()
            query_vec = R.embed_query(query_text)
            t_end_emb = time.perf_counter()

            # Stage b: FAISS Vector Index Search
            candidate_ids, _ = R.search(query_vec, 10)
            t_end_faiss = time.perf_counter()

            # Stage c: Cross-Encoder Joint Scoring
            R.rerank(query_text, candidate_ids)
            t_end_ce = time.perf_counter()

            embedding_durations.append((t_end_emb - t_start_emb) * 1e3)
            faiss_durations.append((t_end_faiss - t_end_emb) * 1e3)
            cross_encoder_durations.append((t_end_ce - t_end_faiss) * 1e3)

        median_func = statistics.median
        stdev_func = statistics.stdev if repeats > 1 else lambda _: 0.0

        profile_records.append(dict(
            id=query_spec["id"],
            embed_median_ms=median_func(embedding_durations),
            faiss_median_ms=median_func(faiss_durations),
            ce_median_ms=median_func(cross_encoder_durations),
            baseline_retrieval_ms=median_func(embedding_durations) + median_func(faiss_durations),
            rerank_retrieval_ms=median_func(embedding_durations) + median_func(faiss_durations) + median_func(cross_encoder_durations),
            retrieval_overhead_ms=median_func(cross_encoder_durations),
            ce_std_ms=stdev_func(cross_encoder_durations)
        ))

    return pd.DataFrame(profile_records)


# %% SECTION 12: EMPIRICAL SCALING TEST & THEORETICAL EXTRAPOLATION
# =================================================================================================
# SECTION 12: EMPIRICAL SCALING TEST & THEORETICAL EXTRAPOLATION
# =================================================================================================
def scaling_test(R, query, repeats=5):
    """
    Rigorously benchmarks FAISS flat vector search vs. exhaustive Cross-Encoder scoring across corpus sizes.

    Tests empirical subsets (N in {10, 50, 100, 250, 500, 880}) and extrapolates to enterprise
    scale (N up to 10M chunks) based on observed CPU inference constants.

    Demonstrates why exhaustive cross-encoding across entire collections is computationally intractable.

    Args:
        R (Retriever): Retriever instance.
        query (str): Representative query string.
        repeats (int): Timing iterations for cross-encoder inference.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (empirical_measurements_df, theoretical_extrapolations_df)
    """
    total_corpus_size = len(R.chunks)
    test_subsets = sorted({size for size in (10, 50, 100, 250, 500) if size < total_corpus_size} | {total_corpus_size})
    query_vector = R.embed_query(query)
    empirical_rows = []

    for subset_size in test_subsets:
        # Build subset FAISS flat index
        subset_index = R.faiss.IndexFlatIP(R.emb.shape[1])
        subset_index.add(R.emb[:subset_size])

        # Benchmark FAISS search across 20 iterations
        faiss_measurements = []
        for _ in range(20):
            t_start = time.perf_counter()
            subset_index.search(query_vector, min(10, subset_size))
            faiss_measurements.append((time.perf_counter() - t_start) * 1e3)

        # Benchmark Cross-Encoder inference across subset_size pairs
        ce_measurements = []
        chunk_pairs = [(query, R.chunks[idx]["text"]) for idx in range(subset_size)]
        for _ in range(repeats):
            t_start = time.perf_counter()
            R.ce.predict(chunk_pairs, batch_size=32, show_progress_bar=False)
            ce_measurements.append((time.perf_counter() - t_start) * 1e3)

        empirical_rows.append(dict(
            n_chunks=subset_size,
            faiss_median_ms=statistics.median(faiss_measurements),
            faiss_std_ms=statistics.stdev(faiss_measurements) if len(faiss_measurements) > 1 else 0.0,
            cross_encoder_median_ms=statistics.median(ce_measurements),
            cross_encoder_std_ms=statistics.stdev(ce_measurements) if len(ce_measurements) > 1 else 0.0
        ))

    empirical_df = pd.DataFrame(empirical_rows)
    # Compute per-unit empirical constants from the full corpus run (N=880)
    per_pair_ce_ms = empirical_df.cross_encoder_median_ms.iloc[-1] / empirical_df.n_chunks.iloc[-1]
    per_vector_faiss_ms = empirical_df.faiss_median_ms.iloc[-1] / empirical_df.n_chunks.iloc[-1]

    # Theoretical linear extrapolation to enterprise corpus scales
    extrapolation_rows = [
        dict(
            corpus_chunks=n_scale,
            estimated_cross_encoder_seconds=round(per_pair_ce_ms * n_scale / 1e3, 2),
            estimated_faiss_flat_ms=round(per_vector_faiss_ms * n_scale, 2),
            note="Theoretical linear extrapolation assuming fixed batch size and single-thread CPU execution"
        )
        for n_scale in (10_000, 100_000, 1_000_000, 10_000_000)
    ]
    extrapolation_df = pd.DataFrame(extrapolation_rows)
    return empirical_df, extrapolation_df


# %% SECTION 13: CANDIDATE POOL SIZE SWEEP (PARETO FRONTIER)
# =================================================================================================
# SECTION 13: CANDIDATE POOL SIZE SWEEP (PARETO FRONTIER)
# =================================================================================================
def candidate_pool_sweep(queries, R, k_values=(3, 5, 10, 20)):
    """
    Evaluates candidate pool size k to map the Pareto frontier between recall and latency.

    Sweeps k across {3, 5, 10, 20} and measures:
      - Candidate Recall@k: Whether gold chunk exists within initial Bi-Encoder candidates.
      - Hit@3 (Re-Ranked) : Whether gold chunk was promoted into the Top-3 LLM context window.
      - Cross-Encoder Latency: Inference cost scaling with pool size k.

    Args:
        queries (list[dict]): Benchmark queries.
        R (Retriever): Retriever instance.
        k_values (tuple[int]): Candidate pool sizes to test.

    Returns:
        pd.DataFrame: Pareto trade-off table across all tested k values.
    """
    sweep_records = []
    answerable_queries = [q for q in queries if not q.get("is_unanswerable", False)]

    for candidate_pool_size in k_values:
        recalls_at_k, hits_at_3, ce_timing_measurements = [], [], []

        for query_spec in answerable_queries:
            query_vec = R.embed_query(query_spec["query"])
            retrieved_ids, _ = R.search(query_vec, candidate_pool_size)
            gold_ids = set(query_spec.get("gold_ids", []))
            recalls_at_k.append(float(bool(gold_ids & set(retrieved_ids))))

            # Measure Cross-Encoder re-ranking latency for this pool size
            t_start = time.perf_counter()
            ce_logits = R.rerank(query_spec["query"], retrieved_ids)
            ce_timing_measurements.append((time.perf_counter() - t_start) * 1e3)

            # Re-rank candidates and check if gold evidence appears in Top-3
            sorted_candidate_indices = sorted(range(len(retrieved_ids)), key=lambda idx: -ce_logits[idx])
            top_3_reranked_ids = [retrieved_ids[idx] for idx in sorted_candidate_indices[:3]]
            hits_at_3.append(float(bool(gold_ids & set(top_3_reranked_ids))))

        sweep_records.append(dict(
            candidate_pool_k=candidate_pool_size,
            candidate_recall_at_k=round(float(np.mean(recalls_at_k)), 3),
            hit_at_3_after_rerank=round(float(np.mean(hits_at_3)), 3),
            cross_encoder_median_ms=round(statistics.median(ce_timing_measurements), 1),
            cross_encoder_mean_ms=round(float(np.mean(ce_timing_measurements)), 1)
        ))

    return pd.DataFrame(sweep_records)


# %% SECTION 14: STATISTICAL INFERENCE & BOOTSTRAP CONFIDENCE INTERVALS
# =================================================================================================
# SECTION 14: STATISTICAL INFERENCE & BOOTSTRAP CONFIDENCE INTERVALS
# =================================================================================================
def bootstrap_ci(values, n_boot=1000, ci=0.95):
    """
    Computes non-parametric empirical bootstrap confidence intervals over benchmark queries.

    Treats queries as independent experimental units and resamples with replacement B=1,000 times
    to calculate robust distribution-free 95% confidence intervals.

    Args:
        values (list or np.ndarray): Metric values per query.
        n_boot (int): Bootstrap resampling iterations (default: 1,000).
        ci (float): Confidence level (default: 0.95).

    Returns:
        tuple[float, float, float]: (sample_mean, lower_percentile_bound, upper_percentile_bound)
    """
    clean_values = np.array([v for v in values if pd.notna(v)])
    if len(clean_values) == 0:
        return 0.0, 0.0, 0.0

    sample_size = len(clean_values)
    bootstrap_sample_means = [
        float(np.mean(np.random.choice(clean_values, size=sample_size, replace=True)))
        for _ in range(n_boot)
    ]
    alpha = (1.0 - ci) / 2.0
    lower_bound = float(np.percentile(bootstrap_sample_means, 100 * alpha))
    upper_bound = float(np.percentile(bootstrap_sample_means, 100 * (1.0 - alpha)))
    return float(np.mean(clean_values)), lower_bound, upper_bound


def summarize(recs, top_n=3):
    """
    Aggregates headline retrieval metrics, category breakdowns, and outcome distributions.

    Args:
        recs (list[dict]): Per-query result records from run_benchmark.
        top_n (int): Context window cutoff (default: 3).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (headline_summary_df, category_breakdown_df, status_counts_df)
    """
    df = pd.DataFrame(recs)
    answerable_df = df[~df.is_unanswerable].copy()

    # Core retrieval metrics with 95% bootstrap confidence intervals
    cand_rec, cr_low, cr_high = bootstrap_ci(answerable_df.candidate_recall10)
    base_hit, bh_low, bh_high = bootstrap_ci([(pd.notna(x) and x <= top_n) for x in answerable_df.gold_rank_base])
    rerank_hit, rh_low, rh_high = bootstrap_ci([(pd.notna(x) and x <= top_n) for x in answerable_df.gold_rank_rerank])

    base_mrr, bm_low, bm_high = bootstrap_ci([1.0 / x if (pd.notna(x) and x > 0) else 0.0 for x in answerable_df.gold_rank_base])
    rerank_mrr, rm_low, rm_high = bootstrap_ci([1.0 / x if (pd.notna(x) and x > 0) else 0.0 for x in answerable_df.gold_rank_rerank])

    base_ndcg, bn_low, bn_high = bootstrap_ci(answerable_df.ndcg3_base)
    rerank_ndcg, rn_low, rn_high = bootstrap_ci(answerable_df.ndcg3_rerank)

    total_answerable = len(answerable_df)
    n_fixed = (answerable_df.status == "FIXED_BY_RERANK").sum()
    n_degraded = (answerable_df.status == "DEGRADED_BY_RERANK").sum()

    headline_summary = pd.DataFrame([
        dict(
            pipeline="Baseline (Bi-Encoder)",
            Candidate_Recall10=f"{cand_rec:.3f}",
            Hit_at_3=f"{base_hit:.3f} [{bh_low:.3f}, {bh_high:.3f}]",
            MRR_at_10=f"{base_mrr:.3f} [{bm_low:.3f}, {bm_high:.3f}]",
            nDCG_at_3=f"{base_ndcg:.3f} [{bn_low:.3f}, {bn_high:.3f}]",
            Mean_Gold_Rank=round(float(np.mean([x if pd.notna(x) else 11 for x in answerable_df.gold_rank_base])), 2),
            Regression_Rate="N/A",
            Fix_Rate="N/A"
        ),
        dict(
            pipeline="Re-Ranked (+Cross-Encoder)",
            Candidate_Recall10=f"{cand_rec:.3f}",
            Hit_at_3=f"{rerank_hit:.3f} [{rh_low:.3f}, {rh_high:.3f}]",
            MRR_at_10=f"{rerank_mrr:.3f} [{rm_low:.3f}, {rm_high:.3f}]",
            nDCG_at_3=f"{rerank_ndcg:.3f} [{rn_low:.3f}, {rn_high:.3f}]",
            Mean_Gold_Rank=round(float(np.mean([x if pd.notna(x) else 11 for x in answerable_df.gold_rank_rerank])), 2),
            Regression_Rate=f"{n_degraded / total_answerable:.1%} ({n_degraded}/{total_answerable})",
            Fix_Rate=f"{n_fixed / total_answerable:.1%} ({n_fixed}/{total_answerable})"
        )
    ])

    category_breakdown = df.groupby("category").apply(lambda g: pd.Series(dict(
        n=len(g),
        candidate_recall10=round(float(np.mean([x for x in g.candidate_recall10 if pd.notna(x)])), 3) if any(pd.notna(g.candidate_recall10)) else None,
        base_hit3=round(float(np.mean([(pd.notna(x) and x <= top_n) for x in g.gold_rank_base])), 3) if any(pd.notna(g.gold_rank_base)) else None,
        rerank_hit3=round(float(np.mean([(pd.notna(x) and x <= top_n) for x in g.gold_rank_rerank])), 3) if any(pd.notna(g.gold_rank_rerank)) else None,
        fixed=(g.status == "FIXED_BY_RERANK").sum(),
        degraded=(g.status == "DEGRADED_BY_RERANK").sum(),
        both_ok=(g.status == "BOTH_OK").sum(),
        gold_missing=(g.status == "GOLD_NOT_IN_TOP10").sum(),
        unanswerable=(g.status == "UNANSWERABLE").sum()
    )), include_groups=False).reset_index()

    status_distribution = df.status.value_counts().rename_axis("status").reset_index(name="count")
    return headline_summary, category_breakdown, status_distribution


# %% SECTION 15: VISUALIZATION GENERATORS (MATPLOTLIB)
# =================================================================================================
# SECTION 15: VISUALIZATION GENERATORS (MATPLOTLIB)
# =================================================================================================
def make_plots(recs, lat_df, scal_df, out_dir):
    """
    Renders publication-ready scientific visualizations for rank shifts, latencies, and scaling.

    Generated Figures:
      1. figures/rank_shift.png : Horizontal paired bar chart of gold evidence rank displacements.
      2. figures/latency.png    : Stage-wise latency bar comparison between baseline and re-ranking.
      3. figures/scaling.png    : Log-log empirical scaling comparison of FAISS vs Cross-Encoder.

    Args:
        recs (list[dict]): Per-query result records.
        lat_df (pd.DataFrame): Latency profiling dataframe.
        scal_df (pd.DataFrame): Empirical scaling measurements dataframe.
        out_dir (str or Path): Results directory.
    """
    import matplotlib
    matplotlib.use("Agg")  # Headless non-interactive backend
    import matplotlib.pyplot as plt

    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------------------------------
    # Figure 1: Rank Shift Promotion Bar Chart
    # ---------------------------------------------------------------------------------------------
    valid_shift_records = [
        r for r in recs if r.get("gold_rank_base") is not None and r.get("gold_rank_rerank") is not None
    ]
    shifted_records = [r for r in valid_shift_records if r["gold_rank_base"] != r["gold_rank_rerank"]]
    # Sort descending by positive rank shift
    shifted_records.sort(key=lambda r: r["gold_rank_base"] - r["gold_rank_rerank"], reverse=True)
    selected_cases = shifted_records[:20]

    if selected_cases:
        fig, ax = plt.subplots(figsize=(10, 0.45 * len(selected_cases) + 2))
        y_positions = np.arange(len(selected_cases))
        ax.barh(y_positions - 0.2, [r["gold_rank_base"] for r in selected_cases], 0.38,
                label="Baseline Rank (Bi-Encoder)", color="#d9534f")
        ax.barh(y_positions + 0.2, [r["gold_rank_rerank"] for r in selected_cases], 0.38,
                label="Re-Ranked Rank (Cross-Encoder)", color="#5cb85c")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{r['id']} ({r['category'][:10]})" for r in selected_cases])
        ax.invert_yaxis()  # Highest promotion at the top
        ax.axvline(3.5, ls="--", color="black", lw=1.5, label="Top-3 LLM Context Threshold")
        ax.set_xlabel("Rank of Gold Evidence Chunk (lower is better; <= 3 enters LLM context)")
        ax.set_title("Gold Chunk Rank Shifts: Baseline vs Cross-Encoder Re-Ranking")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "rank_shift.png", dpi=160)
        plt.close(fig)

    # ---------------------------------------------------------------------------------------------
    # Figure 2: Stage-wise Latency Comparison
    # ---------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["Bi-Encoder (Flat)", "Cross-Encoder Re-Ranking"],
           [lat_df.baseline_retrieval_ms.median(), lat_df.rerank_retrieval_ms.median()],
           color=["#4575b4", "#f46d43"], width=0.55)
    ax.set_ylabel("Median Retrieval Latency (ms)")
    ax.set_title("Retrieval Stage Latency Comparison")
    for i, v in enumerate([lat_df.baseline_retrieval_ms.median(), lat_df.rerank_retrieval_ms.median()]):
        ax.text(i, v + 5, f"{v:.1f} ms", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "latency.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------------------------------------
    # Figure 3: Log-Log Empirical Scaling Comparison
    # ---------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(scal_df.n_chunks, scal_df.faiss_median_ms, "o-",
              label="FAISS IndexFlatIP (Cosine Search)", color="#313695")
    ax.loglog(scal_df.n_chunks, scal_df.cross_encoder_median_ms, "s-",
              label="Cross-Encoder (Joint Attention All-Pairs)", color="#d73027")
    ax.set_xlabel("Number of Chunks Scored (Corpus Size)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Computational Cost: Vector Search vs Cross-Encoder Scoring")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "scaling.png", dpi=160)
    plt.close(fig)


# %% SECTION 16: CLI CONTROLLER & EXPERIMENT ORCHESTRATION
# =================================================================================================
# SECTION 16: CLI CONTROLLER & EXPERIMENT ORCHESTRATION
# =================================================================================================
def main():
    """
    Main driver executing the end-to-end benchmark workflow:
      1. Parse CLI arguments and seed random number generators.
      2. Ingest and sanitize the Constitution PDF into verified semantic chunks.
      3. Instantiate Bi-Encoder, FAISS IndexFlatIP, and Cross-Encoder retriever models.
      4. Align curated benchmark queries with ground-truth chunk evidence via regex matching.
      5. Execute the primary 56-query benchmark with atomic resumable checkpointing.
      6. Record candidate rank shifts, outcome classifications, and headline summary metrics.
      7. Benchmark microsecond-precision stage-wise latencies and empirical scaling curves.
      8. Conduct candidate pool size sensitivity sweep (k in {3, 5, 10, 20}).
      9. Generate publication-ready figures and comprehensive markdown report tables.
    """
    parser = argparse.ArgumentParser(
        description="RAG Assignment: Baseline vs Re-Ranking on The Constitution of India"
    )
    parser.add_argument("--pdf", default=PDF_PATH, help="Path to Constitution PDF document")
    parser.add_argument("--llm", default=LLM_MODEL, help=f"Ollama LLM model name (default: {LLM_MODEL})")
    parser.add_argument("--out", default="results", help="Directory for output deliverables")
    parser.add_argument("--top-k", type=int, default=10, help="Initial candidates retrieved by bi-encoder (default: 10)")
    parser.add_argument("--top-n", type=int, default=3, help="Top passages supplied to LLM context (default: 3)")
    parser.add_argument("--repeats", type=int, default=5, help="Latency measurement repeats per query (default: 5)")
    parser.add_argument("--skip-llm", action="store_true", help="Retrieval-only run (skips LLM generation for rapid testing)")
    parser.add_argument("--mine", type=int, default=0, help="Number of candidate queries to auto-mine with LLM (unselected)")
    parser.add_argument("--mine-keep", type=int, default=5, help="Number of mined buried-chunk examples to retain")
    parser.add_argument("--show-chunks", type=int, default=0, help="Print sample chunks and exit")
    args = parser.parse_args()

    # Validate ranking parameters
    assert args.top_k >= args.top_n >= 1, f"Invalid ranking parameters: top_k={args.top_k}, top_n={args.top_n}"

    # Initialize deterministic seeds for pseudo-random number generators
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------------------------------
    # Step 1: Corpus Ingestion & Semantic Chunk Verification
    # ---------------------------------------------------------------------------------------------
    chunks_cache_file = output_dir / "chunks.jsonl"
    if chunks_cache_file.exists():
        chunks = [json.loads(line) for line in chunks_cache_file.read_text().splitlines()]
    else:
        chunks = build_chunks(args.pdf)
        chunks_cache_file.write_text("\n".join(json.dumps(c) for c in chunks))
    verify_chunks(chunks)

    if args.show_chunks:
        for sample_chunk in random.sample(chunks, args.show_chunks):
            print(f"\n--- chunk {sample_chunk['chunk_id']} | p.{sample_chunk['page_start']} | "
                  f"art={sample_chunk['article']} ---\n{sample_chunk['text']}")
        return

    # ---------------------------------------------------------------------------------------------
    # Step 2: Retriever & Local LLM Client Instantiation
    # ---------------------------------------------------------------------------------------------
    R = Retriever(chunks, output_dir)
    llm = None if args.skip_llm else LLM(args.llm, output_dir / "llm_cache.json")

    # ---------------------------------------------------------------------------------------------
    # Step 3: Ground Truth Evidence Alignment
    # ---------------------------------------------------------------------------------------------
    normalized_chunks = [norm(c["text"]) for c in chunks]
    curated_queries, unmatched_query_ids = [], []

    for query_spec in CURATED_QUERIES:
        if query_spec.get("is_unanswerable", False):
            curated_queries.append({**query_spec, "gold_ids": []})
            continue

        # Match gold evidence regex against normalized chunk text
        matching_chunk_ids = [
            chunk_idx for chunk_idx, chunk_text in enumerate(normalized_chunks)
            if re.search(norm(query_spec["gold"]), chunk_text, re.IGNORECASE)
        ]

        if not matching_chunk_ids:
            unmatched_query_ids.append(query_spec["id"])
            continue
        curated_queries.append({**query_spec, "gold_ids": matching_chunk_ids})

    if unmatched_query_ids:
        print(f"[warn] Gold regex failed to match any chunk for: {unmatched_query_ids}")

    # ---------------------------------------------------------------------------------------------
    # Step 4: Optional Unbiased Query Mining
    # ---------------------------------------------------------------------------------------------
    mined_spec_list = []
    if args.mine > 0:
        if llm is None:
            sys.exit("--mine requires the LLM (cannot be combined with --skip-llm)")
        print(f"[mine] Mining {args.mine} candidates...")
        mined_spec_list = mine_queries(R, llm, args.mine, output_dir, keep=args.mine_keep)

    # Primary benchmark evaluates exclusively curated queries to eliminate selection bias
    benchmark_queries = list(curated_queries)
    (output_dir / "queries.json").write_text(json.dumps(benchmark_queries, indent=2))

    # ---------------------------------------------------------------------------------------------
    # Step 5: Execute Primary Benchmark with Resumable Checkpoints
    # ---------------------------------------------------------------------------------------------
    eval_records = run_benchmark(benchmark_queries, R, llm, output_dir, args.top_k, args.top_n)
    (output_dir / "per_query_results.json").write_text(json.dumps(eval_records, indent=2))

    # ---------------------------------------------------------------------------------------------
    # Step 6: Export Candidate Rank Shift Audit Trail
    # ---------------------------------------------------------------------------------------------
    candidate_shift_rows = [dict(**shift_rec) for q_rec in eval_records for shift_rec in q_rec["top10"]]
    pd.DataFrame(candidate_shift_rows).to_csv(output_dir / "rank_shift.csv", index=False)

    # ---------------------------------------------------------------------------------------------
    # Step 7: Compute & Export Metrics Summaries
    # ---------------------------------------------------------------------------------------------
    summary_df, category_df, status_counts_df = summarize(eval_records, args.top_n)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    category_df.to_csv(output_dir / "metrics_by_category.csv", index=False)
    status_counts_df.to_csv(output_dir / "status_counts.csv", index=False)

    # ---------------------------------------------------------------------------------------------
    # Step 8: Export Answers Comparison
    # ---------------------------------------------------------------------------------------------
    comparison_columns = [
        "id", "category", "query", "expected_answer", "status", "gold_rank_base", "gold_rank_rerank"
    ]
    if llm is not None:
        full_results_df = pd.DataFrame(eval_records)
        evaluation_columns = [
            "same_prompt_chunks", "same_prompt_order",
            "answer1", "answer2",
            "correctness1", "correctness2",
            "completeness1", "completeness2",
            "citation1", "citation2",
            "grounding1", "grounding2",
            "abstention1", "abstention2",
            "llm_seconds1", "llm_seconds2"
        ]
        answers_comparison_df = full_results_df[comparison_columns + evaluation_columns].assign(manual_notes="")
    else:
        answers_comparison_df = pd.DataFrame(eval_records)[comparison_columns]
    answers_comparison_df.to_csv(output_dir / "answers_comparison.csv", index=False)

    # ---------------------------------------------------------------------------------------------
    # Step 9: Stage-wise Latency Profiling
    # ---------------------------------------------------------------------------------------------
    print("[latency] Profiling retrieval stages with warm-up cycles...")
    latency_df = profile_retrieval(benchmark_queries, R, args.repeats)
    latency_df.to_csv(output_dir / "latency.csv", index=False)

    latency_summary_rows = [
        dict(stage="embed_query (Bi-Encoder)", median_ms=round(latency_df.embed_median_ms.median(), 2), mean_ms=round(latency_df.embed_median_ms.mean(), 2)),
        dict(stage="faiss_search (IndexFlatIP)", median_ms=round(latency_df.faiss_median_ms.median(), 2), mean_ms=round(latency_df.faiss_median_ms.mean(), 2)),
        dict(stage="cross_encoder_rerank (10 candidates)", median_ms=round(latency_df.ce_median_ms.median(), 2), mean_ms=round(latency_df.ce_median_ms.mean(), 2)),
        dict(stage="total_baseline_retrieval", median_ms=round(latency_df.baseline_retrieval_ms.median(), 2), mean_ms=round(latency_df.baseline_retrieval_ms.mean(), 2)),
        dict(stage="total_rerank_retrieval", median_ms=round(latency_df.rerank_retrieval_ms.median(), 2), mean_ms=round(latency_df.rerank_retrieval_ms.mean(), 2))
    ]
    if llm is not None:
        # Measure fresh/uncached LLM generation times
        fresh_latencies_1 = [r["llm_seconds1"] * 1e3 for r in eval_records if not r.get("llm_cached1", True)]
        fresh_latencies_2 = [r["llm_seconds2"] * 1e3 for r in eval_records if not r.get("llm_cached2", True)]
        if fresh_latencies_1:
            latency_summary_rows.append(dict(
                stage="llm_generation_fresh_answer1",
                median_ms=round(statistics.median(fresh_latencies_1), 2),
                mean_ms=round(float(np.mean(fresh_latencies_1)), 2)
            ))
        if fresh_latencies_2:
            latency_summary_rows.append(dict(
                stage="llm_generation_fresh_answer2",
                median_ms=round(statistics.median(fresh_latencies_2), 2),
                mean_ms=round(float(np.mean(fresh_latencies_2)), 2)
            ))

    latency_summary_df = pd.DataFrame(latency_summary_rows)
    latency_summary_df.to_csv(output_dir / "latency_summary.csv", index=False)

    # ---------------------------------------------------------------------------------------------
    # Step 10: Empirical Scaling Experiment & Candidate Pool Sweep
    # ---------------------------------------------------------------------------------------------
    print("[scaling] Running empirical scaling test across corpus subsets...")
    scaling_df, projection_df = scaling_test(R, benchmark_queries[0]["query"], repeats=5)
    scaling_df.to_csv(output_dir / "latency_scaling.csv", index=False)
    projection_df.to_csv(output_dir / "latency_projection.csv", index=False)

    print("[sweep] Running candidate pool size sweep (k in {3, 5, 10, 20})...")
    sweep_results_df = candidate_pool_sweep(benchmark_queries, R, k_values=(3, 5, 10, 20))
    sweep_results_df.to_csv(output_dir / "candidate_pool_sweep.csv", index=False)

    # ---------------------------------------------------------------------------------------------
    # Step 11: Render Visualizations
    # ---------------------------------------------------------------------------------------------
    make_plots(eval_records, latency_df, scaling_df, output_dir)

    # ---------------------------------------------------------------------------------------------
    # Step 12: Generate Comprehensive Markdown Deliverables Report
    # ---------------------------------------------------------------------------------------------
    fixed_queries = [r for r in eval_records if r["status"] == "FIXED_BY_RERANK"]
    degraded_queries = [r for r in eval_records if r["status"] == "DEGRADED_BY_RERANK"]

    markdown_report_sections = [
        "# RAG Benchmark Report: Baseline vs. Re-Ranking on The Constitution of India\n",
        "## 1. Executive Retrieval Metrics (Primary Benchmark, 95% Bootstrap Confidence Intervals)\n",
        summary_df.to_markdown(index=False),
        "\n## 2. Category-Wise Retrieval Breakdown\n",
        category_df.to_markdown(index=False),
        "\n## 3. Query Outcome Classification\n",
        status_counts_df.to_markdown(index=False),
        "\n## 4. Candidate Pool Size Sweep (k=3, 5, 10, 20: Recall vs Latency Trade-Off)\n",
        sweep_results_df.to_markdown(index=False),
        "\n## 5. Latency Profiling (ms)\n",
        latency_summary_df.to_markdown(index=False),
        "\n## 6. Empirical Scaling vs Theoretical Projection\n",
        scaling_df.round(2).to_markdown(index=False),
        "\n### Linear Extrapolation to Larger Corpora\n",
        projection_df.to_markdown(index=False),
        "\n## 7. Genuine Failure Cases: Candidate Rank Shifts & Answer Impact\n"
    ]

    # Include detailed 10-candidate rank shift tables for demonstration cases
    demo_cases = fixed_queries[:3] if fixed_queries else eval_records[:2]
    for demo in demo_cases:
        markdown_report_sections.append(f"\n### Case Study: {demo['id']} ({demo['category']}) - '{demo['query']}'")
        markdown_report_sections.append(f"- **Expected Legal Answer**: {demo['expected_answer']}")
        markdown_report_sections.append(f"- **Status**: `{demo['status']}` | Gold Rank Shift: `#{demo.get('gold_rank_base')}` → `#{demo.get('gold_rank_rerank')}`\n")

        candidate_table_rows = []
        for cand in demo["top10"]:
            candidate_table_rows.append(dict(
                Base_Rank=cand["base_rank"],
                CE_Rank=cand["ce_rank"],
                Shift=cand["rank_shift"],
                Base_Score=cand["base_score"],
                CE_Score=cand["ce_score"],
                Is_Gold="YES" if cand["is_gold"] else "no",
                Article=cand["article"],
                Page=cand["page"],
                Text_Snippet=cand["preview"][:100] + "..."
            ))
        markdown_report_sections.append(pd.DataFrame(candidate_table_rows).to_markdown(index=False))

        if llm is not None:
            markdown_report_sections.append("\n**LLM Answer 1 (Baseline Top-3 Context):**")
            markdown_report_sections.append(f"> {demo.get('answer1', 'N/A')}")
            markdown_report_sections.append("\n**LLM Answer 2 (Re-Ranked Top-3 Context):**")
            markdown_report_sections.append(f"> {demo.get('answer2', 'N/A')}")
            markdown_report_sections.append(
                f"\n**Answer Analysis**: Baseline Completeness: {demo.get('completeness1')}, "
                f"Re-Ranked Completeness: {demo.get('completeness2')}, "
                f"Correctness Shift: {demo.get('correctness1')} → {demo.get('correctness2')}\n"
            )

    if degraded_queries:
        markdown_report_sections.append("\n## 8. Documented Regression Cases (Degraded by Re-Ranking)\n")
        for deg in degraded_queries:
            markdown_report_sections.append(
                f"- **{deg['id']}** ({deg['category']}): '{deg['query']}' "
                f"(Gold rank dropped from #{deg['gold_rank_base']} to #{deg['gold_rank_rerank']})"
            )

    (output_dir / "report_tables.md").write_text("\n".join(markdown_report_sections))

    # ---------------------------------------------------------------------------------------------
    # Step 13: System Hardware & Software Provenance Record
    # ---------------------------------------------------------------------------------------------
    import faiss, sentence_transformers, torch
    (output_dir / "run_config.json").write_text(json.dumps(dict(
        corpus=PDF_PATH,
        bi_encoder=BI_ENCODER,
        cross_encoder=CROSS_ENCODER,
        llm=None if llm is None else args.llm,
        chunk_chars=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        pages=[FIRST_PAGE, LAST_PAGE],
        n_chunks=len(chunks),
        top_k=args.top_k,
        top_n=args.top_n,
        seed=SEED,
        n_queries=len(benchmark_queries),
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        sentence_transformers=sentence_transformers.__version__,
        faiss=faiss.__version__,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    ), indent=2))

    print("\n" + "="*80)
    print("BENCHMARK EXECUTION COMPLETE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("\nOutcome Distribution:")
    print(status_counts_df.to_string(index=False))
    print(f"\nAll deliverables and artifacts successfully verified and saved to {output_dir}/")


if __name__ == "__main__":
    main()
