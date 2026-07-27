"""
The critic combines two independent signals so we're not just trusting the
LLM's self-report of its own faithfulness:

  1. llm_faithfulness_score   -> LLM-as-judge score (0-1)
  2. embedding_grounding_score -> for each sentence in the answer, the max
     cosine similarity to any retrieved chunk, averaged across sentences.
     This is a cheap, deterministic, non-LLM proxy for "is this sentence
     actually present in the source material" (semantic entailment lite).

The combined_score is a weighted average of both. Using two signals means
a single miscalibrated LLM critique doesn't get treated as ground truth.
"""
import re
from typing import List

from app.config import settings
from app.embeddings import get_embedding_model
from app.llm import get_llm
from app.models import Critique, RetrievedChunk

LLM_WEIGHT = 0.6
EMBEDDING_WEIGHT = 0.4

FALLBACK_PHRASE = "i don't have enough information"


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter — good enough for grounding checks,
    # avoids pulling in a full NLP tokenizer dependency.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def _embedding_grounding_score(answer: str, chunks: List[RetrievedChunk]) -> float:
    if FALLBACK_PHRASE in answer.lower():
        # The model explicitly declined to answer — that's grounded behavior,
        # not a hallucination, regardless of chunk overlap.
        return 1.0

    if not chunks:
        return 0.0

    embedder = get_embedding_model()
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    sentence_vecs = embedder.encode(sentences)
    chunk_vecs = embedder.encode([c.text for c in chunks])

    per_sentence_max = []
    for s_vec in sentence_vecs:
        sims = [embedder.cosine_sim(s_vec, c_vec) for c_vec in chunk_vecs]
        per_sentence_max.append(max(sims) if sims else 0.0)

    avg = sum(per_sentence_max) / len(per_sentence_max)
    # Cosine sim from MiniLM rarely reaches 1.0 even for near-duplicate text;
    # rescale so a "typical well-grounded" ~0.65 similarity maps closer to 1.0.
    rescaled = min(1.0, avg / 0.75)
    return round(rescaled, 3)


def critique(question: str, answer: str, chunks: List[RetrievedChunk]) -> Critique:
    llm = get_llm()
    verdict, llm_score, unsupported_claims, reasoning = llm.critique_answer(question, answer, chunks)
    embedding_score = _embedding_grounding_score(answer, chunks)

    combined = round(LLM_WEIGHT * llm_score + EMBEDDING_WEIGHT * embedding_score, 3)

    # Final verdict requires BOTH the combined score to clear the bar AND
    # the LLM not to have flagged it outright — the embedding score alone
    # can be fooled by lexical overlap on wrong facts.
    final_verdict = "grounded" if (combined >= settings.faithfulness_threshold and verdict == "grounded") else "hallucinated"

    return Critique(
        verdict=final_verdict,
        llm_faithfulness_score=llm_score,
        embedding_grounding_score=embedding_score,
        combined_score=combined,
        unsupported_claims=unsupported_claims,
        reasoning=reasoning,
    )
