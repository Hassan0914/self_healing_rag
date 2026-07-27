"""
Groq LLM wrapper. Three responsibilities, kept as separate methods so each
step of the pipeline is independently testable/loggable:

  1. generate_answer   -> produce an answer strictly from retrieved context
  2. critique_answer   -> LLM-as-judge: is the answer grounded in the context?
  3. reformulate_query -> rewrite the query for a better retrieval attempt
"""
import json
import re
from functools import lru_cache
from typing import List, Tuple

from groq import Groq

from app.config import settings
from app.models import RetrievedChunk

SYSTEM_GENERATION_PROMPT = """You are a careful question-answering assistant.
Answer the user's question using ONLY the information in the provided context chunks.
Rules:
- Do not use outside knowledge, even if you know the answer.
- If the context does not contain enough information to answer, say exactly:
  "I don't have enough information in the provided documents to answer this."
- Be concise and factual. Do not pad the answer with unsupported elaboration.
- Every factual claim you make must be traceable to the context."""

SYSTEM_CRITIC_PROMPT = """You are a strict fact-checking critic for a RAG system.
You will be given a QUESTION, an ANSWER, and the CONTEXT CHUNKS that were retrieved to produce it.

Your job: determine whether the ANSWER is fully grounded in the CONTEXT, or whether it contains
hallucinated / unsupported claims not present in the context.

Respond with ONLY a JSON object (no markdown fences, no prose outside the JSON) in this exact shape:
{
  "verdict": "grounded" or "hallucinated",
  "faithfulness_score": <float 0.0 to 1.0, where 1.0 = every claim is fully supported>,
  "unsupported_claims": [<list of strings, each a specific claim from the answer not supported by context>],
  "reasoning": "<one or two sentence explanation>"
}

Be strict: if the answer states something plausible-sounding but not actually present in the context,
that counts as hallucinated. An answer that correctly says "I don't have enough information" when the
context is insufficient should be scored as grounded with faithfulness_score 1.0."""

SYSTEM_REFORMULATE_PROMPT = """You rewrite search queries to improve document retrieval.
You will be given the original question, the answer that was produced, and feedback on why that
answer was judged unreliable (e.g. hallucinated, or the context was insufficient).
Rewrite the query to be more likely to retrieve the RIGHT supporting passages next time.
Respond with ONLY the rewritten query text, nothing else. No quotes, no preamble."""


def _format_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context retrieved)"
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Chunk {i} | source: {c.source}]\n{c.text}")
    return "\n\n".join(parts)


class GroqLLM:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def generate_answer(self, question: str, chunks: List[RetrievedChunk]) -> str:
        context = _format_context(chunks)
        completion = self.client.chat.completions.create(
            model=settings.groq_generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_GENERATION_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        return completion.choices[0].message.content.strip()

    def critique_answer(
        self, question: str, answer: str, chunks: List[RetrievedChunk]
    ) -> Tuple[str, float, List[str], str]:
        context = _format_context(chunks)
        completion = self.client.chat.completions.create(
            model=settings.groq_critic_model,
            messages=[
                {"role": "system", "content": SYSTEM_CRITIC_PROMPT},
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXT CHUNKS:\n{context}",
                },
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content.strip()
        parsed = self._safe_parse_json(raw)

        verdict = parsed.get("verdict", "hallucinated")
        score = float(parsed.get("faithfulness_score", 0.0))
        unsupported = parsed.get("unsupported_claims", []) or []
        reasoning = parsed.get("reasoning", "")
        return verdict, max(0.0, min(1.0, score)), unsupported, reasoning

    def reformulate_query(self, question: str, answer: str, feedback: str) -> str:
        completion = self.client.chat.completions.create(
            model=settings.groq_reformulate_model,
            messages=[
                {"role": "system", "content": SYSTEM_REFORMULATE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original question: {question}\n"
                        f"Previous answer: {answer}\n"
                        f"Why it was rejected: {feedback}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=100,
        )
        rewritten = completion.choices[0].message.content.strip().strip('"')
        return rewritten or question

    @staticmethod
    def _safe_parse_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extract the first {...} block in case the model
            # added stray text despite instructions.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {
                "verdict": "hallucinated",
                "faithfulness_score": 0.0,
                "unsupported_claims": ["Critic response could not be parsed."],
                "reasoning": "Failed to parse critic JSON output.",
            }


@lru_cache(maxsize=1)
def get_llm() -> GroqLLM:
    return GroqLLM(api_key=settings.groq_api_key)
