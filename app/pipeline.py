"""
The core self-healing loop:

  1. Retrieve chunks for the current query
  2. Generate an answer from those chunks
  3. Critique the answer (LLM judge + embedding grounding score)
  4. If grounded -> return it
     If hallucinated and retries remain -> reformulate the query, retry
     If retries exhausted -> return an honest fallback, not a guess

Every attempt is logged into the trace so the API response makes the
self-healing behavior visible (this is the whole point of the project).
"""
from app.config import settings
from app.critic import critique
from app.llm import get_llm
from app.models import AskResponse, Attempt
from app.vector_store import get_vector_store

FALLBACK_ANSWER = "I don't have enough information in the provided documents to answer this reliably."


def run_pipeline(
    question: str,
    corpus_id: str,
    top_k: int = None,
    max_retries: int = None,
) -> AskResponse:
    top_k = top_k or settings.top_k
    max_retries = settings.max_retries if max_retries is None else max_retries

    vector_store = get_vector_store()
    llm = get_llm()

    attempts = []
    current_query = question
    attempt_number = 1

    while True:
        retrieved = vector_store.query(corpus_id, current_query, top_k)
        answer = llm.generate_answer(question, retrieved)
        result = critique(question, answer, retrieved)
        accepted = result.verdict == "grounded"

        attempts.append(
            Attempt(
                attempt_number=attempt_number,
                query_used=current_query,
                retrieved_chunks=retrieved,
                answer=answer,
                critique=result,
                accepted=accepted,
            )
        )

        if accepted:
            return AskResponse(
                question=question,
                final_answer=answer,
                was_fallback=False,
                total_attempts=attempt_number,
                attempts=attempts,
            )

        if attempt_number > max_retries:
            return AskResponse(
                question=question,
                final_answer=FALLBACK_ANSWER,
                was_fallback=True,
                total_attempts=attempt_number,
                attempts=attempts,
            )

        # Reformulate the query using the critic's feedback before retrying.
        feedback = result.reasoning or "Answer was judged not grounded in the retrieved context."
        current_query = llm.reformulate_query(question, answer, feedback)
        attempt_number += 1
