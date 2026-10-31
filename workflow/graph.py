import logging
import time

from agents.generator import NOT_AVAILABLE_MESSAGE
from knowledge import cache
from workflow.chain import build_iteration_chain
from workflow.results import (
    cached_result,
    final_result,
    history_item,
    no_knowledge_result,
)
from workflow.retrieval import format_context, merge_chunks, retrieve

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 4


class WorkflowError(Exception):
    """Raised when no answer could be produced at all."""


def _validate_question(question):
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    if len(question) > 2000:
        raise ValueError(
            f"The question is too long (maximum {2000} characters)."
        )
    return question


def _extend_context(question, chunks, feedback, state):
    """Try to retrieve more knowledge based on the feedback."""
    try:
        extra = retrieve(f"{question}\n{feedback[:300]}")
        merged = merge_chunks(chunks, extra, limit=4 * 2)
        if len(merged) != len(chunks):
            state["context"] = format_context(merged)
            logger.info("Context extended to %s chunks", len(merged))
            return merged
    except Exception:
        logger.exception("Re-retrieval failed; keeping the old context")
    return chunks


def _run_loop(question, chunks, generator, evaluator, use_history):

    chain = build_iteration_chain(generator, evaluator)
    state = {
        "question": question,
        "context": format_context(chunks),
        "feedback": None,
        "iteration": 0,           # the loop counter lives in the state
        "use_history": use_history,
    }
    history, answer, approved, note = [], "", False, ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        state["iteration"] = iteration
        iteration_started = time.perf_counter()

        try:
            output = chain.invoke(state)
        except Exception as exc:
            logger.exception("Iteration %s failed", iteration)
            if not answer:
                raise WorkflowError(
                    "The language model could not be reached. Please try again."
                ) from exc
            note = "A model call failed, so the latest answer could not be improved."
            break

        answer = output["answer"]
        evaluation = output["evaluation"]
        evaluation_failed = bool(evaluation.get("error"))
        approved = bool(evaluation["approved"]) and not evaluation_failed

        history.append(history_item(iteration, answer, evaluation, approved, iteration_started))
        logger.info("Iteration %s/%s finished. approved=%s", iteration, MAX_ITERATIONS, approved)

        if evaluation_failed:
            note = "The Evaluator was unavailable, so the answer could not be validated."
            break
        if approved:
            break

        state["feedback"] = evaluation["feedback"]
        if iteration < MAX_ITERATIONS and True:
            chunks = _extend_context(question, chunks, evaluation["feedback"], state)

    return answer, approved, note, history, chunks


def run_workflow(question, generator, evaluator, use_history=False):
    """Run the full workflow and return a result dictionary."""
    started = time.perf_counter()
    question = _validate_question(question)
    logger.info("Workflow started (follow-up=%s)", use_history)

    # 1) QA cache. Conversation dependent answers include the conversation
    signature = generator.conversation_signature() if use_history else ""
    question_key = cache.normalize_question(question)

    cached = cache.cache_get("qa", question_key, signature)
    if cached:
        generator.remember_turn(question, cached["answer"])
        logger.info("Returned cached answer")
        return cached_result(cached, started)

    # 2) Retrieval
    retrieval_query = question
    if use_history and generator.last_question():
        retrieval_query = f"{generator.last_question()} {question}"

    retrieval_started = time.perf_counter()
    chunks = retrieve(retrieval_query)
    retrieval_s = round(time.perf_counter() - retrieval_started, 3)

    if not chunks:
        generator.remember_turn(question, NOT_AVAILABLE_MESSAGE)
        logger.info("No relevant knowledge found; skipping the LLMs")
        return no_knowledge_result(question, retrieval_s, started)

    # 3) Generator <-> Evaluator loop
    answer, approved, note, history, chunks = _run_loop(
        question, chunks, generator, evaluator, use_history
    )

    # 4) Exit
    generator.finish(answer, approved)
    result = final_result(
        question, answer, approved, note, history, chunks, retrieval_s, started
    )

    if approved:  # never cache answers that were not validated
        cache.cache_set("qa", result, question_key, signature)

    logger.info(
        "Workflow finished: status=%s iterations=%s total=%.2fs",
        result["status"], len(history), result["timings"]["total_s"],
    )
    return result
