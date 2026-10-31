import time

from agents.generator import NOT_AVAILABLE_MESSAGE
from workflow.retrieval import summarize_sources

UNVALIDATED_DISCLAIMER = (
    "Disclaimer: this answer could not be fully validated by the Evaluator "
    "(maximum number of iterations reached or the evaluation was unavailable). "
    "Please verify it against the sources."
)


def _timings(started, retrieval_s=None):
    timings = {} if retrieval_s is None else {"retrieval_s": retrieval_s}
    timings["total_s"] = round(time.perf_counter() - started, 3)
    return timings


def cached_result(cached, started):
    """A previously approved result, marked as coming from the cache."""
    result = dict(cached)
    result["from_cache"] = True
    result["timings"] = _timings(started)
    return result


def no_knowledge_result(question, retrieval_s, started):
    """Nothing relevant was found, so no LLM was called."""
    return {
        "question": question,
        "answer": NOT_AVAILABLE_MESSAGE,
        "approved": False,
        "status": "no_knowledge",
        "iterations": 0,
        "history": [],
        "sources": [],
        "message": "No relevant information was found in the provided sources, "
                   "so no answer was generated.",
        "from_cache": False,
        "timings": _timings(started, retrieval_s),
    }


def history_item(iteration, answer, evaluation, approved, started):
    return {
        "iteration": iteration,
        "answer": answer,
        "approved": approved,
        "feedback": evaluation["feedback"],
        "scores": evaluation.get("scores", {}),
        "unsupported_claims": evaluation.get("unsupported_claims", []),
        "seconds": round(time.perf_counter() - started, 3),
    }


def final_result(question, answer, approved, note, history, chunks, retrieval_s, started):
    if approved:
        status = "approved"
        message = "Answer approved by the Evaluator."
        final_answer = answer
    else:
        status = "unvalidated"
        reason = note or (
            f"The maximum number of iterations ({4}) was reached "
            "without an approval."
        )
        message = f"The answer could not be fully validated. {reason}"
        final_answer = f"{answer}\n\n---\n{UNVALIDATED_DISCLAIMER}"

    return {
        "question": question,
        "answer": final_answer,
        "approved": approved,
        "status": status,
        "iterations": len(history),
        "history": history,
        "sources": summarize_sources(chunks),
        "message": message,
        "from_cache": False,
        "timings": _timings(started, retrieval_s),
    }
