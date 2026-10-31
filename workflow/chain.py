from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def _evaluation_unavailable(_state):
    """Fallback used when the Evaluator step raises an exception."""
    return {
        "approved": False,
        "feedback": "The Evaluator was unavailable.",
        "scores": {},
        "unsupported_claims": [],
        "error": True,
    }


def build_iteration_chain(generator, evaluator):
    """state -> state + answer + evaluation. """
    generate = RunnableLambda(generator.run, name="generator")
    evaluate = RunnableLambda(evaluator.run, name="evaluator").with_fallbacks(
        [RunnableLambda(_evaluation_unavailable, name="evaluator_fallback")]
    )
    return RunnablePassthrough.assign(answer=generate) | RunnablePassthrough.assign(
        evaluation=evaluate
    )
