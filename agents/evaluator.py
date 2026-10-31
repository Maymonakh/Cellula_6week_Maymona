import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.llm import get_llm
from agents.memory import EvaluatorMemory, MemoryAccessError
from knowledge import cache

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM = """You are the Evaluator in a retrieval-augmented question-answering system.
Judge the generated answer against the retrieved context.

Score every criterion from 1 (very poor) to 5 (excellent):
{criteria}

Rules:
{rules}

Feedback you already gave for this question (check whether it was fixed;
do not repeat feedback that was already addressed):
{previous_feedback}

Return ONLY one valid JSON object, with no Markdown and no other text:
{{
  "approved": true or false,
  "scores": {{{score_schema}}},
  "unsupported_claims": ["each claim of the answer that the context does not support"],
  "feedback": "specific and actionable explanation of what to improve, or why the answer was approved"
}}"""

EVALUATOR_HUMAN = """Original question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}"""

evaluator_prompt = ChatPromptTemplate.from_messages(
    [("system", EVALUATOR_SYSTEM), ("human", EVALUATOR_HUMAN)]
)

REPAIR_SYSTEM = """Convert the evaluation below into one valid JSON object with exactly these fields:
- approved: boolean
- scores: object whose values are integers from 1 to 5
- unsupported_claims: array of strings
- feedback: string

Keep the original decision and feedback. If the decision is unclear use false.
Return ONLY the JSON object, without Markdown or extra text."""

repair_prompt = ChatPromptTemplate.from_messages(
    [("system", REPAIR_SYSTEM), ("human", "{response}")]
)


def build_evaluator_chain():
    return evaluator_prompt | get_llm("openrouter/free") | StrOutputParser()


def build_repair_chain():
    return repair_prompt | get_llm("openrouter/free") | StrOutputParser()


# reading the answer of the Evaluator
def parse_evaluation(text):
    """Turn the text of the Evaluator into a dictionary.
    Returns None if the text is not a valid JSON evaluation."""
    if not isinstance(text, str):
        return None

    # take everything between the first { and the last }
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    approved = data.get("approved")
    if approved in ("true", "True"):
        approved = True
    elif approved in ("false", "False"):
        approved = False
    if not isinstance(approved, bool):
        return None

    feedback = data.get("feedback")
    feedback = feedback.strip() if isinstance(feedback, str) else ""
    if not approved and not feedback:
        feedback = (
            "No specific feedback was given. Re-check that every statement is "
            "supported by the context and that the question is fully answered."
        )

    scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    claims = data.get("unsupported_claims") if isinstance(data.get("unsupported_claims"), list) else []
    claims = [str(c).strip() for c in claims if str(c).strip().lower() not in ("", "none", "n/a")]

    return {"approved": approved, "feedback": feedback, "scores": scores, "unsupported_claims": claims}


class Evaluator:
    def __init__(self, memory, chain=None, repair_chain=None):
        if not isinstance(memory, EvaluatorMemory):
            raise MemoryAccessError("The Evaluator needs an EvaluatorMemory (not the Generator's memory).")
        self._memory = memory
        self._chain = chain
        self._repair_chain = repair_chain

    @property
    def chain(self):
        if self._chain is None:
            self._chain = build_evaluator_chain()
        return self._chain

    @property
    def repair_chain(self):
        if self._repair_chain is None:
            self._repair_chain = build_repair_chain()
        return self._repair_chain

    # the workflow step
    def run(self, state):
        question = state["question"]
        context = state["context"]
        answer = state["answer"]

        if state.get("iteration", 1) == 1:
            self._memory.start_question(question)

        cache_parts = (question, context, answer)
        evaluation = cache.cache_get("eval", *cache_parts)

        if evaluation is None:
            try:
                evaluation = self._evaluate(question, context, answer)
            except Exception:
                logger.exception("Evaluator call failed")
                raise
            if not evaluation.get("error"):
                cache.cache_set("eval", evaluation, *cache_parts)

        self._memory.add_evaluation(question, answer, evaluation)
        logger.info(
            "Evaluator decision: approved=%s scores=%s",
            evaluation["approved"],
            evaluation.get("scores"),
        )
        return evaluation

    def _evaluate(self, question, context, answer):
        raw = self.chain.invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
                "criteria": self._memory.criteria_text(),
                "rules": self._memory.rules_text(),
                "previous_feedback": self._memory.previous_feedback_text(),
                "score_schema": self._memory.score_schema(),
            }
        )
        result = parse_evaluation(raw)

        if result is None:
            logger.warning("Invalid evaluation format; asking the model to repair it")
            result = parse_evaluation(self.repair_chain.invoke({"response": raw}))

        if result is None:
            logger.error("Evaluator output could not be parsed")
            return {
                "approved": False,
                "feedback": "The Evaluator returned an invalid response format.",
                "scores": {},
                "unsupported_claims": [],
                "error": True,
            }
        return result

    def reset_memory(self):
        self._memory.reset()

    def memory_summary(self):
        return {"evaluations": len(self._memory.history)}
