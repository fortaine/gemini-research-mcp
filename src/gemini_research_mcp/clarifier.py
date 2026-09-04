"""
LLM-driven query clarification for deep research.

This module implements the "Question-First" pattern (OpenAI-style):
1. Analyze query to determine if clarification is needed
2. Generate contextual clarifying questions using Gemini Flash (fast/cheap)
3. Refine the query based on user answers

The goal is to gather enough context to produce high-quality research results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from google import genai
from google.genai.types import GenerateContentConfig

from gemini_research_mcp.config import LOGGER_NAME, get_api_key

logger = logging.getLogger(LOGGER_NAME)

# Fast model for clarification - Gemini 3.8 Flash for quick, intelligent responses
CLARIFIER_MODEL = "gemini-3.8-flash"

# Maximum questions to avoid user fatigue (research shows 3-5 is optimal)
MAX_QUESTIONS = 5
MIN_QUESTIONS = 2

# Confidence threshold - if query is clear enough, skip clarification
CONFIDENCE_THRESHOLD = 0.7

@dataclass
class ClarifyingQuestion:
    """A single clarifying question with metadata."""

    question: str
    """The question text to present to the user."""

    purpose: str
    """Why this question helps improve research quality."""

    priority: int = 1
    """1=essential, 2=important, 3=nice-to-have."""

    default_answer: str | None = None
    """Optional default/suggested answer."""


@dataclass
class QueryAnalysis:
    """Result of analyzing a query for clarification needs."""

    needs_clarification: bool
    """Whether the query would benefit from clarification."""

    confidence: float
    """0.0-1.0 confidence that we understand the query intent."""

    questions: list[ClarifyingQuestion] = field(default_factory=list)
    """Generated clarifying questions, ordered by priority."""

    detected_intent: str | None = None
    """What we think the user is trying to research."""

    ambiguities: list[str] = field(default_factory=list)
    """Specific ambiguities or gaps identified in the query."""


@dataclass
class RefinedQuery:
    """A query enhanced with user-provided context."""

    original_query: str
    """The original user query."""

    refined_query: str
    """The enhanced query with context incorporated."""

    context_summary: str
    """Summary of the clarifications provided."""

    answers: dict[str, str] = field(default_factory=dict)
    """Question -> Answer mapping for reference."""


# System prompt for query analysis and question generation
CLARIFIER_SYSTEM_PROMPT = """\
You are a research query analyst. Your job is to analyze research queries and generate \
clarifying questions that will help produce better, more targeted research results.

## Your Task
1. Analyze the query to understand what the user wants to research
2. Identify ambiguities, missing context, or areas that need clarification
3. Generate focused questions that will help narrow down the research scope
4. Assess your confidence in understanding the query

## Guidelines for Questions
- Ask only questions that will MEANINGFULLY improve research quality
- Prioritize questions about: scope, time period, specific aspects, intended use case
- Avoid obvious or redundant questions
- Each question should have a clear purpose
- Keep questions concise and easy to answer

## Output Format
Return a JSON object with this structure:
{
    "needs_clarification": true/false,
    "confidence": 0.0-1.0,
    "detected_intent": "What you think they want to research",
    "ambiguities": ["List of specific gaps or ambiguities"],
    "questions": [
        {
            "question": "The question text",
            "purpose": "Why this helps improve research",
            "priority": 1-3,
            "default_answer": "Optional suggested answer or null"
        }
    ]
}

## Examples of Good Questions
- "Are you interested in recent developments (last 1-2 years) or a historical overview?"
- "Should the research focus on technical implementation or business/strategic aspects?"
- "Is there a specific industry or domain context for this research?"
- "What's the intended use of this research (academic paper, business decision, learning)?"

## When to Skip Clarification (confidence >= 0.7)
- Query is already specific with clear scope
- Query includes time constraints, domain, and focus area
- Intent is unambiguous
"""


async def analyze_query(query: str) -> QueryAnalysis:
    """
    Analyze a research query to determine if clarification is needed.

    Uses Gemini Flash to quickly assess the query and generate
    contextual clarifying questions if needed.

    Args:
        query: The user's research query

    Returns:
        QueryAnalysis with confidence score and optional questions
    """
    client = genai.Client(api_key=get_api_key())

    logger.debug("Analyzing query for clarification needs: %s", query[:100])

    try:
        response = await client.aio.models.generate_content(
            model=CLARIFIER_MODEL,
            contents=f"Analyze this research query and generate clarifying questions:\n\n{query}",
            config=GenerateContentConfig(
                system_instruction=CLARIFIER_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.3,  # Low temperature for consistent analysis
            ),
        )

        # Parse JSON response
        response_text = response.text
        if response_text is None:
            raise ValueError("No response text from clarifier")
        result_text = response_text.strip()
        result = json.loads(result_text)

        # Build questions list
        questions = []
        for q in result.get("questions", [])[:MAX_QUESTIONS]:
            questions.append(
                ClarifyingQuestion(
                    question=q.get("question", ""),
                    purpose=q.get("purpose", ""),
                    priority=q.get("priority", 2),
                    default_answer=q.get("default_answer"),
                )
            )

        # Sort by priority
        questions.sort(key=lambda x: x.priority)

        analysis = QueryAnalysis(
            needs_clarification=result.get("needs_clarification", True),
            confidence=float(result.get("confidence", 0.5)),
            questions=questions,
            detected_intent=result.get("detected_intent"),
            ambiguities=result.get("ambiguities", []),
        )

        logger.info(
            "Query analysis: confidence=%.2f, needs_clarification=%s, questions=%d",
            analysis.confidence,
            analysis.needs_clarification,
            len(analysis.questions),
        )

        return analysis

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse clarifier response as JSON: %s", e)
        # Fallback: assume clarification is needed with generic questions
        return QueryAnalysis(
            needs_clarification=True,
            confidence=0.3,
            questions=[
                ClarifyingQuestion(
                    question="What specific aspect of this topic are you most interested in?",
                    purpose="Helps focus the research on what matters to you",
                    priority=1,
                ),
                ClarifyingQuestion(
                    question="Is there a particular time period or context you want to focus on?",
                    purpose="Helps narrow scope and ensure relevance",
                    priority=2,
                ),
            ],
            detected_intent=None,
            ambiguities=["Could not fully analyze query"],
        )
    except Exception as e:
        logger.error("Error analyzing query: %s", e)
        # On error, proceed without clarification rather than blocking
        return QueryAnalysis(
            needs_clarification=False,
            confidence=0.5,
            questions=[],
            detected_intent=None,
            ambiguities=[],
        )


async def refine_query_with_answers(
    original_query: str,
    questions: list[ClarifyingQuestion],
    answers: list[str],
) -> RefinedQuery:
    """
    Refine the original query by incorporating user answers.

    Uses Gemini Flash to intelligently merge the original query
    with the clarification context.

    Args:
        original_query: The user's original research query
        questions: The clarifying questions that were asked
        answers: The user's answers (in same order as questions)

    Returns:
        RefinedQuery with enhanced query text
    """
    if not questions or not answers:
        return RefinedQuery(
            original_query=original_query,
            refined_query=original_query,
            context_summary="No clarifications provided",
            answers={},
        )

    # Build Q&A context
    qa_pairs = []
    answer_dict = {}
    for q, a in zip(questions, answers, strict=False):
        if a and a.strip():  # Only include non-empty answers
            qa_pairs.append(f"Q: {q.question}\nA: {a}")
            answer_dict[q.question] = a

    if not qa_pairs:
        return RefinedQuery(
            original_query=original_query,
            refined_query=original_query,
            context_summary="No clarifications provided",
            answers={},
        )

    context_text = "\n\n".join(qa_pairs)

    client = genai.Client(api_key=get_api_key())

    refine_prompt = f"""\
Given this original research query and the user's clarifying answers, \
create an enhanced research query that incorporates the context.

## Original Query
{original_query}

## Clarifications
{context_text}

## Instructions
1. Create a refined query that naturally incorporates the clarification context
2. Keep the refined query concise but comprehensive
3. Don't lose any important details from the original query
4. Also provide a brief summary of the key clarifications

Return JSON:
{{
    "refined_query": "The enhanced research query",
    "context_summary": "Brief summary of clarifications provided"
}}
"""

    try:
        response = await client.aio.models.generate_content(
            model=CLARIFIER_MODEL,
            contents=refine_prompt,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        response_text = response.text
        if response_text is None:
            raise ValueError("No response text from refiner")
        result = json.loads(response_text.strip())

        return RefinedQuery(
            original_query=original_query,
            refined_query=result.get("refined_query", original_query),
            context_summary=result.get("context_summary", ""),
            answers=answer_dict,
        )

    except Exception as e:
        logger.warning("Failed to refine query with LLM, using fallback: %s", e)
        # Fallback: simple concatenation
        qa_summaries = [
            f"{q.question}: {a}"
            for q, a in zip(questions, answers, strict=False)
            if a
        ]
        context_summary = "; ".join(qa_summaries)
        refined = f"{original_query}\n\nContext: {context_summary}"

        return RefinedQuery(
            original_query=original_query,
            refined_query=refined,
            context_summary=context_summary,
            answers=answer_dict,
        )


def should_clarify(analysis: QueryAnalysis) -> bool:
    """
    Determine if we should ask clarifying questions based on analysis.

    Args:
        analysis: The QueryAnalysis from analyze_query()

    Returns:
        True if clarification would improve research quality
    """
    # High confidence = clear query, no need to clarify
    if analysis.confidence >= CONFIDENCE_THRESHOLD:
        return False

    # Need at least some questions to ask
    if not analysis.questions:
        return False

    # The analysis explicitly says clarification is needed
    return analysis.needs_clarification
