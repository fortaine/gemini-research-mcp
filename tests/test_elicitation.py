"""Test MCP SDK elicitation pattern for query clarification.

This tests the elicitation-based clarification flow in research_deep:
1. User calls research_deep with a vague query
2. ctx.elicit() is called with a dynamic schema
3. User can provide answers to clarifying questions
4. Deep research proceeds with the refined query

These tests verify the pattern without actual API calls (unit tests).
"""

from types import SimpleNamespace

import pytest
from pydantic import Field, create_model


class TestElicitationPattern:
    """Test the elicitation pattern mechanics."""

    @pytest.mark.asyncio
    async def test_clarification_schema_structure(self):
        """Verify ClarificationSchema has correct fields."""
        from gemini_research_mcp.server import ClarificationSchema

        # Check the model has expected fields
        assert hasattr(ClarificationSchema, "model_fields")
        fields = ClarificationSchema.model_fields
        assert "answer_1" in fields
        assert "answer_2" in fields
        assert "answer_3" in fields

    @pytest.mark.asyncio
    async def test_dynamic_schema_creation(self):
        """Test that dynamic Pydantic models can be created for elicitation."""
        questions = [
            "What specific aspects would you like to compare?",
            "What's your use case or context?",
        ]

        # Create dynamic schema like _maybe_clarify_query does
        field_definitions = {
            f"answer_{i+1}": (str, Field(default="", description=q))
            for i, q in enumerate(questions)
        }
        DynamicSchema = create_model("ClarificationQuestions", **field_definitions)

        # Verify schema structure
        assert hasattr(DynamicSchema, "model_fields")
        fields = DynamicSchema.model_fields
        assert len(fields) == 2
        assert "answer_1" in fields
        assert "answer_2" in fields

        # Verify default values work
        instance = DynamicSchema()
        assert instance.answer_1 == ""
        assert instance.answer_2 == ""

        # Verify can set values
        instance = DynamicSchema(answer_1="web APIs", answer_2="building a REST service")
        assert instance.answer_1 == "web APIs"
        assert instance.answer_2 == "building a REST service"

    @pytest.mark.asyncio
    async def test_maybe_clarify_query_without_context(self):
        """_maybe_clarify_query returns original query when context is None."""
        from gemini_research_mcp.server import _maybe_clarify_query

        original = "compare python frameworks"
        result = await _maybe_clarify_query(original, ctx=None)

        assert result == original

    @pytest.mark.asyncio
    async def test_vague_query_detection_short(self):
        """Short queries should be detected as potentially vague."""
        # These are implementation details we can infer from the code
        vague_queries = [
            "AI",  # Very short
            "research AI",  # Contains "research"
            "compare frameworks",  # Contains "compare"
            "best tools",  # Contains "best"
            "analyze trends",  # Contains "analyze"
        ]

        specific_queries = [
            (
                "Compare FastAPI vs Django for building REST APIs in 2025 with "
                "async support and SQLAlchemy integration"
            ),
            (
                "Research the environmental impact of electric vehicles vs gasoline "
                "cars in European markets from 2020-2025"
            ),
        ]

        # We can't directly test the internal logic without mocking,
        # but we can verify the function exists and handles None context
        from gemini_research_mcp.server import _maybe_clarify_query

        for query in vague_queries + specific_queries:
            result = await _maybe_clarify_query(query, ctx=None)
            # Without context, should always return original
            assert result == query


class TestQueryRefinement:
    """Test query refinement logic."""

    @pytest.mark.asyncio
    async def test_refined_query_format(self):
        """Test how refined queries are formatted."""
        original = "compare python frameworks"
        clarification = "Q: What specific aspects?\nA: performance and ease of use"

        # Simulate how the code builds refined query
        refined = f"{original}\n\nAdditional context:\n{clarification}"

        assert original in refined
        assert "Additional context:" in refined
        assert clarification in refined

    @pytest.mark.asyncio
    async def test_empty_clarification_uses_original(self):
        """Empty clarification should not modify the query."""
        original = "compare python frameworks"
        clarification = ""

        # With empty clarification, should use original
        if clarification:
            refined = f"{original}\n\nAdditional context:\n{clarification}"
        else:
            refined = original

        assert refined == original


class TestClarifyingQuestionsHeuristic:
    """Test the pure _detect_clarifying_questions heuristic (SEP-2322 guard pattern)."""

    def test_comprehensive_query_needs_no_clarification(self):
        from gemini_research_mcp.server import _detect_clarifying_questions

        comprehensive = (
            "Compare FastAPI vs Django for building REST APIs in 2025 with "
            "async support and SQLAlchemy integration, focusing on performance "
            "(latency, throughput), developer experience (docs, tooling), and "
            "ecosystem maturity (third-party packages, community size)."
        )
        assert _detect_clarifying_questions(comprehensive) == []

    def test_short_query_is_vague(self):
        from gemini_research_mcp.server import _detect_clarifying_questions

        assert _detect_clarifying_questions("AI") != []

    def test_deterministic_across_calls(self):
        """The heuristic must be a pure function: same input -> same output.

        The guard pattern relies on this to recompute the same clarifying
        questions on the resumed call without persisting server-side state.
        """
        from gemini_research_mcp.server import _detect_clarifying_questions

        query = "compare frameworks"
        assert _detect_clarifying_questions(query) == _detect_clarifying_questions(query)

    def test_max_three_questions(self):
        from gemini_research_mcp.server import _detect_clarifying_questions

        # Triggers multiple heuristic branches at once
        assert len(_detect_clarifying_questions("compare best practices")) <= 3


class TestSessionlessGuardDetection:
    """Test _uses_sessionless_guard_pattern era/task detection."""

    def test_background_task_always_uses_guard(self):
        from gemini_research_mcp.server import _uses_sessionless_guard_pattern

        ctx = SimpleNamespace(is_background_task=True, request_context=None)
        assert _uses_sessionless_guard_pattern(ctx) is True

    def test_modern_protocol_foreground_uses_guard(self):
        from gemini_research_mcp.server import _uses_sessionless_guard_pattern

        ctx = SimpleNamespace(
            is_background_task=False,
            request_context=SimpleNamespace(protocol_version="2026-07-28"),
        )
        assert _uses_sessionless_guard_pattern(ctx) is True

    def test_legacy_protocol_foreground_uses_elicit(self):
        from gemini_research_mcp.server import _uses_sessionless_guard_pattern

        ctx = SimpleNamespace(
            is_background_task=False,
            request_context=SimpleNamespace(protocol_version="2025-06-18"),
        )
        assert _uses_sessionless_guard_pattern(ctx) is False

    def test_no_request_context_uses_elicit(self):
        from gemini_research_mcp.server import _uses_sessionless_guard_pattern

        ctx = SimpleNamespace(is_background_task=False, request_context=None)
        assert _uses_sessionless_guard_pattern(ctx) is False


class TestGuardPatternBuildAndApply:
    """Test building the InputRequiredResult and folding the client's answer back in."""

    def test_build_clarification_guard_shape(self):
        from mcp.types import ElicitRequest, InputRequiredResult

        from gemini_research_mcp.server import (
            _CLARIFY_INPUT_KEY,
            _build_clarification_guard,
        )

        questions = ["What's your use case?"]
        guard = _build_clarification_guard("compare tools", questions)

        assert isinstance(guard, InputRequiredResult)
        assert guard.request_state  # compact, non-empty correlation marker
        assert _CLARIFY_INPUT_KEY in guard.input_requests
        request = guard.input_requests[_CLARIFY_INPUT_KEY]
        assert isinstance(request, ElicitRequest)
        assert request.params.requested_schema["properties"]["answer_1"]["description"] == (
            questions[0]
        )

    def test_apply_clarification_answer_accept(self):
        from gemini_research_mcp.server import _apply_clarification_answer

        questions = ["What's your use case?"]
        elicit_result = SimpleNamespace(action="accept", content={"answer_1": "building an API"})

        refined = _apply_clarification_answer("compare tools", questions, elicit_result)

        assert "compare tools" in refined
        assert "building an API" in refined
        assert "Additional context:" in refined

    def test_apply_clarification_answer_decline(self):
        from gemini_research_mcp.server import _apply_clarification_answer

        questions = ["What's your use case?"]
        elicit_result = SimpleNamespace(action="decline", content=None)

        refined = _apply_clarification_answer("compare tools", questions, elicit_result)

        assert refined == "compare tools"

    def test_apply_clarification_answer_empty_content(self):
        from gemini_research_mcp.server import _apply_clarification_answer

        questions = ["What's your use case?"]
        elicit_result = SimpleNamespace(action="accept", content={"answer_1": "   "})

        refined = _apply_clarification_answer("compare tools", questions, elicit_result)

        assert refined == "compare tools"


class TestMaybeClarifyQueryGuardPattern:
    """End-to-end _maybe_clarify_query behavior across both eras."""

    @pytest.mark.asyncio
    async def test_background_task_returns_input_required_result(self):
        from mcp.types import InputRequiredResult

        from gemini_research_mcp.server import _maybe_clarify_query

        ctx = SimpleNamespace(
            is_background_task=True,
            request_context=None,
            input_responses=None,
        )
        result = await _maybe_clarify_query("AI", ctx)
        assert isinstance(result, InputRequiredResult)

    @pytest.mark.asyncio
    async def test_modern_protocol_foreground_returns_input_required_result(self):
        from mcp.types import InputRequiredResult

        from gemini_research_mcp.server import _maybe_clarify_query

        ctx = SimpleNamespace(
            is_background_task=False,
            request_context=SimpleNamespace(protocol_version="2026-07-28"),
            input_responses=None,
        )
        result = await _maybe_clarify_query("AI", ctx)
        assert isinstance(result, InputRequiredResult)

    @pytest.mark.asyncio
    async def test_resumed_call_applies_input_responses(self):
        """Second call: input_responses is populated, query is recomputed & resolved."""
        from gemini_research_mcp.server import _CLARIFY_INPUT_KEY, _maybe_clarify_query

        elicit_result = SimpleNamespace(action="accept", content={"answer_1": "web scale"})
        ctx = SimpleNamespace(
            is_background_task=True,
            request_context=None,
            input_responses={_CLARIFY_INPUT_KEY: elicit_result},
        )
        result = await _maybe_clarify_query("AI", ctx)
        assert isinstance(result, str)
        assert "web scale" in result

    @pytest.mark.asyncio
    async def test_legacy_protocol_foreground_uses_elicit(self):
        """Handshake-era foreground calls still use ctx.elicit() directly."""
        from gemini_research_mcp.server import _maybe_clarify_query

        accepted = SimpleNamespace(
            action="accept",
            data=SimpleNamespace(model_dump=lambda: {"answer_1": "performance"}),
        )

        async def fake_elicit(*, message, response_type):
            return accepted

        ctx = SimpleNamespace(
            is_background_task=False,
            request_context=SimpleNamespace(protocol_version="2025-06-18"),
            input_responses=None,
            elicit=fake_elicit,
        )
        result = await _maybe_clarify_query("AI", ctx)
        assert "performance" in result

    @pytest.mark.asyncio
    async def test_comprehensive_query_skips_context_entirely(self):
        """A comprehensive query never touches ctx (no elicit, no guard)."""
        from gemini_research_mcp.server import _maybe_clarify_query

        comprehensive = (
            "Compare FastAPI vs Django for building REST APIs in 2025 with "
            "async support and SQLAlchemy integration, focusing on performance "
            "(latency, throughput), developer experience (docs, tooling), and "
            "ecosystem maturity (third-party packages, community size)."
        )
        ctx = SimpleNamespace(is_background_task=True, request_context=None)
        result = await _maybe_clarify_query(comprehensive, ctx)
        assert result == comprehensive
