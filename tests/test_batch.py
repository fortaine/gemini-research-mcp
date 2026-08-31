#!/usr/bin/env python3
"""
Batch test script for Gemini Deep Research API.
Runs multiple queries and saves results with timing.

Usage:
    cd .github/skills/deep-research/mcp-server
    uv run python test_batch.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai

# Configuration
DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"
MAX_POLL_TIME = 3600.0  # 60 minutes
STREAM_POLL_INTERVAL = 10.0

# Test queries - variety of complexity
TEST_QUERIES = [
    {
        "id": "simple",
        "query": "What is Model Context Protocol (MCP) and how does it work?",
    },
    {
        "id": "medium",
        "query": (
            "Compare FHIR R4 and OMOP CDM for healthcare data interoperability. "
            "What are the key differences?"
        ),
    },
    {
        "id": "complex",
        "query": (
            "Comprehensive comparison of OMOP CDM vs FHIR R4: data models, "
            "use cases, tooling, and migration strategies"
        ),
    },
]

# Output directory
OUTPUT_DIR = Path(__file__).parent / "test_results"


def log(msg: str, start_time: float) -> None:
    """Print timestamped log message."""
    elapsed = time.time() - start_time
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{elapsed:6.1f}s] {msg}")


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _is_reasoning_content(value: Any) -> bool:
    content_type = _field(value, "type")
    if content_type in {"thought", "thinking", "reasoning"}:
        return True
    return any(
        _field(value, field_name) is True
        for field_name in ("thought", "thinking", "reasoning")
    )


def _extract_interaction_text(interaction: Any) -> str:
    output_text = _field(interaction, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts: list[str] = []
    for step in _sequence(_field(interaction, "steps")):
        if _field(step, "type") != "model_output":
            continue
        for item in _sequence(_field(step, "content")):
            if _is_reasoning_content(item):
                continue
            text = _field(item, "text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
    return "\n\n".join(text_parts)


async def run_single_test(query_info: dict, client: genai.Client, start_time: float) -> dict:
    """Run a single deep research test and return results."""
    query_id = query_info["id"]
    query = query_info["query"]
    
    result = {
        "id": query_id,
        "query": query,
        "start_time": datetime.now().isoformat(),
        "chunks": 0,
        "thoughts": [],
        "text": "",
        "status": None,
        "errors": [],
        "elapsed_seconds": 0,
    }
    
    test_start = time.time()
    text_chunks = []
    thinking_summaries = []
    interaction_id = None
    final_status = None
    errors = []
    chunk_count = 0
    
    log(f"🔬 Starting test '{query_id}': {query[:60]}...", start_time)
    
    try:
        stream = await client.aio.interactions.create(
            input=query,
            agent=DEEP_RESEARCH_AGENT,
            background=True,
            stream=True,
            agent_config={
                "type": "deep-research",
                "thinking_summaries": "auto",
            },
        )
        
        async for event in stream:
            chunk_count += 1
            event_type = getattr(event, 'event_type', type(event).__name__)
            
            if event_type == "interaction.created":
                if hasattr(event, 'interaction'):
                    interaction_id = getattr(event.interaction, 'id', None)
                continue
            
            if event_type == "step.delta":
                delta = getattr(event, 'delta', None)
                if delta:
                    delta_type = getattr(delta, 'type', None)
                    if delta_type == 'thought_summary':
                        content = getattr(delta, 'content', None)
                        if content:
                            thought_text = getattr(content, 'text', None)
                            if thought_text:
                                thinking_summaries.append(thought_text)
                                thought_preview = thought_text[:60]
                                log(
                                    f"   🧠 Thought {len(thinking_summaries)}: "
                                    f"{thought_preview}...",
                                    start_time,
                                )
                    elif delta_type == 'text':
                        text_content = getattr(delta, 'text', None)
                        if not text_content:
                            content = getattr(delta, 'content', None)
                            if content:
                                text_content = getattr(content, 'text', None)
                        if text_content:
                            text_chunks.append(text_content)
                continue
            
            if event_type == "interaction.completed":
                if hasattr(event, 'interaction'):
                    final_status = getattr(event.interaction, 'status', None)
                    output = getattr(event.interaction, 'output', None)
                    if output:
                        for item in output:
                            parts = getattr(item, 'parts', [])
                            for part in parts:
                                part_text = getattr(part, 'text', None)
                                if part_text:
                                    text_chunks.append(part_text)
                continue
            
            if event_type == "error":
                error_obj = getattr(event, 'error', None)
                if error_obj:
                    errors.append(str(error_obj))
                    log(f"   ❌ ERROR: {error_obj}", start_time)
                continue
        
        log(f"   📡 Stream ended: {chunk_count} chunks, status={final_status}", start_time)
        
    except Exception as e:
        log(f"   ❌ Stream exception: {e}", start_time)
        errors.append(str(e))
    
    # Poll if needed
    final_text = "".join(text_chunks)
    
    if interaction_id and (final_status == "in_progress" or not final_text):
        log(f"   ⏳ Polling (status={final_status}, text={len(final_text)} chars)...", start_time)
        
        poll_start = time.time()
        poll_count = 0
        
        while True:
            poll_count += 1
            poll_elapsed = time.time() - poll_start
            
            if poll_elapsed > MAX_POLL_TIME:
                errors.append(f"Polling timeout after {poll_elapsed:.1f}s")
                break
            
            try:
                interaction = await client.aio.interactions.get(interaction_id)
                status = getattr(interaction, 'status', 'unknown')
                
                if poll_count % 6 == 0:  # Log every minute
                    log(f"   🔄 Poll #{poll_count}: status={status}", start_time)
                
                if status == "completed":
                    final_status = status
                    final_text = _extract_interaction_text(interaction)
                    if final_text:
                        log(f"   ✅ Got final text: {len(final_text)} chars", start_time)
                    else:
                        log("   🔍 No output_text or model_output step text", start_time)
                    break
                elif status == "failed":
                    final_status = status
                    error_details = getattr(interaction, 'error', None)
                    if error_details:
                        errors.append(str(error_details))
                    log(f"   ❌ Research failed: {error_details}", start_time)
                    break
                
                await asyncio.sleep(STREAM_POLL_INTERVAL)
                
            except Exception as e:
                log(f"   ⚠️ Poll error: {e}", start_time)
                await asyncio.sleep(STREAM_POLL_INTERVAL)
    
    elapsed = time.time() - test_start
    
    result["chunks"] = chunk_count
    result["thoughts"] = thinking_summaries
    result["text"] = final_text
    result["status"] = final_status
    result["errors"] = errors
    result["elapsed_seconds"] = round(elapsed, 1)
    result["end_time"] = datetime.now().isoformat()
    
    status_emoji = "✅" if final_status == "completed" and final_text else "❌"
    log(
        f"{status_emoji} Test '{query_id}' complete: {elapsed:.1f}s, "
        f"{len(final_text)} chars, status={final_status}",
        start_time,
    )
    
    return result


async def main():
    """Run all tests."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    batch_start = time.time()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*70)
    print("🧪 DEEP RESEARCH BATCH TEST")
    print(f"   Batch ID: {batch_id}")
    print(f"   Tests: {len(TEST_QUERIES)}")
    print(f"   Agent: {DEEP_RESEARCH_AGENT}")
    print("="*70)
    
    results = []
    
    for query_info in TEST_QUERIES:
        result = await run_single_test(query_info, client, batch_start)
        results.append(result)
        
        # Save individual result
        result_file = OUTPUT_DIR / f"{batch_id}_{result['id']}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save text to markdown
        if result["text"]:
            text_file = OUTPUT_DIR / f"{batch_id}_{result['id']}.md"
            with open(text_file, 'w') as f:
                f.write(f"# {result['query']}\n\n")
                f.write(f"**Status**: {result['status']}\n")
                f.write(f"**Elapsed**: {result['elapsed_seconds']}s\n")
                f.write(f"**Thoughts**: {len(result['thoughts'])}\n\n")
                f.write("---\n\n")
                f.write(result["text"])
        
        print()
    
    # Save batch summary
    total_elapsed = time.time() - batch_start
    summary = {
        "batch_id": batch_id,
        "total_tests": len(results),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "tests": [
            {
                "id": r["id"],
                "status": r["status"],
                "elapsed_seconds": r["elapsed_seconds"],
                "text_length": len(r["text"]),
                "thought_count": len(r["thoughts"]),
                "error_count": len(r["errors"]),
            }
            for r in results
        ],
    }
    
    summary_file = OUTPUT_DIR / f"{batch_id}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("="*70)
    print("📊 BATCH SUMMARY")
    print(f"   Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"   Results saved to: {OUTPUT_DIR}/")
    print()
    for r in results:
        status = "✅" if r["status"] == "completed" else "❌"
        print(f"   {status} {r['id']}: {r['elapsed_seconds']}s, {len(r['text'])} chars")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
