"""
Pre-built format instruction templates for research_deep.

These templates guide the Deep Research Agent to produce structured,
professionally formatted reports for common use cases.

Usage:
    from gemini_research_mcp.templates import EXECUTIVE_BRIEFING
    result = await research_deep(query, format_instructions=EXECUTIVE_BRIEFING)

Inspired by ADK Deep Search Agent's structured sectioning approach.
"""

from dataclasses import dataclass
from enum import StrEnum


class TemplateCategory(StrEnum):
    """Categories of format instruction templates."""

    BUSINESS = "business"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    ANALYSIS = "analysis"


@dataclass
class FormatTemplate:
    """A format instruction template with metadata."""

    name: str
    description: str
    category: TemplateCategory
    instructions: str

    def __str__(self) -> str:
        """Return the format instructions for use with research_deep."""
        return self.instructions


# =============================================================================
# Business Templates
# =============================================================================

EXECUTIVE_BRIEFING = FormatTemplate(
    name="Executive Briefing",
    description="Concise C-suite summary with key findings, implications, and recommendations",
    category=TemplateCategory.BUSINESS,
    instructions="""
Structure the report as an Executive Briefing for senior leadership:

## Executive Summary
- 2-3 paragraph overview of the topic and key conclusions
- Lead with the most important insight

## Key Findings
- 5-7 bullet points of the most critical discoveries
- Each finding should be actionable or decision-relevant

## Analysis
- 3-4 focused sections exploring the main themes
- Use subheadings for clarity
- Include relevant data points and statistics

## Strategic Implications
- What does this mean for decision-making?
- Risks and opportunities identified

## Recommendations
- 3-5 prioritized action items
- Include timeframes where appropriate (immediate, short-term, long-term)

## Sources
- Numbered citations with URLs

Keep the tone professional and concise. Avoid jargon where possible.
Target length: 1500-2500 words.
""",
)

COMPETITIVE_ANALYSIS = FormatTemplate(
    name="Competitive Analysis",
    description=(
        "Deep dive comparison of competitors with strengths, "
        "weaknesses, and market positioning"
    ),
    category=TemplateCategory.BUSINESS,
    instructions="""
Structure the report as a Competitive Analysis:

## Market Overview
- Current market landscape and key players
- Market size and growth trends
- Key success factors in this space

## Competitor Profiles
For each major competitor (aim for 3-5), provide:
### [Competitor Name]
- **Overview**: Brief company description
- **Key Products/Services**: What they offer
- **Strengths**: What they do well
- **Weaknesses**: Where they struggle
- **Market Position**: Their positioning and target audience
- **Recent Developments**: Notable news, launches, or changes

## Comparison Matrix
Create a structured comparison covering:
- Pricing
- Features/capabilities
- Target market
- Geographic presence
- Technology/innovation

## Competitive Dynamics
- How do competitors interact?
- Emerging threats or disruptors
- Barriers to entry

## Strategic Recommendations
- Differentiation opportunities
- Competitive gaps to exploit
- Defensive strategies

## Sources
- Numbered citations with URLs

Use tables and bullet points for easy scanning.
Target length: 2000-3500 words.
""",
)

MARKET_RESEARCH = FormatTemplate(
    name="Market Research Report",
    description="Comprehensive market analysis with trends, segments, and forecasts",
    category=TemplateCategory.BUSINESS,
    instructions="""
Structure the report as a Market Research Report:

## Executive Summary
- Key market insights in 3-4 paragraphs
- Critical numbers and trends upfront

## Market Overview
- Market definition and scope
- Current market size (with sources)
- Historical growth patterns

## Market Segmentation
- By product/service type
- By customer segment
- By geography
- By use case/application

## Industry Trends
- Major trends shaping the market
- Technology developments
- Regulatory changes
- Consumer behavior shifts

## Competitive Landscape
- Key players and market shares
- Competitive intensity
- Recent M&A activity

## Market Drivers & Challenges
### Drivers
- Factors accelerating market growth
### Challenges
- Barriers and headwinds

## Future Outlook
- Growth forecasts (with timeframes)
- Emerging opportunities
- Potential disruptions

## Methodology Notes
- Data sources and limitations

## Sources
- Numbered citations with URLs

Include specific data points and statistics wherever available.
Target length: 2500-4000 words.
""",
)


# =============================================================================
# Analysis Templates
# =============================================================================

COMPARISON_TABLE = FormatTemplate(
    name="Comparison Table",
    description="Structured side-by-side comparison with pros/cons and verdict",
    category=TemplateCategory.ANALYSIS,
    instructions="""
Structure the report as a Comparison Analysis:

## Overview
- Brief context on what is being compared and why
- Key criteria for comparison

## Quick Comparison Table
Create a markdown table comparing all options across key dimensions:
| Criteria | Option A | Option B | Option C |
|----------|----------|----------|----------|
| Feature 1 | ... | ... | ... |
| Feature 2 | ... | ... | ... |
(Include 6-10 criteria)

## Detailed Analysis

### [Option A]
**Best for:** [Use case summary]
#### Pros
- Strength 1
- Strength 2
- Strength 3
#### Cons
- Weakness 1
- Weakness 2
#### Key Differentiator
What makes this option unique

(Repeat for each option)

## Head-to-Head Comparisons
- Option A vs Option B: [Key difference]
- Option B vs Option C: [Key difference]
- Option A vs Option C: [Key difference]

## Verdict & Recommendations
- **Best Overall**: [Option] - why
- **Best for Budget**: [Option] - why
- **Best for [Specific Use Case]**: [Option] - why

## Sources
- Numbered citations with URLs

Use emojis sparingly for visual scanning (✅ ❌ ⭐).
Target length: 1500-2500 words.
""",
)

PROS_CONS_ANALYSIS = FormatTemplate(
    name="Pros & Cons Analysis",
    description="Balanced evaluation of advantages and disadvantages with weighted assessment",
    category=TemplateCategory.ANALYSIS,
    instructions="""
Structure the report as a Pros & Cons Analysis:

## Executive Summary
- Topic overview
- Bottom-line assessment (positive, negative, or mixed)

## Background & Context
- What is being evaluated
- Why this analysis matters
- Key stakeholders or affected parties

## Advantages (Pros)

### Major Advantages
1. **[Advantage 1]**
   - Explanation and evidence
   - Who benefits most
2. **[Advantage 2]**
   - Explanation and evidence
   - Who benefits most
(Continue for 3-5 major pros)

### Minor Advantages
- Additional smaller benefits

## Disadvantages (Cons)

### Major Disadvantages
1. **[Disadvantage 1]**
   - Explanation and evidence
   - Impact and severity
2. **[Disadvantage 2]**
   - Explanation and evidence
   - Impact and severity
(Continue for 3-5 major cons)

### Minor Disadvantages
- Additional smaller drawbacks

## Risk Assessment
- Likelihood and impact of downsides
- Mitigation strategies

## Balance Sheet
| Pros | Cons |
|------|------|
| ... | ... |

## Verdict
- Overall assessment
- Under what conditions pros outweigh cons (and vice versa)
- Recommendations for different scenarios

## Sources
- Numbered citations with URLs

Be balanced and evidence-based. Avoid advocacy.
Target length: 1500-2500 words.
""",
)

DEEP_DIVE = FormatTemplate(
    name="Deep Dive",
    description="Comprehensive exploration of a topic with multiple angles and perspectives",
    category=TemplateCategory.ANALYSIS,
    instructions="""
Structure the report as a Deep Dive Investigation:

## Introduction
- Topic overview and scope
- Why this matters now
- Key questions to be answered

## Background
- Historical context
- How we got here
- Key players and stakeholders

## Current State
- Where things stand today
- Recent developments
- Key metrics and data points

## Multiple Perspectives

### Perspective 1: [Stakeholder/Angle A]
- Their view and interests
- Supporting evidence

### Perspective 2: [Stakeholder/Angle B]
- Their view and interests
- Supporting evidence

### Perspective 3: [Stakeholder/Angle C]
- Their view and interests
- Supporting evidence

## Critical Analysis
- What the evidence suggests
- Areas of consensus
- Areas of debate
- Gaps in knowledge

## Future Scenarios
- Scenario A: [Optimistic]
- Scenario B: [Pessimistic]
- Scenario C: [Most likely]

## Key Takeaways
- 5-7 main conclusions
- Implications for different audiences

## Further Reading
- Related topics to explore

## Sources
- Numbered citations with URLs

Be thorough and nuanced. Embrace complexity.
Target length: 3000-5000 words.
""",
)


# =============================================================================
# Technical Templates
# =============================================================================

TECHNICAL_OVERVIEW = FormatTemplate(
    name="Technical Overview",
    description="Technical explanation suitable for engineers and developers",
    category=TemplateCategory.TECHNICAL,
    instructions="""
Structure the report as a Technical Overview:

## TL;DR
- 2-3 sentence summary
- Key technical insight

## Introduction
- What this technology/concept is
- Problem it solves
- Why it matters

## Technical Fundamentals
- Core concepts explained clearly
- Key terminology defined
- How it works at a high level

## Architecture & Design
- System components
- How parts interact
- Design decisions and trade-offs
- Include diagrams described in text if helpful

## Implementation Details
- Technical specifications
- Requirements and dependencies
- Key algorithms or protocols
- Code examples or pseudocode where relevant

## Performance & Scalability
- Benchmarks and metrics (if available)
- Scalability considerations
- Known limitations

## Use Cases
- When to use this
- When NOT to use this
- Real-world examples

## Comparison with Alternatives
- How does this compare to other approaches?
- Trade-offs between options

## Best Practices
- Recommended approaches
- Common pitfalls to avoid

## Resources
- Documentation links
- Tutorials and guides
- Community resources

## Sources
- Numbered citations with URLs

Be precise and technical but accessible. Define jargon.
Target length: 2000-3500 words.
""",
)

API_EVALUATION = FormatTemplate(
    name="API Evaluation",
    description="Technical assessment of an API including DX, capabilities, and limitations",
    category=TemplateCategory.TECHNICAL,
    instructions="""
Structure the report as an API Evaluation:

## Overview
- What the API does
- Provider and pricing model
- Target use cases

## Getting Started
- Authentication methods
- SDK availability
- Quick start complexity

## API Design

### Endpoints & Resources
- Key endpoints and their purposes
- REST, GraphQL, or other paradigms
- Resource naming and structure

### Request/Response Patterns
- Data formats (JSON, etc.)
- Pagination approach
- Error handling

## Developer Experience (DX)

### Documentation Quality
- Completeness
- Examples and tutorials
- API reference clarity

### SDK & Libraries
- Official SDKs available
- Community libraries
- Code generation tools

### Testing & Debugging
- Sandbox/test environment
- Logging and debugging tools
- Rate limiting in development

## Capabilities Assessment
- What it does well
- Feature gaps
- Unique capabilities

## Limitations & Gotchas
- Known issues
- Undocumented behaviors
- Breaking change history

## Performance
- Latency characteristics
- Rate limits
- Reliability and uptime

## Pricing Analysis
- Pricing model
- Cost at scale
- Hidden costs

## Security
- Authentication options
- Data handling
- Compliance certifications

## Verdict
- Overall assessment
- Best for: [use cases]
- Avoid if: [situations]

## Sources
- Numbered citations with URLs

Be specific with technical details. Include version numbers where relevant.
Target length: 2000-3000 words.
""",
)


# =============================================================================
# Academic Templates
# =============================================================================

LITERATURE_REVIEW = FormatTemplate(
    name="Literature Review",
    description="Academic-style synthesis of existing research and publications",
    category=TemplateCategory.ACADEMIC,
    instructions="""
Structure the report as an Academic Literature Review:

## Abstract
- 150-250 word summary
- Scope, methodology, and key findings

## Introduction
- Topic background and significance
- Research questions or objectives
- Scope and limitations of the review

## Methodology
- Search strategy (databases, keywords)
- Inclusion/exclusion criteria
- Time period covered

## Thematic Analysis

### Theme 1: [Major Research Area]
- Key studies and findings
- Consensus and debates
- Methodological approaches

### Theme 2: [Major Research Area]
- Key studies and findings
- Consensus and debates
- Methodological approaches

### Theme 3: [Major Research Area]
- Key studies and findings
- Consensus and debates
- Methodological approaches

## Synthesis & Discussion
- How themes interconnect
- Evolution of thinking over time
- Current state of knowledge

## Research Gaps
- What remains unknown
- Methodological limitations in existing research
- Opportunities for future research

## Conclusions
- Summary of key findings
- Implications for practice
- Recommendations for future research

## References
- Full citations in consistent format (APA, Chicago, etc.)

Use formal academic tone. Be comprehensive and objective.
Target length: 3000-5000 words.
""",
)

RESEARCH_BRIEF = FormatTemplate(
    name="Research Brief",
    description="Concise summary of research findings for non-academic audiences",
    category=TemplateCategory.ACADEMIC,
    instructions="""
Structure the report as a Research Brief:

## Key Messages
- 3-4 bullet points summarizing the most important findings
- Written for a general audience

## Background
- Why this research matters
- Context for non-experts
- 2-3 paragraphs maximum

## What the Research Shows

### Finding 1
- Plain language explanation
- Supporting evidence
- Confidence level

### Finding 2
- Plain language explanation
- Supporting evidence
- Confidence level

### Finding 3
- Plain language explanation
- Supporting evidence
- Confidence level

## What This Means
- Practical implications
- Who is affected
- What actions might follow

## Limitations
- What the research doesn't tell us
- Caveats and uncertainties

## Further Reading
- Key studies for those who want more detail
- Accessible resources

## Sources
- Numbered citations with URLs

Avoid jargon. Keep it accessible and actionable.
Target length: 800-1500 words.
""",
)


# =============================================================================
# Template Registry
# =============================================================================

# All templates for easy access
ALL_TEMPLATES: dict[str, FormatTemplate] = {
    # Business
    "executive_briefing": EXECUTIVE_BRIEFING,
    "competitive_analysis": COMPETITIVE_ANALYSIS,
    "market_research": MARKET_RESEARCH,
    # Analysis
    "comparison_table": COMPARISON_TABLE,
    "pros_cons": PROS_CONS_ANALYSIS,
    "deep_dive": DEEP_DIVE,
    # Technical
    "technical_overview": TECHNICAL_OVERVIEW,
    "api_evaluation": API_EVALUATION,
    # Academic
    "literature_review": LITERATURE_REVIEW,
    "research_brief": RESEARCH_BRIEF,
}

# Category groupings
TEMPLATES_BY_CATEGORY: dict[TemplateCategory, list[FormatTemplate]] = {
    TemplateCategory.BUSINESS: [
        EXECUTIVE_BRIEFING,
        COMPETITIVE_ANALYSIS,
        MARKET_RESEARCH,
    ],
    TemplateCategory.ANALYSIS: [
        COMPARISON_TABLE,
        PROS_CONS_ANALYSIS,
        DEEP_DIVE,
    ],
    TemplateCategory.TECHNICAL: [
        TECHNICAL_OVERVIEW,
        API_EVALUATION,
    ],
    TemplateCategory.ACADEMIC: [
        LITERATURE_REVIEW,
        RESEARCH_BRIEF,
    ],
}


def get_template(name: str) -> FormatTemplate | None:
    """Get a template by name (case-insensitive, supports aliases)."""
    normalized = name.lower().replace(" ", "_").replace("-", "_")
    return ALL_TEMPLATES.get(normalized)


def list_templates() -> list[dict[str, str]]:
    """List all available templates with metadata."""
    return [
        {
            "name": t.name,
            "key": key,
            "description": t.description,
            "category": t.category.value,
        }
        for key, t in ALL_TEMPLATES.items()
    ]
