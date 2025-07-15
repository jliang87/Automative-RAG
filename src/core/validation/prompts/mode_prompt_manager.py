"""
Mode-Specific Prompt Manager
Provides specialized prompts for different query modes and validation phases
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_pre_validation_prompt(query_mode: str, context) -> str:
    """Get pre-validation prompt for specific query mode."""

    mode_prompts = {
        "facts": _get_facts_pre_validation_prompt,
        "features": _get_features_pre_validation_prompt,
        "tradeoffs": _get_tradeoffs_pre_validation_prompt,
        "scenarios": _get_scenarios_pre_validation_prompt,
        "debate": _get_debate_pre_validation_prompt,
        "quotes": _get_quotes_pre_validation_prompt
    }

    prompt_function = mode_prompts.get(query_mode, _get_facts_pre_validation_prompt)
    return prompt_function(context)


def get_post_validation_prompt(query_mode: str, context, generated_response: str) -> str:
    """Get post-validation prompt for specific query mode."""

    mode_prompts = {
        "facts": _get_facts_post_validation_prompt,
        "features": _get_features_post_validation_prompt,
        "tradeoffs": _get_tradeoffs_post_validation_prompt,
        "scenarios": _get_scenarios_post_validation_prompt,
        "debate": _get_debate_post_validation_prompt,
        "quotes": _get_quotes_post_validation_prompt
    }

    prompt_function = mode_prompts.get(query_mode, _get_facts_post_validation_prompt)
    return prompt_function(context, generated_response)


# FACTS MODE PROMPTS
def _get_facts_pre_validation_prompt(context) -> str:
    """Pre-validation prompt for facts mode (车辆规格查询)."""

    documents_summary = _format_documents_for_prompt(context.documents)

    return f"""
FACTS MODE PRE-VALIDATION PHASE

TASK: Validate documents and context for automotive specification accuracy.

QUERY: {context.query_text}
QUERY MODE: Facts (车辆规格查询) - Specification Verification

DOCUMENTS RETRIEVED:
{documents_summary}

VALIDATION CHECKS TO PERFORM:
1. Document Relevance for Specs: Do documents contain relevant automotive specifications?
2. Technical Completeness: Sufficient technical detail for accurate spec extraction?
3. Source Expertise: Are sources qualified to provide automotive specifications?
4. Specification Consistency: Are technical claims consistent across sources?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "facts_pre_validation",
  "results": {{
    "document_relevance_for_specs": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "automotive_specs_found": ["list of specs found"]
    }},
    "technical_completeness": {{
      "status": "PASS|WARN|FAIL", 
      "confidence": 0.0-1.0,
      "issues": [],
      "missing_technical_details": []
    }},
    "source_expertise_assessment": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "expert_sources": [],
      "questionable_sources": []
    }},
    "specification_consistency": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "consistent_specs": [],
      "conflicting_specs": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "proceed_to_inference": true|false,
    "meta_validation_needed": {{
      "auto_fetch": [],
      "user_guidance": []
    }}
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


def _get_facts_post_validation_prompt(context, generated_response: str) -> str:
    """Post-validation prompt for facts mode."""

    documents_summary = _format_documents_for_prompt(context.documents)

    return f"""
FACTS MODE POST-VALIDATION PHASE

TASK: Verify accuracy of automotive specifications in generated response.

ORIGINAL QUERY: {context.query_text}

DOCUMENTS USED:
{documents_summary}

GENERATED RESPONSE TO VALIDATE:
{generated_response}

VALIDATION CHECKS TO PERFORM:
1. Specification Accuracy: Are all automotive specs accurately extracted from documents?
2. Citation Verification: Are specifications properly cited to correct sources?
3. Technical Precision: Are technical terms and values precisely stated?
4. No Hallucination Check: Any fabricated or unsupported specifications?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "facts_post_validation",
  "results": {{
    "specification_accuracy": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "accurate_specs": [],
      "inaccurate_specs": [],
      "unsupported_specs": []
    }},
    "citation_verification": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "correct_citations": [],
      "incorrect_citations": [],
      "missing_citations": []
    }},
    "technical_precision": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "precise_statements": [],
      "imprecise_statements": [],
      "unit_errors": []
    }},
    "no_hallucination_check": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "verified_claims": [],
      "fabricated_claims": [],
      "unsupported_claims": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "critical_issues": [],
    "corrections_needed": [],
    "final_recommendation": "APPROVE|REVISE|REJECT"
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


# FEATURES MODE PROMPTS
def _get_features_pre_validation_prompt(context) -> str:
    """Pre-validation prompt for features mode (功能分析)."""

    documents_summary = _format_documents_for_prompt(context.documents)

    return f"""
FEATURES MODE PRE-VALIDATION PHASE

TASK: Validate documents for automotive feature comparison analysis.

QUERY: {context.query_text}
QUERY MODE: Features (功能分析) - Feature Comparison

DOCUMENTS RETRIEVED:
{documents_summary}

VALIDATION CHECKS TO PERFORM:
1. Feature Coverage: Do documents cover the automotive features being compared?
2. Comparison Feasibility: Sufficient information for meaningful feature comparison?
3. Source Balance: Are multiple vehicles/brands represented fairly?
4. Feature Detail Depth: Adequate detail for feature analysis?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "features_pre_validation",
  "results": {{
    "feature_coverage": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "features_covered": [],
      "missing_features": []
    }},
    "comparison_feasibility": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "comparable_vehicles": [],
      "comparison_limitations": []
    }},
    "source_balance": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "vehicles_represented": [],
      "bias_concerns": []
    }},
    "feature_detail_depth": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "detailed_features": [],
      "surface_level_features": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "proceed_to_inference": true|false,
    "meta_validation_needed": {{
      "auto_fetch": [],
      "user_guidance": []
    }}
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


def _get_features_post_validation_prompt(context, generated_response: str) -> str:
    """Post-validation prompt for features mode."""

    return f"""
FEATURES MODE POST-VALIDATION PHASE

TASK: Verify feature comparison accuracy and balance.

ORIGINAL QUERY: {context.query_text}

GENERATED RESPONSE TO VALIDATE:
{generated_response}

VALIDATION CHECKS TO PERFORM:
1. Feature Comparison Accuracy: Are feature comparisons accurate?
2. Balance and Fairness: Is the comparison balanced across vehicles?
3. Completeness: Are important features covered adequately?
4. Evidence Support: Are comparisons supported by document evidence?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "features_post_validation",
  "results": {{
    "comparison_accuracy": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "accurate_comparisons": [],
      "inaccurate_comparisons": []
    }},
    "balance_and_fairness": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "balanced_aspects": [],
      "biased_aspects": []
    }},
    "completeness": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "covered_features": [],
      "missing_important_features": []
    }},
    "evidence_support": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "well_supported_claims": [],
      "unsupported_claims": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "critical_issues": [],
    "corrections_needed": [],
    "final_recommendation": "APPROVE|REVISE|REJECT"
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


# TRADEOFFS MODE PROMPTS
def _get_tradeoffs_pre_validation_prompt(context) -> str:
    """Pre-validation prompt for tradeoffs mode."""

    documents_summary = _format_documents_for_prompt(context.documents)

    return f"""
TRADEOFFS MODE PRE-VALIDATION PHASE

TASK: Validate documents for automotive trade-off analysis.

QUERY: {context.query_text}
QUERY MODE: Tradeoffs - Trade-off Analysis

DOCUMENTS RETRIEVED:
{documents_summary}

VALIDATION CHECKS TO PERFORM:
1. Trade-off Dimensions: Are relevant trade-off aspects covered (performance vs efficiency, cost vs features, etc.)?
2. Multi-perspective Data: Multiple viewpoints on trade-offs available?
3. Quantitative Support: Numerical data to support trade-off analysis?
4. Real-world Context: Practical implications of trade-offs discussed?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "tradeoffs_pre_validation",
  "results": {{
    "tradeoff_dimensions": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "dimensions_covered": [],
      "missing_dimensions": []
    }},
    "multi_perspective_data": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "perspectives_available": [],
      "limited_perspectives": []
    }},
    "quantitative_support": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "numerical_data_available": [],
      "missing_quantitative_data": []
    }},
    "real_world_context": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "practical_implications": [],
      "theoretical_only": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "proceed_to_inference": true|false,
    "meta_validation_needed": {{
      "auto_fetch": [],
      "user_guidance": []
    }}
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


def _get_tradeoffs_post_validation_prompt(context, generated_response: str) -> str:
    """Post-validation prompt for tradeoffs mode."""

    return f"""
TRADEOFFS MODE POST-VALIDATION PHASE

TASK: Verify trade-off analysis accuracy and completeness.

ORIGINAL QUERY: {context.query_text}

GENERATED RESPONSE TO VALIDATE:
{generated_response}

VALIDATION CHECKS TO PERFORM:
1. Trade-off Logic: Are trade-offs logically sound and well-reasoned?
2. Evidence-based Claims: Are trade-off assertions supported by evidence?
3. Balanced Perspective: Are both sides of trade-offs fairly presented?
4. Practical Relevance: Are trade-offs relevant to real-world decisions?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "tradeoffs_post_validation",
  "results": {{
    "tradeoff_logic": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "logical_tradeoffs": [],
      "flawed_logic": []
    }},
    "evidence_based_claims": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "well_supported_claims": [],
      "unsupported_claims": []
    }},
    "balanced_perspective": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "balanced_aspects": [],
      "one_sided_aspects": []
    }},
    "practical_relevance": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "relevant_tradeoffs": [],
      "impractical_considerations": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "critical_issues": [],
    "corrections_needed": [],
    "final_recommendation": "APPROVE|REVISE|REJECT"
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


# SCENARIOS MODE PROMPTS
def _get_scenarios_pre_validation_prompt(context) -> str:
    """Pre-validation prompt for scenarios mode."""

    documents_summary = _format_documents_for_prompt(context.documents)

    return f"""
SCENARIOS MODE PRE-VALIDATION PHASE

TASK: Validate documents for use case scenario analysis.

QUERY: {context.query_text}
QUERY MODE: Scenarios - Use Case Scenario Comparison

DOCUMENTS RETRIEVED:
{documents_summary}

VALIDATION CHECKS TO PERFORM:
1. Scenario Coverage: Are relevant use case scenarios covered?
2. Real-world Applicability: Do scenarios reflect realistic usage patterns?
3. Vehicle Suitability Data: Information about vehicle performance in different scenarios?
4. Context Completeness: Sufficient context for scenario-based recommendations?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "scenarios_pre_validation",
  "results": {{
    "scenario_coverage": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "scenarios_covered": [],
      "missing_scenarios": []
    }},
    "real_world_applicability": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "realistic_scenarios": [],
      "unrealistic_scenarios": []
    }},
    "vehicle_suitability_data": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "suitability_info_available": [],
      "missing_suitability_data": []
    }},
    "context_completeness": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "complete_contexts": [],
      "incomplete_contexts": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "proceed_to_inference": true|false,
    "meta_validation_needed": {{
      "auto_fetch": [],
      "user_guidance": []
    }}
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


def _get_scenarios_post_validation_prompt(context, generated_response: str) -> str:
    """Post-validation prompt for scenarios mode."""

    return f"""
SCENARIOS MODE POST-VALIDATION PHASE

TASK: Verify scenario analysis accuracy and practicality.

ORIGINAL QUERY: {context.query_text}

GENERATED RESPONSE TO VALIDATE:
{generated_response}

VALIDATION CHECKS TO PERFORM:
1. Scenario Realism: Are described scenarios realistic and relevant?
2. Vehicle-Scenario Matching: Are vehicle recommendations appropriate for scenarios?
3. Practical Considerations: Are real-world factors considered?
4. Completeness: Are important scenarios adequately addressed?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "scenarios_post_validation",
  "results": {{
    "scenario_realism": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "realistic_scenarios": [],
      "unrealistic_scenarios": []
    }},
    "vehicle_scenario_matching": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "appropriate_matches": [],
      "inappropriate_matches": []
    }},
    "practical_considerations": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "practical_factors": [],
      "missing_considerations": []
    }},
    "completeness": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "addressed_scenarios": [],
      "missed_scenarios": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "critical_issues": [],
    "corrections_needed": [],
    "final_recommendation": "APPROVE|REVISE|REJECT"
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


# DEBATE MODE PROMPTS
def _get_debate_pre_validation_prompt(context) -> str:
    """Pre-validation prompt for debate mode."""

    documents_summary = _format_documents_for_prompt(context.documents)

    return f"""
DEBATE MODE PRE-VALIDATION PHASE

TASK: Validate documents for expert debate analysis.

QUERY: {context.query_text}
QUERY MODE: Debate - Expert Debate Analysis

DOCUMENTS RETRIEVED:
{documents_summary}

VALIDATION CHECKS TO PERFORM:
1. Expert Opinion Diversity: Multiple expert perspectives available?
2. Argument Quality: Well-reasoned arguments from different sides?
3. Evidence Base: Strong evidence supporting different positions?
4. Debate Balance: Fair representation of different viewpoints?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "debate_pre_validation",
  "results": {{
    "expert_opinion_diversity": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "expert_perspectives": [],
      "missing_perspectives": []
    }},
    "argument_quality": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "strong_arguments": [],
      "weak_arguments": []
    }},
    "evidence_base": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "well_evidenced_positions": [],
      "poorly_evidenced_positions": []
    }},
    "debate_balance": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "balanced_aspects": [],
      "imbalanced_aspects": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "proceed_to_inference": true|false,
    "meta_validation_needed": {{
      "auto_fetch": [],
      "user_guidance": []
    }}
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


def _get_debate_post_validation_prompt(context, generated_response: str) -> str:
    """Post-validation prompt for debate mode."""

    return f"""
DEBATE MODE POST-VALIDATION PHASE

TASK: Verify expert debate analysis fairness and accuracy.

ORIGINAL QUERY: {context.query_text}

GENERATED RESPONSE TO VALIDATE:
{generated_response}

VALIDATION CHECKS TO PERFORM:
1. Perspective Balance: Are different expert perspectives fairly represented?
2. Argument Accuracy: Are expert arguments accurately presented?
3. Evidence Attribution: Are claims properly attributed to sources?
4. Neutrality: Is the analysis neutral and unbiased?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "debate_post_validation",
  "results": {{
    "perspective_balance": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "balanced_perspectives": [],
      "underrepresented_perspectives": []
    }},
    "argument_accuracy": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "accurate_arguments": [],
      "misrepresented_arguments": []
    }},
    "evidence_attribution": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "properly_attributed": [],
      "misattributed": []
    }},
    "neutrality": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "neutral_aspects": [],
      "biased_aspects": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "critical_issues": [],
    "corrections_needed": [],
    "final_recommendation": "APPROVE|REVISE|REJECT"
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


# QUOTES MODE PROMPTS
def _get_quotes_pre_validation_prompt(context) -> str:
    """Pre-validation prompt for quotes mode (用户评论)."""

    documents_summary = _format_documents_for_prompt(context.documents)

    return f"""
QUOTES MODE PRE-VALIDATION PHASE

TASK: Validate documents for user experience and quote analysis.

QUERY: {context.query_text}
QUERY MODE: Quotes (用户评论) - User Experience Analysis

DOCUMENTS RETRIEVED:
{documents_summary}

VALIDATION CHECKS TO PERFORM:
1. User Experience Coverage: Genuine user experiences and reviews available?
2. Quote Authenticity: Are user quotes/reviews authentic and verified?
3. Experience Diversity: Range of different user experiences represented?
4. Context Adequacy: Sufficient context about user backgrounds and use cases?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "quotes_pre_validation",
  "results": {{
    "user_experience_coverage": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "experience_types_covered": [],
      "missing_experience_types": []
    }},
    "quote_authenticity": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "authentic_sources": [],
      "questionable_sources": []
    }},
    "experience_diversity": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "diverse_perspectives": [],
      "limited_perspectives": []
    }},
    "context_adequacy": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "issues": [],
      "well_contextualized": [],
      "lacking_context": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "proceed_to_inference": true|false,
    "meta_validation_needed": {{
      "auto_fetch": [],
      "user_guidance": []
    }}
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


def _get_quotes_post_validation_prompt(context, generated_response: str) -> str:
    """Post-validation prompt for quotes mode."""

    return f"""
QUOTES MODE POST-VALIDATION PHASE

TASK: Verify user experience analysis accuracy and authenticity.

ORIGINAL QUERY: {context.query_text}

GENERATED RESPONSE TO VALIDATE:
{generated_response}

VALIDATION CHECKS TO PERFORM:
1. Quote Accuracy: Are user quotes accurately represented?
2. Attribution Correctness: Are quotes properly attributed to sources?
3. Context Preservation: Is user context preserved and relevant?
4. Representative Balance: Do quotes represent diverse user experiences?

RESPOND WITH EXACTLY THIS JSON STRUCTURE:
```json
{{
  "validation_phase": "quotes_post_validation",
  "results": {{
    "quote_accuracy": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "accurate_quotes": [],
      "misrepresented_quotes": []
    }},
    "attribution_correctness": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "correct_attributions": [],
      "incorrect_attributions": []
    }},
    "context_preservation": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "preserved_contexts": [],
      "lost_contexts": []
    }},
    "representative_balance": {{
      "status": "PASS|WARN|FAIL",
      "confidence": 0.0-1.0,
      "balanced_representation": [],
      "skewed_representation": []
    }}
  }},
  "overall_assessment": {{
    "status": "PASS|WARN|FAIL",
    "overall_confidence": 0.0-1.0,
    "critical_issues": [],
    "corrections_needed": [],
    "final_recommendation": "APPROVE|REVISE|REJECT"
  }}
}}
```

CRITICAL: Respond ONLY with valid JSON. Do not include markdown code blocks or other text.
"""


# HELPER FUNCTIONS
def _format_documents_for_prompt(documents) -> str:
    """Format documents for inclusion in validation prompts."""

    if not documents:
        return "No documents available"

    formatted_docs = []
    for i, doc in enumerate(documents[:5], 1):  # Limit to first 5 documents
        metadata = doc.get('metadata', {})
        content_preview = doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get(
            'content', '')

        doc_summary = f"""
Document {i}:
- Source: {metadata.get('url', metadata.get('source', 'Unknown'))}
- Type: {metadata.get('source_type', metadata.get('sourcePlatform', 'Unknown'))}
- Content Preview: {content_preview}
"""
        formatted_docs.append(doc_summary)

    if len(documents) > 5:
        formatted_docs.append(f"... and {len(documents) - 5} more documents")

    return "\n".join(formatted_docs)


def get_failure_guidance_prompt(failed_step: str, failure_context: Dict[str, Any]) -> str:
    """Generate user_guidance prompt for validation failures."""

    return f"""
VALIDATION FAILURE GUIDANCE

FAILED STEP: {failed_step}
FAILURE CONTEXT: {failure_context}

TASK: Generate user-friendly user_guidance for resolving this validation failure.

Provide user_guidance in this format:
1. What went wrong (simple explanation)
2. What's needed to fix it
3. Specific suggestions for user action
4. Expected improvement from user contribution

Keep user_guidance concise and actionable.
"""