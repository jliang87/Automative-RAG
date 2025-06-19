"""
UI Components Module

This module contains reusable UI components for the Streamlit interface.
"""

# Import the new unified validation display
from .validation_display import (
    render_unified_validation_display,
    render_quick_validation_badge,
    render_validation_help,
    render_real_time_validation_feedback
)

# Import other existing components
try:
    from .components import (
        header,
        api_status_indicator,
        metadata_filters,
        display_document,
        loading_spinner,
        job_chain_status_card,
        worker_health_indicator,
        safe_dataframe_display,
        clean_dataframe_for_display,
        safe_queue_display
    )
except ImportError:
    # In case components.py doesn't exist or has import issues
    pass

# Import other component modules if they exist
try:
    from .contextual_help import ContextualHelpSystem, render_contextual_help
except ImportError:
    pass

try:
    from .query_history import QueryHistoryManager, render_query_history_insights
except ImportError:
    pass

try:
    from .query_refinement import QueryRefinementAssistant, render_query_refinement
except ImportError:
    pass

try:
    from .query_templates import QueryTemplateSystem, render_query_templates
except ImportError:
    pass

try:
    from .result_quality import ResultQualityIndicator, render_result_quality_indicator
except ImportError:
    pass

try:
    from .smart_suggestions import SmartQuerySuggester, render_smart_suggestions
except ImportError:
    pass

try:
    from .usage_analytics import UsageAnalytics, render_satisfaction_feedback
except ImportError:
    pass

# Export all available components
__all__ = [
    # Validation display (new unified system)
    "render_unified_validation_display",
    "render_quick_validation_badge",
    "render_validation_help",
    "render_real_time_validation_feedback",

    # Advanced components (if available)
    "ContextualHelpSystem",
    "render_contextual_help",
    "QueryHistoryManager",
    "render_query_history_insights",
    "QueryRefinementAssistant",
    "render_query_refinement",
    "QueryTemplateSystem",
    "render_query_templates",
    "ResultQualityIndicator",
    "render_result_quality_indicator",
    "SmartQuerySuggester",
    "render_smart_suggestions",
    "UsageAnalytics",
    "render_satisfaction_feedback",
]