"""Phase 10: grounded Gemini market-research sidecar.

Read-only and informational. It consumes the HMM system's state and external
web evidence; it cannot place orders or change strategy/risk/HMM settings.
"""

from research.config import ResearchConfig, load_research_config
from research.orchestrator import ResearchOrchestrator
from research.schemas import ResearchOutcome, ResearchRequest, ResearchResult

__all__ = [
    "ResearchConfig", "ResearchOrchestrator", "ResearchOutcome", "ResearchRequest",
    "ResearchResult", "load_research_config",
]
