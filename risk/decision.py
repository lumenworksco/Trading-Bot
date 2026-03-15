"""V8: Unified risk decision pipeline.

Structured decision type for all risk checks, enabling shadow P&L
analysis on rejected signals.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RiskVerdict(Enum):
    APPROVED = "approved"
    MODIFIED = "modified"   # Approved but size/params changed
    REJECTED = "rejected"


@dataclass
class RiskDecision:
    """Structured result from the risk decision pipeline."""
    verdict: RiskVerdict
    original_qty: int
    approved_qty: int           # May be reduced
    reasons: list[str] = field(default_factory=list)

    # For MODIFIED verdicts
    size_adjustment_factor: float = 1.0
    adjusted_stop: Optional[float] = None
    adjusted_target: Optional[float] = None

    def add_reason(self, reason: str):
        """Append a reason to the decision."""
        self.reasons.append(reason)

    def reject(self, reason: str) -> 'RiskDecision':
        """Mark as rejected with reason."""
        self.verdict = RiskVerdict.REJECTED
        self.approved_qty = 0
        self.add_reason(reason)
        return self

    def modify(self, new_qty: int, reason: str) -> 'RiskDecision':
        """Mark as modified with reduced qty."""
        self.verdict = RiskVerdict.MODIFIED
        self.approved_qty = new_qty
        self.size_adjustment_factor = new_qty / self.original_qty if self.original_qty > 0 else 0
        self.add_reason(reason)
        return self

    @property
    def is_approved(self) -> bool:
        return self.verdict in (RiskVerdict.APPROVED, RiskVerdict.MODIFIED)

    @property
    def reasons_str(self) -> str:
        return "; ".join(self.reasons) if self.reasons else ""

    @classmethod
    def approve(cls, qty: int) -> 'RiskDecision':
        """Create an approved decision."""
        return cls(
            verdict=RiskVerdict.APPROVED,
            original_qty=qty,
            approved_qty=qty,
        )
