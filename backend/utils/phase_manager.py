"""
Phase Manager Utility
Manages multi-step workflows with explicit context handoffs between phases
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PhaseResult:
    """Represents the result of a completed phase"""
    phase_name: str
    findings: str
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class PhaseManager:
    """
    Manages multi-phase workflows with context reuse

    Key Features:
    - Tracks phase results in conversation context
    - Generates phase handoff prompts
    - Prioritizes conversation memory over file re-processing
    - Supports artifact tracking

    Usage:
        manager = PhaseManager()

        # Phase 1: Analysis
        phase1_prompt = manager.create_initial_phase_prompt(
            phase_name="Data Analysis",
            task="Analyze the uploaded datasets...",
            expected_outputs=["Key statistics", "Outlier identification"]
        )

        # Execute phase 1...
        result1 = "Analysis shows 10 outliers..."
        manager.record_phase_result("Data Analysis", result1, artifacts=["stats.npy"])

        # Phase 2: Visualization (with handoff from Phase 1)
        phase2_prompt = manager.create_handoff_phase_prompt(
            phase_name="Visualization",
            task="Create charts based on Phase 1 findings...",
            expected_outputs=["Charts in temp_charts/"]
        )

        # Execute phase 2...
        result2 = "Generated 5 charts..."
        manager.record_phase_result("Visualization", result2, artifacts=["chart1.png", "chart2.png"])

        # Get summary of all phases
        summary = manager.get_workflow_summary()
    """

    def __init__(self):
        self.phases: List[PhaseResult] = []
        self._current_phase: Optional[str] = None

    def create_initial_phase_prompt(
        self,
        phase_name: str,
        task: str,
        expected_outputs: Optional[List[str]] = None,
        files_as_fallback: bool = True
    ) -> str:
        """
        Create prompt for the first phase in a workflow

        Args:
            phase_name: Name of the phase (e.g., "Data Analysis")
            task: Detailed task description
            expected_outputs: List of expected outputs from this phase
            files_as_fallback: Whether to include files-as-fallback instruction

        Returns:
            Formatted prompt string
        """
        self._current_phase = phase_name

        prompt_parts = [
            f"**PHASE 1: {phase_name.upper()}**\n",
            f"TASK: {task}\n"
        ]

        if expected_outputs:
            prompt_parts.append("\n**Required Outputs:**")
            for i, output in enumerate(expected_outputs, 1):
                prompt_parts.append(f"{i}. {output}")
            prompt_parts.append("")

        if files_as_fallback:
            prompt_parts.append(
                "\n**Note:** If files are attached, use them for your analysis. "
                "Store your findings in conversation memory for use in subsequent phases."
            )

        logger.info(f"[PhaseManager] Created initial phase prompt: {phase_name}")
        return "\n".join(prompt_parts)

    def create_handoff_phase_prompt(
        self,
        phase_name: str,
        task: str,
        expected_outputs: Optional[List[str]] = None,
        files_as_fallback: bool = True,
        reference_phases: Optional[List[str]] = None
    ) -> str:
        """
        Create prompt for a subsequent phase with context handoff

        Args:
            phase_name: Name of the current phase
            task: Detailed task description
            expected_outputs: List of expected outputs
            files_as_fallback: Whether to include files-as-fallback instruction
            reference_phases: Specific phases to reference (defaults to all previous)

        Returns:
            Formatted prompt with context from previous phases
        """
        self._current_phase = phase_name

        # Determine which phases to reference
        if reference_phases:
            ref_phases = [p for p in self.phases if p.phase_name in reference_phases]
        else:
            ref_phases = self.phases

        if not ref_phases:
            logger.warning(f"[PhaseManager] No previous phases to reference for {phase_name}")
            return self.create_initial_phase_prompt(phase_name, task, expected_outputs, files_as_fallback)

        # Build prompt with phase handoff
        phase_num = len(self.phases) + 1
        prompt_parts = [
            f"**PHASE {phase_num}: {phase_name.upper()}**\n",
            f"**PRIORITY: Use your previous phase findings first.**\n"
        ]

        # Summarize previous phases
        prompt_parts.append("**Previous Work Completed:**")
        for prev_phase in ref_phases:
            prompt_parts.append(f"\n• **{prev_phase.phase_name}:**")
            prompt_parts.append(f"  - {prev_phase.findings[:200]}..." if len(prev_phase.findings) > 200 else f"  - {prev_phase.findings}")
            if prev_phase.artifacts:
                prompt_parts.append(f"  - Generated artifacts: {', '.join(prev_phase.artifacts[:5])}")

        prompt_parts.append(f"\n**Current Task:**\n{task}\n")

        if expected_outputs:
            prompt_parts.append("**Required Outputs:**")
            for i, output in enumerate(expected_outputs, 1):
                prompt_parts.append(f"{i}. {output}")
            prompt_parts.append("")

        if files_as_fallback:
            prompt_parts.append(
                "\n**IMPORTANT:** The attached files are ONLY for reference if you need to verify specific values. "
                f"Your primary data source should be what you already calculated in previous phases "
                f"({', '.join(p.phase_name for p in ref_phases)}). "
                "DO NOT re-analyze the raw data files from scratch."
            )

        logger.info(f"[PhaseManager] Created handoff prompt: {phase_name} (references {len(ref_phases)} previous phases)")
        return "\n".join(prompt_parts)

    def record_phase_result(
        self,
        phase_name: str,
        findings: str,
        artifacts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record the result of a completed phase

        Args:
            phase_name: Name of the completed phase
            findings: Summary of findings (will be used in handoff prompts)
            artifacts: List of generated files/artifacts
            metadata: Additional metadata to store
        """
        result = PhaseResult(
            phase_name=phase_name,
            findings=findings,
            artifacts=artifacts or [],
            metadata=metadata or {}
        )
        self.phases.append(result)
        logger.info(f"[PhaseManager] Recorded phase result: {phase_name} ({len(artifacts or [])} artifacts)")

    def get_phase_result(self, phase_name: str) -> Optional[PhaseResult]:
        """Get result for a specific phase"""
        for phase in self.phases:
            if phase.phase_name == phase_name:
                return phase
        return None

    def get_all_artifacts(self) -> List[str]:
        """Get all artifacts generated across all phases"""
        all_artifacts = []
        for phase in self.phases:
            all_artifacts.extend(phase.artifacts)
        return all_artifacts

    def get_workflow_summary(self) -> Dict[str, Any]:
        """
        Get summary of the entire workflow

        Returns:
            Dictionary with workflow statistics and phase summaries
        """
        return {
            "total_phases": len(self.phases),
            "total_artifacts": len(self.get_all_artifacts()),
            "phases": [
                {
                    "name": p.phase_name,
                    "findings_preview": p.findings[:100] + "..." if len(p.findings) > 100 else p.findings,
                    "artifact_count": len(p.artifacts),
                    "timestamp": p.timestamp.isoformat()
                }
                for p in self.phases
            ]
        }

    def reset(self):
        """Reset the phase manager (clear all phase history)"""
        self.phases = []
        self._current_phase = None
        logger.info("[PhaseManager] Reset - cleared all phase history")
