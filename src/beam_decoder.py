"""Beam-search decoder for fixed-slot symbolic assignments.

The default domain is the current five-symbol equation layout:

    d1 op d2 = d3

Class labels match the existing project convention: digits are 0-9, plus is 10,
minus is 11, and equals is 12. The decoder is independent of the CNN and the
dataset; it only consumes per-position class probabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


DIGIT_LABELS = tuple(range(10))
PLUS_LABEL = 10
MINUS_LABEL = 11
EQUALS_LABEL = 12

DEFAULT_CANDIDATE_SETS = (
    DIGIT_LABELS,
    (PLUS_LABEL, MINUS_LABEL),
    DIGIT_LABELS,
    (EQUALS_LABEL,),
    DIGIT_LABELS,
)


def is_valid_equation(assignment: Sequence[int]) -> bool:
    """Return True when an assignment is a valid single-digit equation.

    Parameters
    ----------
    assignment:
        Five class labels in the order ``d1, op, d2, equals, d3``. Digits use
        labels 0-9, plus uses 10, minus uses 11, and equals uses 12.
    """
    if len(assignment) != 5:
        return False

    d1, op, d2, equals, d3 = (int(value) for value in assignment)
    if d1 not in DIGIT_LABELS or d2 not in DIGIT_LABELS or d3 not in DIGIT_LABELS:
        return False
    if equals != EQUALS_LABEL:
        return False
    if op == PLUS_LABEL:
        return d1 + d2 == d3
    if op == MINUS_LABEL:
        return d1 - d2 == d3
    return False


@dataclass(frozen=True)
class BeamCandidate:
    """One partial or complete assignment in the beam."""

    assignment: tuple[int, ...]
    score: float


class BeamSearchDecoder:
    """Generic beam search over fixed-position class probabilities.

    The decoder is independent of the CNN and dataset. A domain supplies:

    * ``candidate_sets``: allowed labels at each output position.
    * ``constraint_checker``: a predicate for completed assignments.
    * ``probabilities``: per-position class probabilities at decode time.

    The default configuration implements the five-slot arithmetic benchmark,
    but the search logic itself is reusable for other fixed-slot constrained
    prediction tasks.
    """

    def __init__(
        self,
        beam_width: int,
        candidate_sets: Sequence[Sequence[int]] | None = None,
        constraint_checker: Callable[[Sequence[int]], bool] = is_valid_equation,
        epsilon: float = 1e-12,
        recover_final_if_empty: bool = False,
    ) -> None:
        """Create a decoder.

        Parameters
        ----------
        beam_width:
            Number of hypotheses retained after each expansion.
        candidate_sets:
            Allowed labels at each output position. If omitted, the arithmetic
            layout ``digit, operator, digit, equals, digit`` is used.
        constraint_checker:
            Predicate used to filter completed assignments.
        epsilon:
            Lower clipping bound before taking logarithms.
        recover_final_if_empty:
            If true, the decoder can fall back to all last-step expanded
            complete assignments when final beam pruning leaves no valid
            equation. Experiments use false so pruning behavior remains visible.
        """
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.beam_width = int(beam_width)
        self.candidate_sets = tuple(
            tuple(int(label) for label in labels)
            for labels in (candidate_sets or DEFAULT_CANDIDATE_SETS)
        )
        self.constraint_checker = constraint_checker
        self.epsilon = float(epsilon)
        self.recover_final_if_empty = bool(recover_final_if_empty)
        self.trace: list[dict] = []
        self.last_stats: dict = {}
        self._last_complete_expanded: list[BeamCandidate] = []

    def decode(self, probabilities: Sequence[Sequence[float]]) -> tuple[list[int] | None, list[dict]]:
        """Decode probabilities into the best valid assignment and search trace.

        Returns ``(assignment, trace)``. ``assignment`` is ``None`` if no valid
        completed candidate remains after pruning and validity filtering.
        """
        assignment, trace, _ = self.decode_with_stats(probabilities)
        return assignment, trace

    def decode_with_stats(
        self,
        probabilities: Sequence[Sequence[float]],
    ) -> tuple[list[int] | None, list[dict], dict]:
        """Decode probabilities and return instrumentation for this search.

        This method extends :meth:`decode` with aggregate counts used by the
        efficiency experiments, including expanded candidates, validity checks,
        score computations, and wall-clock time for the symbolic decoder.
        """
        start_time = time.perf_counter()
        probability_array = np.asarray(probabilities, dtype=np.float64)
        self._validate_probabilities(probability_array)

        log_probs = np.log(np.clip(probability_array, self.epsilon, 1.0))
        self.trace = []
        self._last_complete_expanded = []
        initial_beam = [BeamCandidate(assignment=(), score=0.0)]
        final_beam = self.search(position=0, beam=initial_beam, log_probs=log_probs)

        elapsed_seconds = time.perf_counter() - start_time
        assignment = None if not final_beam else list(final_beam[0].assignment)
        self.last_stats = self._summarize_trace(elapsed_seconds)
        return assignment, self.trace, self.last_stats

    def search(
        self,
        position: int,
        beam: Sequence[BeamCandidate],
        log_probs: np.ndarray,
    ) -> list[BeamCandidate]:
        """Recursively expand and prune the beam.

        ``position`` indexes the next slot to fill. At each nonterminal step,
        every retained hypothesis is extended by each allowed label at that
        position, scored by adding the corresponding log probability, sorted,
        and pruned to ``beam_width``. At the terminal step, completed candidates
        are filtered by ``constraint_checker``.
        """
        if position == len(self.candidate_sets):
            candidates_to_filter = list(beam)
            recovered_from_final_pruned = False
            valid = [
                candidate
                for candidate in candidates_to_filter
                if self.constraint_checker(candidate.assignment)
            ]
            if not valid and self.recover_final_if_empty and self._last_complete_expanded:
                candidates_to_filter = self._last_complete_expanded
                recovered_from_final_pruned = True
                valid = [
                    candidate
                    for candidate in candidates_to_filter
                    if self.constraint_checker(candidate.assignment)
                ]
            invalid = [
                candidate
                for candidate in candidates_to_filter
                if not self.constraint_checker(candidate.assignment)
            ]
            valid = self._sort_candidates(valid)
            invalid = self._sort_candidates(invalid)
            kept = valid[: self.beam_width]
            pruned = valid[self.beam_width :] + invalid
            self._record_step(
                position=position,
                action="validity_filter",
                input_beam=beam,
                kept=kept,
                pruned=pruned,
                candidate_labels=[],
                expanded_count=len(candidates_to_filter),
            )
            self.trace[-1]["recovered_from_final_pruned"] = recovered_from_final_pruned
            return kept

        candidate_labels = self._rank_candidate_labels(position, log_probs)
        expanded = []
        for candidate in beam:
            for label in candidate_labels:
                assignment = candidate.assignment + (label,)
                score = candidate.score + float(log_probs[position, label])
                expanded.append(BeamCandidate(assignment=assignment, score=score))

        expanded = self._sort_candidates(expanded)
        if position == len(self.candidate_sets) - 1:
            self._last_complete_expanded = expanded
        kept = expanded[: self.beam_width]
        pruned = expanded[self.beam_width :]
        self._record_step(
            position=position,
            action="expand",
            input_beam=beam,
            kept=kept,
            pruned=pruned,
            candidate_labels=candidate_labels,
            expanded_count=len(expanded),
        )

        return self.search(position + 1, kept, log_probs)

    def _validate_probabilities(self, probabilities: np.ndarray) -> None:
        if probabilities.ndim != 2:
            raise ValueError("probabilities must have shape (num_positions, num_classes)")
        if probabilities.shape[0] != len(self.candidate_sets):
            raise ValueError(
                "probabilities must have one row for each candidate set "
                f"({len(self.candidate_sets)} expected)"
            )

        max_label = max(max(labels) for labels in self.candidate_sets if labels)
        if probabilities.shape[1] <= max_label:
            raise ValueError(
                "probabilities must include scores for every candidate label "
                f"(need class index {max_label})"
            )
        if np.any(probabilities < 0):
            raise ValueError("probabilities cannot contain negative values")

    def _rank_candidate_labels(self, position: int, log_probs: np.ndarray) -> list[int]:
        labels = list(self.candidate_sets[position])
        labels.sort(key=lambda label: (-float(log_probs[position, label]), label))
        return labels

    def _sort_candidates(self, candidates: Sequence[BeamCandidate]) -> list[BeamCandidate]:
        return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.assignment))

    def _record_step(
        self,
        position: int,
        action: str,
        input_beam: Sequence[BeamCandidate],
        kept: Sequence[BeamCandidate],
        pruned: Sequence[BeamCandidate],
        candidate_labels: Sequence[int],
        expanded_count: int,
    ) -> None:
        self.trace.append(
            {
                "position": position,
                "action": action,
                "beam_width": self.beam_width,
                "candidate_labels": list(candidate_labels),
                "input_beam": [self._candidate_to_dict(candidate) for candidate in input_beam],
                "kept": [self._candidate_to_dict(candidate) for candidate in kept],
                "pruned": [self._candidate_to_dict(candidate) for candidate in pruned],
                "expanded_count": expanded_count,
                "kept_count": len(kept),
                "pruned_count": len(pruned),
            }
        )

    def _summarize_trace(self, elapsed_seconds: float) -> dict:
        expanded_candidates = sum(
            int(step.get("expanded_count", 0))
            for step in self.trace
            if step.get("action") == "expand"
        )
        validity_checked_candidates = sum(
            int(step.get("expanded_count", 0))
            for step in self.trace
            if step.get("action") == "validity_filter"
        )
        retained_candidates = sum(
            int(step.get("kept_count", 0))
            for step in self.trace
            if step.get("action") == "expand"
        )
        total_candidates_evaluated = expanded_candidates + validity_checked_candidates
        return {
            "beam_width": self.beam_width,
            "elapsed_seconds": float(elapsed_seconds),
            "elapsed_ms": float(elapsed_seconds * 1000.0),
            "expanded_candidates": int(expanded_candidates),
            "retained_candidates": int(retained_candidates),
            "validity_checked_candidates": int(validity_checked_candidates),
            "total_candidates_evaluated": int(total_candidates_evaluated),
            "score_computations": int(expanded_candidates),
            "log_probability_sums": int(expanded_candidates),
            "final_valid_candidates": int(self.trace[-1].get("kept_count", 0)) if self.trace else 0,
            "search_steps": int(len(self.trace)),
        }

    @staticmethod
    def _candidate_to_dict(candidate: BeamCandidate) -> dict:
        return {
            "assignment": list(candidate.assignment),
            "score": float(candidate.score),
        }
