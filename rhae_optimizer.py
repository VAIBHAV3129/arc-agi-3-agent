"""
RHAE Optimization Module: Minimize env.step() calls.

RHAE Score = (Human Actions / AI Actions)²

Strategy:
1. Active Learning: Pick actions that maximize information gain
2. Action Grouping: Detect multi-object moves
3. Early Termination: Stop when goal detected
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from arc_agent import EntityMetadata, GridSnapshot
import math


@dataclass
class HypothesisScore:
    """Score for a hypothesis based on verification."""
    rule_id: str
    likelihood: float  # 0-1
    verified_frames: int
    failed_frames: int
    entropy: float  # Information uncertainty


class RHAEOptimizer:
    """
    RHAE Score Optimizer: Minimize env.step() calls.
    
    Theory:
    - Information Gain: Choose actions that eliminate most hypotheses
    - Multi-Object Moves: Detect group movements that reduce steps
    - Active Learning: Uncertainty sampling > random exploration
    """
    
    def __init__(self, max_steps_per_puzzle: int = 50):
        self.max_steps = max_steps_per_puzzle
        self.steps_taken = 0
        self.hypothesis_scores: Dict[str, HypothesisScore] = {}
    
    def compute_information_gain(
        self,
        hypothesis_set_before: Set[str],
        hypothesis_set_after: Set[str]
    ) -> float:
        """
        Compute information gain in bits for eliminating hypotheses.
        
        Information Gain = log2(|S_before| / |S_after|)
        
        Args:
            hypothesis_set_before: Hypotheses before action
            hypothesis_set_after: Hypotheses after action
            
        Returns:
            Bits of information gained (0 to log2(|S_before|))
        """
        before = len(hypothesis_set_before)
        after = len(hypothesis_set_after)
        
        if after == 0:
            return math.inf  # Perfect disambiguation
        if before == after:
            return 0.0  # No information gained
        
        return math.log2(before / after)
    
    def rank_actions_by_information_gain(
        self,
        candidate_actions: List[Dict],
        hypothesis_predictions: Dict[str, np.ndarray],
        true_observation: np.ndarray
    ) -> List[Tuple[Dict, float]]:
        """Rank actions by information gain (Active Learning).
        
        Args:
            candidate_actions: List of possible moves/paints
            hypothesis_predictions: Dict[rule_id -> predicted_grid]
            true_observation: Actual observation after action
            
        Returns:
            Sorted list of (action, info_gain) tuples
        """
        ranked = []
        
        for action in candidate_actions:
            # Simulate action under each hypothesis
            active_hypotheses = set(hypothesis_predictions.keys())
            surviving_hypotheses = set()
            
            for rule_id, prediction in hypothesis_predictions.items():
                # Check if this hypothesis matches true observation
                if np.allclose(prediction, true_observation):
                    surviving_hypotheses.add(rule_id)
            
            info_gain = self.compute_information_gain(
                active_hypotheses,
                surviving_hypotheses
            )
            ranked.append((action, info_gain))
        
        # Sort by descending information gain
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def detect_action_groups(
        self,
        entities: List[EntityMetadata]
    ) -> List[List[str]]:
        """Detect entities that can be moved together (group actions).
        
        Returns:
            List of entity ID groups that should move together
        """
        groups = []
        
        # Simple heuristic: same color, adjacent entities
        color_groups: Dict[int, List[EntityMetadata]] = {}
        for entity in entities:
            if entity.color not in color_groups:
                color_groups[entity.color] = []
            color_groups[entity.color].append(entity)
        
        for color, group_entities in color_groups.items():
            if len(group_entities) > 1:
                # Check adjacency
                group_ids = [e.id for e in group_entities]
                groups.append(group_ids)
        
        return groups
    
    def should_terminate_early(
        self,
        current_grid: np.ndarray,
        goal_predicates: List[Tuple[str, bool]]
    ) -> bool:
        """Detect goal state and terminate early.
        
        Args:
            current_grid: Current grid state
            goal_predicates: List of (predicate_name, is_satisfied) tuples
            
        Returns:
            True if goal achieved
        """
        # Goal achieved if all predicates are satisfied
        return all(satisfied for _, satisfied in goal_predicates)
    
    def estimate_rhae_score(
        self,
        human_actions: int,
        ai_actions: int
    ) -> float:
        """Compute RHAE score.
        
        RHAE = (H / A)²
        """
        if ai_actions == 0:
            return 0.0
        
        ratio = human_actions / ai_actions
        return ratio ** 2


class ActiveLearner:
    """Active Learning strategy for hypothesis disambiguation.
    
    Uncertainty Sampling: Pick actions that maximize disagreement among hypotheses.
    """
    
    def __init__(self, confidence_threshold: float = 0.95):
        self.confidence_threshold = confidence_threshold
        self.hypothesis_entropy: Dict[str, float] = {}
    
    def compute_uncertainty(
        self,
        hypothesis_set: Dict[str, HypothesisScore]
    ) -> float:
        """Compute Shannon entropy over hypothesis likelihood.
        
        H = -Σ p_i * log2(p_i)
        
        High entropy = high uncertainty = good for learning
        """
        total_likelihood = sum(h.likelihood for h in hypothesis_set.values())
        if total_likelihood == 0:
            return 0.0
        
        entropy = 0.0
        for hypothesis in hypothesis_set.values():
            p = hypothesis.likelihood / total_likelihood
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def select_most_uncertain_action(
        self,
        candidate_actions: List[Dict],
        uncertainty_scores: List[float]
    ) -> Dict:
        """Pick action with highest uncertainty (most informative)."""
        if not candidate_actions:
            return {}
        
        idx = np.argmax(uncertainty_scores)
        return candidate_actions[idx]


class GoalDetector:
    """Goal state detection for early termination.
    
    Patterns:
    - Coverage: All of color X covered by color Y
    - Alignment: Objects aligned on axis
    - Containment: Object inside container
    - Filling: Grid completely filled
    """
    
    @staticmethod
    def detect_coverage(grid: np.ndarray, src: int, dst: int) -> bool:
        """All src pixels covered by dst."""
        src_mask = grid == src
        if not np.any(src_mask):
            return True
        dst_mask = grid == dst
        from scipy import ndimage
        dilated = ndimage.binary_dilation(dst_mask)
        return np.all(src_mask[dilated])
    
    @staticmethod
    def detect_uniform_color(grid: np.ndarray, color: int) -> bool:
        """Entire grid is single color."""
        return np.all(grid == color)
    
    @staticmethod
    def detect_symmetry(grid: np.ndarray) -> bool:
        """Grid has reflectional symmetry."""
        return (np.allclose(grid, np.fliplr(grid)) or
                np.allclose(grid, np.flipud(grid)))
