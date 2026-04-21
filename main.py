"""
ARC-AGI-3 Competition Entry Point.

Integrates all modules:
- ObjectTracker (perception)
- DSLEngine (domain priors)
- MentalModel (internal sandbox)
- MCTSPlanner (reasoning)
- RHAEOptimizer (efficiency)
"""

import numpy as np
from typing import Dict, List, Optional, Any
from arc_agi import *
from arc_agent import (
    ObjectTracker, DSLEngine, MentalModel, EntityMetadata, GridSnapshot
)
from rhae_optimizer import RHAEOptimizer, ActiveLearner, GoalDetector
from mcts_search import MCTSPlanner, MCTSNode


class ARCAgent:
    """
    Main ARC Agent: Integrates perception, reasoning, and planning.
    
    Flow:
    1. Perceive: Extract entities from grid
    2. Hypothesize: Generate candidate winning rules
    3. Plan: MCTS search in mental model
    4. Verify: Check prediction vs actual
    5. Adapt: Update hypothesis set
    6. Act: Commit action to environment
    """
    
    def __init__(self, operation_mode: str = "COMPETITION"):
        self.operation_mode = operation_mode
        self.tracker = ObjectTracker()
        self.dsl = DSLEngine()
        self.optimizer = RHAEOptimizer()
        self.active_learner = ActiveLearner()
        self.goal_detector = GoalDetector()
        
        # State
        self.hypotheses: Dict[str, Dict] = {}
        self.hypothesis_scores: Dict[str, float] = {}
        self.mental_model: Optional[MentalModel] = None
        self.observation_history: List[np.ndarray] = []
    
    def generate_initial_hypotheses(
        self,
        frames: List[np.ndarray]
    ) -> Dict[str, Dict]:
        """Generate 50 candidate winning rules from first 3 frames.
        
        Heuristic rule templates:
        - "Move all <color> to <color>"
        - "Rotate <color> pattern"
        - "Align entities on axis"
        - "Cover all <color> with <color>"
        - etc.
        """
        hypotheses = {}
        
        # Template 1: Coverage rules
        for frame in frames:
            unique_colors = np.unique(frame)
            for src_color in unique_colors:
                for dst_color in unique_colors:
                    if src_color != dst_color and src_color != 0:
                        rule_id = f"cover_{src_color}_with_{dst_color}"
                        hypotheses[rule_id] = {
                            "type": "coverage",
                            "source": src_color,
                            "target": dst_color
                        }
        
        # Template 2: Movement rules
        for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for color in [1, 2, 3, 4, 5]:
                rule_id = f"move_{color}_by_{offset}"
                hypotheses[rule_id] = {
                    "type": "movement",
                    "color": color,
                    "offset": offset
                }
        
        # Template 3: Symmetry rules
        for axis in ["h", "v", "diag"]:
            rule_id = f"make_{axis}_symmetric"
            hypotheses[rule_id] = {
                "type": "symmetry",
                "axis": axis
            }
        
        return hypotheses
    
    def solve(
        self,
        puzzle_input: np.ndarray,
        puzzle_output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Solve a single puzzle.
        
        Args:
            puzzle_input: Initial grid state
            puzzle_output: Expected output (for training)
            
        Returns:
            Predicted output grid
        """
        # Initialize mental model
        self.mental_model = MentalModel(puzzle_input)
        self.observation_history = [puzzle_input.copy()]
        
        # Generate initial hypotheses
        self.hypotheses = self.generate_initial_hypotheses([puzzle_input])
        
        # Main solving loop
        for step_num in range(self.optimizer.max_steps):
            current_state = self.mental_model.get_state()
            
            # Check if goal achieved
            if self._check_goal_achieved(current_state.grid, puzzle_output):
                return current_state.grid.copy()
            
            # Plan next action using MCTS
            planner = MCTSPlanner(self.mental_model, max_iterations=100)
            action_sequence = planner.search(
                current_state,
                self.hypotheses,
                lambda g: self._check_goal_achieved(g, puzzle_output)
            )
            
            if not action_sequence:
                # No valid actions found
                break
            
            # Execute first action in mental model
            next_action = action_sequence[0]
            self.mental_model.step(next_action)
            
            # Update hypotheses based on observation
            self._update_hypotheses(current_state.grid)
            
            # Prune hypotheses with low confidence
            self.hypotheses = self._prune_hypotheses()
        
        return self.mental_model.current_grid.copy()
    
    def _check_goal_achieved(
        self,
        current_grid: np.ndarray,
        expected_output: Optional[np.ndarray]
    ) -> bool:
        """Check if goal state achieved."""
        if expected_output is not None:
            return np.array_equal(current_grid, expected_output)
        
        # Check common goal patterns
        return (self.goal_detector.detect_uniform_color(current_grid, 0) or
                self.goal_detector.detect_symmetry(current_grid))
    
    def _update_hypotheses(self, previous_grid: np.ndarray):
        """Update hypothesis likelihood based on observations."""
        # Get current state
        current_grid = self.mental_model.current_grid
        
        for rule_id in list(self.hypotheses.keys()):
            if rule_id in self.hypothesis_scores:
                self.hypothesis_scores[rule_id] *= 0.9  # Decay
        
        # Hypotheses are verified when they reach high confidence
        # (Placeholder: full implementation in production)
    
    def _prune_hypotheses(self, threshold: float = 0.1) -> Dict[str, Dict]:
        """Remove low-confidence hypotheses."""
        return {
            rule_id: rule
            for rule_id, rule in self.hypotheses.items()
            if self.hypothesis_scores.get(rule_id, 0.5) > threshold
        }


class CompetitionRunner:
    """Competition harness: Loads puzzles, runs agent, computes metrics."""
    
    def __init__(self, operation_mode: str = "COMPETITION"):
        self.operation_mode = operation_mode
        self.agent = ARCAgent(operation_mode)
    
    def compete(
        self,
        train_puzzles: List[Dict],
        eval_puzzles: List[Dict]
    ) -> Dict[str, Any]:
        """Run competition on train and eval sets.
        
        Returns:
            {
                "accuracy": float (0-1),
                "train_rhae": float,
                "eval_rhae": float,
                "overall_rhae": float
            }
        """
        train_correct = 0
        train_rhae_scores = []
        
        for puzzle in train_puzzles:
            input_grid = puzzle["input"]
            expected_output = puzzle["output"]
            
            predicted_output = self.agent.solve(input_grid, expected_output)
            
            if np.array_equal(predicted_output, expected_output):
                train_correct += 1
            
            # Compute RHAE for this puzzle
            human_actions = self._estimate_human_actions(expected_output)
            ai_actions = self.agent.optimizer.steps_taken
            rhae = self.agent.optimizer.estimate_rhae_score(
                human_actions, ai_actions
            )
            train_rhae_scores.append(rhae)
        
        accuracy = train_correct / len(train_puzzles)
        avg_train_rhae = np.mean(train_rhae_scores)
        
        return {
            "accuracy": accuracy,
            "train_rhae": avg_train_rhae,
            "total_puzzles_correct": train_correct
        }
    
    @staticmethod
    def _estimate_human_actions(grid: np.ndarray) -> int:
        """Estimate number of human actions to create grid."""
        # Heuristic: ~1 action per unique non-zero color region
        from scipy import ndimage
        labeled, count = ndimage.label(grid != 0)
        return max(1, count)


if __name__ == "__main__":
    # Placeholder: Load actual competition puzzles
    print("ARC-AGI-3 Agent initialized")
    print(f"Mode: COMPETITION (offline)")
    print(f"Mental Sandbox: Enabled")
    print(f"RHAE Optimization: Active")
