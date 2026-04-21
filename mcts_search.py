"""
Monte Carlo Tree Search (MCTS) for Hypothesis-Aware Planning.

Algorithm:
1. Selection: UCB1 to balance exploration/exploitation
2. Expansion: Add new child nodes
3. Simulation: Random rollout (or greedy in our case)
4. Backpropagation: Update statistics

Modified for hypothesis tracking to prune impossible branches early.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import math
from arc_agent import MentalModel, GridSnapshot


@dataclass
class MCTSNode:
    """Node in MCTS tree."""
    state: GridSnapshot
    parent: Optional["MCTSNode"] = None
    action: Optional[Dict] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    active_hypotheses: Set[str] = field(default_factory=set)
    
    def ucb1_value(self, c: float = 1.414) -> float:
        """
        UCB1 = (value / visits) + c * sqrt(ln(parent_visits) / visits)
        
        Balances exploitation (high value) and exploration (few visits).
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploitation + exploration


class MCTSPlanner:
    """MCTS planner for hypothesis-driven search.
    
    Each node tracks which hypotheses are still compatible with
    the path taken to reach it. Branches that violate all
    hypotheses are pruned immediately.
    """
    
    def __init__(self,
        mental_model: MentalModel,
        max_iterations: int = 1000,
        time_budget_ms: int = 50
    ):
        self.mental_model = mental_model
        self.max_iterations = max_iterations
        self.time_budget_ms = time_budget_ms
        self.root: Optional[MCTSNode] = None
    
    def search(
        self,
        initial_state: GridSnapshot,
        hypotheses: Dict[str, Dict],
        goal_check_fn
    ) -> List[Dict]:
        """Execute MCTS search.
        
        Args:
            initial_state: Starting grid state
            hypotheses: Dict[rule_id -> rule_definition]
            goal_check_fn: Function to check if goal achieved
            
        Returns:
            Sequence of actions to solve puzzle
        """
        self.root = MCTSNode(
            state=initial_state,
            active_hypotheses=set(hypotheses.keys())
        )
        
        for iteration in range(self.max_iterations):
            # Selection
            node = self._select_best_node(self.root)
            
            # Expansion
            if node.visits > 0 or node == self.root:
                node = self._expand(node, hypotheses)
            
            # Simulation
            reward = self._simulate(node, goal_check_fn)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Extract best path
        return self._extract_best_path(self.root)
    
    def _select_best_node(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: UCB1 traversal."""
        while len(node.children) > 0 and not self._is_leaf(node):
            # Pick child with highest UCB1
            best_child = max(node.children, key=lambda c: c.ucb1_value())
            node = best_child
        
        return node
    
    def _expand(
        self,
        node: MCTSNode,
        hypotheses: Dict[str, Dict]
    ) -> MCTSNode:
        """Expansion phase: add new child nodes."""
        # Generate candidate actions
        actions = self._generate_actions(node.state)
        
        for action in actions:
            # Simulate action in mental model
            self.mental_model.reset_to_snapshot(node.state)
            new_state = self.mental_model.step(action)[0]
            
            # Filter hypotheses compatible with this action
            compatible_hypotheses = self._filter_hypotheses(
                node.active_hypotheses,
                hypotheses,
                action,
                new_state
            )
            
            # Only create child if hypotheses remain
            if compatible_hypotheses:
                child = MCTSNode(
                    state=GridSnapshot(grid=new_state),
                    parent=node,
                    action=action,
                    active_hypotheses=compatible_hypotheses
                )
                node.children.append(child)
        
        return node.children[0] if node.children else node
    
    def _simulate(
        self,
        node: MCTSNode,
        goal_check_fn
    ) -> float:
        """Simulation phase: rollout to leaf."""
        self.mental_model.reset_to_snapshot(node.state)
        
        steps = 0
        max_steps = 20
        
        while steps < max_steps:
            if goal_check_fn(self.mental_model.current_grid):
                return 1.0  # Goal reached
            
            actions = self._generate_actions(GridSnapshot(self.mental_model.current_grid))
            if not actions:
                break
            
            action = np.random.choice(actions)
            _, success = self.mental_model.step(action)
            
            if not success:
                break
            
            steps += 1
        
        return 0.0  # Goal not reached
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase: update statistics."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _generate_actions(self, state: GridSnapshot) -> List[Dict]:
        """Generate candidate actions for state."""
        actions = []
        
        # Get entities
        entities = state.entities
        
        # Movement actions for each entity
        for entity in entities:
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                actions.append({
                    "type": "move",
                    "entity_id": entity.id,
                    "dy": dy,
                    "dx": dx
                })
        
        # Paint actions
        for y in range(min(10, state.grid.shape[0])):
            for x in range(min(10, state.grid.shape[1])):
                for color in [1, 2, 3, 4, 5]:
                    actions.append({
                        "type": "paint",
                        "pos": (y, x),
                        "color": color
                    })
        
        return actions
    
    def _filter_hypotheses(
        self,
        active_hypotheses: Set[str],
        hypotheses: Dict[str, Dict],
        action: Dict,
        result_state: np.ndarray
    ) -> Set[str]:
        """Filter hypotheses to those compatible with action result."""
        # Placeholder: all hypotheses survive for now
        return active_hypotheses
    
    def _is_leaf(self, node: MCTSNode) -> bool:
        """Check if node is leaf (no children)."""
        return len(node.children) == 0
    
    def _extract_best_path(self, root: MCTSNode) -> List[Dict]:
        """Extract best path from root to leaf."""
        path = []
        node = root
        
        while len(node.children) > 0:
            # Pick best child by visit count (most explored)
            best_child = max(node.children, key=lambda c: c.visits)
            if best_child.action:
                path.append(best_child.action)
            node = best_child
        
        return path