"""
Tree Kernel Implementation

Tree Kernel 提供基於共同子結構計數的樹相似度計算，作為 TED 的高效替代方案。
時間複雜度從 TED 的 O(n²m²) 降至 O(nm)。

支援的 Kernel 類型：
1. SubtreeKernel: 基於完整子樹匹配的 kernel（已實作）
2. SubsetTreeKernel: 基於部分後代匹配的 kernel（框架，待實作）

參考文獻:
    Collins, M., & Duffy, N. (2001). Convolution Kernels for Natural Language. 
    NIPS 2001.
"""

from typing import List, Optional, Callable
from functools import lru_cache
import numpy as np

from .tree_edit_distance import TreeNode

# DEAP 是可選依賴
try:
    from deap import gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    gp = None


class SubtreeKernel:
    """
    Subtree Kernel 計算器
    
    計算兩棵樹之間的共同完整子樹數量。完整子樹定義為：
    節點及其所有後代形成的完整分支。
    
    Attributes:
        lambda_decay: 衰減因子 (0 < λ ≤ 1)，控制深度懲罰
                     λ 越小，深層匹配的權重越低
    """
    
    def __init__(self, lambda_decay: float = 0.5):
        """
        Initialize Subtree Kernel.
        
        Args:
            lambda_decay: Decay factor for depth penalty. 
                         Lower values penalize deep matches more.
                         Default 0.5 provides balanced decay.
        """
        if not 0 < lambda_decay <= 1:
            raise ValueError(f"lambda_decay must be in (0, 1], got {lambda_decay}")
        self.lambda_decay = lambda_decay
        self._cache: dict = {}
    
    def clear_cache(self):
        """Clear the computation cache. Call this between different populations."""
        self._cache.clear()
    
    @staticmethod
    def count_nodes(root: 'TreeNode') -> int:
        """Count total nodes in a tree."""
        if root is None:
            return 0
        count = 1
        for child in root.children:
            count += SubtreeKernel.count_nodes(child)
        return count
    
    def _get_node_key(self, node: TreeNode) -> int:
        """Generate unique key for caching."""
        return id(node)
    
    def _delta(self, n1: TreeNode, n2: TreeNode) -> float:
        """
        Recursively compute the number of common subtrees rooted at n1 and n2.
        
        This is the core of Subtree Kernel: if two nodes have the same label
        and the same number of children, we recursively check all children.
        
        Args:
            n1: Node from first tree
            n2: Node from second tree
            
        Returns:
            Contribution to kernel value from this node pair
        """
        # Cache key based on node identity
        cache_key = (self._get_node_key(n1), self._get_node_key(n2))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Different labels -> no match
        if n1.label != n2.label:
            self._cache[cache_key] = 0.0
            return 0.0
        
        # Both are leaves with same label
        if not n1.children and not n2.children:
            result = self.lambda_decay
            self._cache[cache_key] = result
            return result
        
        # Different number of children -> no subtree match
        # (Subtree kernel requires complete structural match)
        if len(n1.children) != len(n2.children):
            self._cache[cache_key] = 0.0
            return 0.0
        
        # Recursively compute: all children must match
        product = self.lambda_decay
        for c1, c2 in zip(n1.children, n2.children):
            child_delta = self._delta(c1, c2)
            product *= (1.0 + child_delta)
        
        self._cache[cache_key] = product
        return product
    
    def compute(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """
        Compute the Subtree Kernel value between two trees.
        
        K(T1, T2) = Σ_{n1 ∈ T1} Σ_{n2 ∈ T2} Δ(n1, n2)
        
        Note: Cache is NOT cleared automatically. Call clear_cache() 
        between different populations to avoid memory growth.
        
        Args:
            tree1: First tree (TreeNode root)
            tree2: Second tree (TreeNode root)
            
        Returns:
            Kernel value (higher = more similar)
        """
        
        # Collect all nodes from both trees
        nodes1 = self._collect_nodes(tree1)
        nodes2 = self._collect_nodes(tree2)
        
        # Sum over all node pairs
        total = 0.0
        for n1 in nodes1:
            for n2 in nodes2:
                total += self._delta(n1, n2)
        
        return total
    
    def _collect_nodes(self, root: TreeNode) -> List[TreeNode]:
        """Collect all nodes in the tree via pre-order traversal."""
        nodes = [root]
        for child in root.children:
            nodes.extend(self._collect_nodes(child))
        return nodes
    
    def compute_normalized(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """
        Compute normalized kernel similarity in [0, 1].
        
        Normalized kernel: K(T1, T2) / sqrt(K(T1, T1) * K(T2, T2))
        
        Args:
            tree1: First tree
            tree2: Second tree
            
        Returns:
            Normalized similarity score in [0, 1]
        """
        k12 = self.compute(tree1, tree2)
        k11 = self.compute(tree1, tree1)
        k22 = self.compute(tree2, tree2)
        
        if k11 == 0 or k22 == 0:
            return 0.0
        
        return k12 / np.sqrt(k11 * k22)
    
    def compute_distance(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """
        Convert kernel similarity to distance metric.
        
        Distance: d(T1, T2) = sqrt(K(T1,T1) + K(T2,T2) - 2*K(T1,T2))
        
        This formula derives from the kernel distance in feature space.
        
        Args:
            tree1: First tree
            tree2: Second tree
            
        Returns:
            Distance value (lower = more similar)
        """
        k12 = self.compute(tree1, tree2)
        k11 = self.compute(tree1, tree1)
        k22 = self.compute(tree2, tree2)
        
        # Ensure non-negative due to numerical precision
        distance_sq = max(0.0, k11 + k22 - 2 * k12)
        return np.sqrt(distance_sq)
    
    def compute_normalized_distance(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """
        Compute normalized distance in [0, 1] range.
        
        d_norm = 1 - normalized_kernel
        
        Args:
            tree1: First tree
            tree2: Second tree
            
        Returns:
            Normalized distance in [0, 1] (0 = identical, 1 = completely different)
        """
        return 1.0 - self.compute_normalized(tree1, tree2)


class SubsetTreeKernel:
    """
    Subset Tree Kernel (SST) 計算器 - 框架
    
    SST 允許部分後代匹配（深度可截斷，但不允許 partial production）。
    比 Subtree Kernel 更細粒度，能捕捉更多子結構相似性。
    
    Status: NOT IMPLEMENTED (placeholder for future extension)
    
    參考文獻:
        Collins, M., & Duffy, N. (2001). Convolution Kernels for Natural Language.
    """
    
    def __init__(self, lambda_decay: float = 0.5, mu: float = 0.5):
        """
        Initialize Subset Tree Kernel.
        
        Args:
            lambda_decay: Decay factor for tree depth
            mu: Decay factor for production rule matching
        """
        self.lambda_decay = lambda_decay
        self.mu = mu
        raise NotImplementedError(
            "SubsetTreeKernel is not yet implemented. "
            "Use SubtreeKernel as an alternative."
        )
    
    def compute(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """Compute SST kernel value."""
        raise NotImplementedError("SubsetTreeKernel.compute() not implemented")
    
    def compute_normalized(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """Compute normalized SST kernel."""
        raise NotImplementedError("SubsetTreeKernel.compute_normalized() not implemented")
    
    def compute_distance(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """Convert SST kernel to distance."""
        raise NotImplementedError("SubsetTreeKernel.compute_distance() not implemented")


# ============================================================================
# Convenience Functions (similar to tree_edit_distance.py pattern)
# ============================================================================

def deap_to_tree_node(individual) -> TreeNode:
    """
    Convert DEAP Individual to TreeNode.
    
    Re-exported from tree_edit_distance for convenience.
    
    Args:
        individual: DEAP Individual (PrimitiveTree)
        
    Returns:
        TreeNode representation
    """
    from .tree_edit_distance import deap_to_tree_node as _deap_to_tree_node
    return _deap_to_tree_node(individual)


def compute_subtree_kernel(tree1, tree2, lambda_decay: float = 0.5) -> float:
    """
    Compute Subtree Kernel between two DEAP trees.
    
    Convenience function that handles DEAP to TreeNode conversion.
    
    Args:
        tree1: DEAP Individual or TreeNode
        tree2: DEAP Individual or TreeNode
        lambda_decay: Decay factor (default 0.5)
        
    Returns:
        Kernel value
    """
    # Convert if DEAP individuals
    if not isinstance(tree1, TreeNode):
        tree1 = deap_to_tree_node(tree1)
    if not isinstance(tree2, TreeNode):
        tree2 = deap_to_tree_node(tree2)
    
    kernel = SubtreeKernel(lambda_decay=lambda_decay)
    return kernel.compute(tree1, tree2)


def compute_kernel_similarity(tree1, tree2, lambda_decay: float = 0.5) -> float:
    """
    Compute normalized similarity between two DEAP trees.
    
    Args:
        tree1: DEAP Individual or TreeNode
        tree2: DEAP Individual or TreeNode
        lambda_decay: Decay factor
        
    Returns:
        Normalized similarity in [0, 1]
    """
    if not isinstance(tree1, TreeNode):
        tree1 = deap_to_tree_node(tree1)
    if not isinstance(tree2, TreeNode):
        tree2 = deap_to_tree_node(tree2)
    
    kernel = SubtreeKernel(lambda_decay=lambda_decay)
    return kernel.compute_normalized(tree1, tree2)


def compute_kernel_distance(tree1, tree2, lambda_decay: float = 0.5) -> float:
    """
    Compute kernel-based distance between two DEAP trees.
    
    This is the primary function for niching: lower distance = more similar.
    
    Args:
        tree1: DEAP Individual or TreeNode
        tree2: DEAP Individual or TreeNode
        lambda_decay: Decay factor
        
    Returns:
        Distance value (0 = identical)
    """
    if not isinstance(tree1, TreeNode):
        tree1 = deap_to_tree_node(tree1)
    if not isinstance(tree2, TreeNode):
        tree2 = deap_to_tree_node(tree2)
    
    kernel = SubtreeKernel(lambda_decay=lambda_decay)
    return kernel.compute_distance(tree1, tree2)
