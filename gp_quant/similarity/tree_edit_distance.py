"""
Tree Edit Distance (TED) 實作

基於 Zhang-Shasha 演算法的受限樹編輯距離計算。
使用動態規劃方法，將森林距離問題簡化為字串編輯距離問題。

參考文獻:
    Zhang, K., & Shasha, D. (1989). Simple fast algorithms for the editing 
    distance between trees and related problems. SIAM Journal on Computing, 
    18(6), 1245-1262.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

# DEAP 是可選依賴，只在需要時導入
try:
    from deap import gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    gp = None


class TreeNode:
    """樹節點的統一表示"""
    
    def __init__(self, label: str, children: List['TreeNode'] = None):
        """
        初始化樹節點
        
        Args:
            label: 節點標籤
            children: 子節點列表
        """
        self.label = label
        self.children = children if children is not None else []
        self.id = None  # 後序遍歷的索引
    
    def __repr__(self):
        if not self.children:
            return f"{{{self.label}}}"
        children_str = "".join([repr(child) for child in self.children])
        return f"{{{self.label}{children_str}}}"


class TreeEditDistance:
    """
    Tree Edit Distance (TED) 計算器
    
    使用 Zhang-Shasha 演算法計算兩棵有序標籤樹之間的編輯距離。
    支援三種編輯操作：插入、刪除、重命名。
    
    Attributes:
        cost_insert: 插入操作的成本函數
        cost_delete: 刪除操作的成本函數
        cost_rename: 重命名操作的成本函數
    """
    
    def __init__(self, 
                 cost_insert=None, 
                 cost_delete=None, 
                 cost_rename=None):
        """
        初始化 TED 計算器
        
        Args:
            cost_insert: 插入成本函數 f(node) -> float，默認為 1.0
            cost_delete: 刪除成本函數 f(node) -> float，默認為 1.0
            cost_rename: 重命名成本函數 f(node1, node2) -> float，
                        默認為 0.0 (相同) 或 1.0 (不同)
        """
        self.cost_insert = cost_insert or (lambda node: 1.0)
        self.cost_delete = cost_delete or (lambda node: 1.0)
        self.cost_rename = cost_rename or (
            lambda n1, n2: 0.0 if n1.label == n2.label else 1.0
        )
    
    def compute(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """
        計算兩棵樹之間的編輯距離
        
        Args:
            tree1: 第一棵樹
            tree2: 第二棵樹
            
        Returns:
            float: 編輯距離
        """
        # 步驟 1: 後序遍歷，為節點分配 ID
        nodes1 = self._post_order_traversal(tree1)
        nodes2 = self._post_order_traversal(tree2)
        
        n1 = len(nodes1)
        n2 = len(nodes2)
        
        # 步驟 2: 初始化 DP 表
        # D_tree[i][j] = 以 i 為根的子樹與以 j 為根的子樹的距離
        # D_forest[i][j] = 根植於 i 的森林與根植於 j 的森林的距離
        D_tree = np.zeros((n1 + 1, n2 + 1))
        D_forest = np.zeros((n1 + 1, n2 + 1))
        
        # 步驟 3: 初始化邊界條件（刪除/插入整棵樹）
        for i in range(1, n1 + 1):
            D_tree[i][0] = D_tree[i-1][0] + self.cost_delete(nodes1[i-1])
        
        for j in range(1, n2 + 1):
            D_tree[0][j] = D_tree[0][j-1] + self.cost_insert(nodes2[j-1])
        
        # 步驟 4: 動態規劃主循環
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                node_i = nodes1[i-1]
                node_j = nodes2[j-1]
                
                # 計算森林距離 D_forest[i][j]
                D_forest[i][j] = self._compute_forest_distance(
                    node_i, node_j, nodes1, nodes2, D_tree, i, j
                )
                
                # 計算樹距離 D_tree[i][j]
                # 三種情況的最小值：
                # 1. 刪除 node_i
                cost_delete = D_tree[i-1][j] + self.cost_delete(node_i)
                
                # 2. 插入 node_j
                cost_insert = D_tree[i][j-1] + self.cost_insert(node_j)
                
                # 3. 重命名 node_i -> node_j
                cost_rename = D_forest[i][j] + self.cost_rename(node_i, node_j)
                
                D_tree[i][j] = min(cost_delete, cost_insert, cost_rename)
        
        return D_tree[n1][n2]
    
    def _post_order_traversal(self, root: TreeNode) -> List[TreeNode]:
        """
        後序遍歷樹，並為每個節點分配 ID
        
        Args:
            root: 樹的根節點
            
        Returns:
            List[TreeNode]: 後序遍歷的節點列表
        """
        result = []
        self._post_order_helper(root, result)
        
        # 分配 ID
        for idx, node in enumerate(result):
            node.id = idx
        
        return result
    
    def _post_order_helper(self, node: TreeNode, result: List[TreeNode]):
        """後序遍歷的遞迴輔助函數"""
        if node is None:
            return
        
        # 先遍歷所有子節點
        for child in node.children:
            self._post_order_helper(child, result)
        
        # 再訪問當前節點
        result.append(node)
    
    def _compute_forest_distance(self, 
                                 node_i: TreeNode, 
                                 node_j: TreeNode,
                                 nodes1: List[TreeNode],
                                 nodes2: List[TreeNode],
                                 D_tree: np.ndarray,
                                 i: int,
                                 j: int) -> float:
        """
        計算森林距離，將其簡化為字串編輯距離問題
        
        Args:
            node_i: 樹1的當前節點
            node_j: 樹2的當前節點
            nodes1: 樹1的所有節點（後序）
            nodes2: 樹2的所有節點（後序）
            D_tree: 樹距離 DP 表
            i: node_i 在 nodes1 中的索引 (1-based)
            j: node_j 在 nodes2 中的索引 (1-based)
            
        Returns:
            float: 森林距離
        """
        children1 = node_i.children
        children2 = node_j.children
        
        m = len(children1)
        n = len(children2)
        
        # E[s][t] = 前 s 個子樹與前 t 個子樹的距離
        E = np.zeros((m + 1, n + 1))
        
        # 初始化邊界條件
        for s in range(1, m + 1):
            child_id = children1[s-1].id + 1  # 轉為 1-based
            E[s][0] = E[s-1][0] + D_tree[child_id][0]
        
        for t in range(1, n + 1):
            child_id = children2[t-1].id + 1  # 轉為 1-based
            E[0][t] = E[0][t-1] + D_tree[0][child_id]
        
        # 計算字串編輯距離
        for s in range(1, m + 1):
            for t in range(1, n + 1):
                child1_id = children1[s-1].id + 1
                child2_id = children2[t-1].id + 1
                
                # 三種操作
                cost_replace = E[s-1][t-1] + D_tree[child1_id][child2_id]
                cost_delete = E[s-1][t] + D_tree[child1_id][0]
                cost_insert = E[s][t-1] + D_tree[0][child2_id]
                
                E[s][t] = min(cost_replace, cost_delete, cost_insert)
        
        return E[m][n]
    
    def compute_similarity(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """
        計算兩棵樹的相似度（0-1之間，越大越相似）
        
        Args:
            tree1: 第一棵樹
            tree2: 第二棵樹
            
        Returns:
            float: 相似度分數 [0, 1]
        """
        distance = self.compute(tree1, tree2)
        
        # 使用標準化公式將距離轉換為相似度
        # similarity = 1 / (1 + distance)
        # 當 distance = 0 時，similarity = 1.0 (完全相同)
        # 當 distance → ∞ 時，similarity → 0.0 (完全不同)
        similarity = 1.0 / (1.0 + distance)
        
        return similarity


def deap_to_tree_node(individual) -> TreeNode:
    """
    將 DEAP Individual 轉換為 TreeNode
    
    Args:
        individual: DEAP Individual (PrimitiveTree)
        
    Returns:
        TreeNode: 轉換後的樹節點
    """
    if not DEAP_AVAILABLE:
        raise ImportError("DEAP is required for this function. Install it with: pip install deap")
    
    if not individual:
        return None
    
    # 使用堆疊來構建樹
    stack = []
    
    for node in reversed(individual):
        if isinstance(node, gp.Primitive):
            # Primitive 節點：需要從堆疊中取出子節點
            children = []
            for _ in range(node.arity):
                if stack:
                    children.insert(0, stack.pop())
            
            tree_node = TreeNode(node.name, children)
            stack.append(tree_node)
        
        elif isinstance(node, gp.Terminal):
            # Terminal 節點：沒有子節點
            tree_node = TreeNode(str(node.value))
            stack.append(tree_node)
    
    # 堆疊中應該只剩下根節點
    return stack[0] if stack else None


def tree_node_to_bracket(node: TreeNode) -> str:
    """
    將 TreeNode 轉換為括號表示法
    
    Args:
        node: 樹節點
        
    Returns:
        str: 括號表示法字符串
    """
    if not node.children:
        return f"{{{node.label}}}"
    
    children_str = "".join([tree_node_to_bracket(child) for child in node.children])
    return f"{{{node.label}{children_str}}}"


# 便捷函數
def compute_ted(tree1, tree2) -> float:
    """
    計算兩棵 DEAP tree 的編輯距離
    
    Args:
        tree1: DEAP Individual
        tree2: DEAP Individual
        
    Returns:
        float: 編輯距離
    """
    node1 = deap_to_tree_node(tree1)
    node2 = deap_to_tree_node(tree2)
    
    ted = TreeEditDistance()
    return ted.compute(node1, node2)


def compute_similarity(tree1, tree2) -> float:
    """
    計算兩棵 DEAP tree 的相似度
    
    Args:
        tree1: DEAP Individual
        tree2: DEAP Individual
        
    Returns:
        float: 相似度 [0, 1]
    """
    node1 = deap_to_tree_node(tree1)
    node2 = deap_to_tree_node(tree2)
    
    ted = TreeEditDistance()
    return ted.compute_similarity(node1, node2)
