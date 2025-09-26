#  EASY
from functools import cache
from typing import Optional, List


class Solution:
    def searchInsertInefficient(self, nums: list[int], target: int) -> int:
        l = 0
        r = len(nums)
        while True:
            m = (l + r) // 2
            if len(nums[l:r]) == 1:
                if target > nums[m]:
                    return m + 1
                else:
                    return m
            if target > nums[m]:
                l = m if m < r - 1 else r - 1
            else:
                r = m

    def searchInsert(self, nums: list[int], target: int) -> int:
        l = 0
        r = len(nums)
        while l<=r:
            m = (l + r) // 2
            if  target == nums[m]:
                return m
            elif target > nums[m]:
                l = m + 1
            else:
                r = m - 1

        return m

    def singleNumberInefficient(self, nums: list[int]) -> int:
        big_n = 6 * 10**4 + 1
        for e in nums:
            e_b = e + 3 * 10**4
            big_n ^= e_b

        return (6 * 10**4 + 1 ^ big_n)  - 30000

    def singleNumber(self, nums: list[int]) -> int:
        res = 0
        for e in nums:
            res ^= e

        return  res

# MEDIUM
class Solution:
    def subsets(self, nums: list[int]) -> list[list[int]]:
        power_set = [[],nums]
        for i in range(len(nums), 1, -1):
            inter = list()
            for j in range(0, len(nums)-i+1):
                sub = nums[j:j+i]
                for i_sub in range(len(sub)):
                    l = sub[:i_sub]
                    r = sub[i_sub+1:]
                    a = l+r
                    inter.append(tuple(a))
            inter = list(set(inter))
            power_set.extend(inter)
        return power_set

class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # ith row to jth column
        # i = 0 then j = n-1

        # Flip over the main diag -> flip vertically
        for i in range(len(matrix)):
            for j in range(i, len(matrix[i])):
                c = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = c
        for i in range(len(matrix)):
            for j in range(len(matrix[0]) // 2):
                c = matrix[i][j]
                matrix[i][j] = matrix[i][len(matrix)-j-1]
                matrix[i][len(matrix)-j-1] = c

        return matrix


from collections import deque

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def constructMaximumBinaryTree2(self, nums: list[int]) -> Optional[TreeNode]:
        print("nums=", nums)
        if not nums:
            return None

        maximum_e = -1
        maximum_i = -1
        for i, e in enumerate(nums):
            if e > maximum_e:
                maximum_e = e
                maximum_i = i

        val = maximum_e
        l_nums = nums[:maximum_i]
        r_nums = nums[maximum_i+1:]

        print("l_nums=", l_nums)
        print("r_nums=", r_nums)

        return TreeNode(
            val=val,
            left=self.constructMaximumBinaryTree2(l_nums),
            right=self.constructMaximumBinaryTree2(r_nums),
        )

    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        tree = self.constructMaximumBinaryTree2(nums)

        if not tree:
            return []

        res = list()
        dq = deque([tree])
        while dq:
            node = dq.popleft()
            if node:
                res.append(node.val)
                dq.append(node.left)
                dq.append(node.right)
            else:
                res.append(None)

        while res and res[-1] is None:
            res.pop()

        return res

class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        stack = []  #build a decreasing stack
        for i in nums:
            node = TreeNode(i)
            lastpop = None

            while stack and stack[-1].val < i:  #popping process
                lastpop = stack.pop()
            node.left = lastpop

            if stack:
                stack[-1].right = node
            stack.append(node)

        return stack[0]


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def find(self, node: Optional[TreeNode], q: int, flag_stack = False):
        if not node:
            return False

        if flag_stack:
            # FIXME
            [].append(node.val)

        if node.val == q:
            return True
        if self.find(node.left, q, flag_stack):
            return True
        if self.find(node.right, q, flag_stack):
            return True

        return False

    def find_traverse(self, root: Optional[TreeNode], k: int, flag_stack: bool=False):
        q = k - root.val
        f_l = self.find(root.left, q, flag_stack)
        f_r = self.find(root.right, q, flag_stack)

        return f_l or f_r

    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        self.find_traverse(root, k, True)

        for v in self.stack:
            root.val = v
            f = self.find_traverse(root, k, False)
            if f:
                return True

        return False

    # https://leetcode.com/problems/count-and-say/submissions/1778040877/?envType=problem-list-v2&envId=string
    class Solution:
        def rle_encode(self, s: str):
            res = ""
            i_s_p = s[0]
            c = 1
            for i_s in s[1:]:
                if i_s_p == i_s:
                    c += 1
                else:
                    res += str(c) + i_s_p
                    c = 1
                    i_s_p = i_s

            res += str(c) + i_s_p
            return res

    def countAndSay(self, n: int) -> str:
        s = "1"
        for i in range(n-1):
            s = self.rle_encode(s)

        return s

# https://leetcode.com/problems/edit-distance/?envType=problem-list-v2&envId=string
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)

        dp = [[None]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[i][j] = j
                if j == 0:
                    dp[i][j] = i
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])

        return dp[m][n]

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)

        dp = [[j if i ==0 else i for j in range(n+1)] for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])

        return dp[m][n]

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)

        if n > m:
            word1, word2 = word2, word1
            m, n = n, m

        prev_row = list(range(n+1))
        curr_row = [0] * (n+1)

        for i in range(1, m+1):
            curr_row[0] = i
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    curr_row[j] = prev_row[j-1]
                else:
                    curr_row[j] = 1 + min(prev_row[j-1],prev_row[j],curr_row[j-1])
            prev_row, curr_row = curr_row, prev_row

        return prev_row[n]

# https://leetcode.com/problems/substring-with-concatenation-of-all-words/?envType=problem-list-v2&envId=string
class Solution:
    def check_sequence(self, sub, words, stride):
        for i in range(0, len(sub), stride):
            flag = False
            for i_w, word in enumerate(words):
                if word == sub[i:i+stride]:
                    popped = words.pop(i_w)
                    flag = True
                    break
            if not flag:
                return False

        return len(words)==0

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        w_length = len(words[0])
        s_length = len(words) * w_length
        res = list()
        for i in range(0, len(s)):
            is_concatination = self.check_sequence(s[i:i+s_length], words[:], w_length)
            if is_concatination:
                res.append(i)

        return res

# https://leetcode.com/problems/pascals-triangle/?envType=problem-list-v2&envId=dynamic-programming
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        prev = [1]
        for i in range(1, numRows):
            prev = [0] + prev + [0]
            curr = []
            for j in range(0, i+1):
                curr.append(prev[j] + prev[j+1])
            res.append(curr)
            prev = curr
        return res


# https://leetcode.com/problems/climbing-stairs/?envType=problem-list-v2&envId=dynamic-programming
def climb(curr, n):
    if curr == n:
        return 1
    if curr > n:
        return 0

    return climb(curr+1, n) + climb(curr+2, n)

class Solution:
    def climbStairs(self, n: int) -> int:
        return climb(0, n)

class Solution:
    def climbStairs(self, n: int) -> int:
        steps = [1, 2]
        for i in range(2, n):
            steps.append(steps[-1] + steps[-2])

        return steps[n-1]

# https://leetcode.com/problems/maximum-subarray/?envType=problem-list-v2&envId=dynamic-programming
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        sub_max = nums[0]
        neg_buff = 0
        for _, e in enumerate(nums[1:]):
            print("e=", e)
            print("sub_max=", sub_max)
            print("neg_buff=", neg_buff)
            if e >= 0 and sub_max < 0:
                sub_max = e
                neg_buff = 0
            elif e >= 0 and sub_max > 0:
                if (sub_max + neg_buff) + e > sub_max:
                    sub_max += neg_buff + e
                    neg_buff = 0
                elif e+neg_buff > 0:
                    # need to create pos_buff => SOLUTION FAILS
                    pass
                else:
                    sub_max = e
                    neg_buff = 0
            elif e < 0 and sub_max < 0:
                if e > sub_max:
                    sub_max = e
                    neg_buff = 0
                    # TODO: check this case
                else:
                    pass
            elif e < 0 and sub_max >= 0:
                neg_buff += e
            else:
                print("something went wrong")

            print("sub_max=", sub_max)
            print("neg_buff=", neg_buff)
            print()
        return sub_max

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        m = list()
        for i_s in range(len(nums)):
            curr = [nums[i_s]]
            for j_e in range(i_s+1, len(nums)):
                curr.append(curr[j_e - i_s -1] + nums[j_e])
            m.append(max(curr))

        return max(m)

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        m = nums[0]
        for i_s in range(len(nums)):
            v = nums[i_s]
            curr = [v]
            if v > m:
                m = v
            for j_e in range(i_s+1, len(nums)):
                v = curr[j_e - i_s - 1] + nums[j_e]
                curr.append(v)
                if v > m:
                    m = v

        return m

# https://leetcode.com/problems/jump-game/?envType=problem-list-v2&envId=dynamic-programming
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        i = 0
        n = len(nums)
        prev = "1" * n
        for i, e in enumerate(nums[:-1]):
            curr = ["0"] * n
            for j in range(e+1):
                curr[i+j] = "1"
            curr = "".join(curr)
            # logical and with prev
            u = bin(int(prev, 2) & int(curr, 2))
            print(u)
            # u > 0
            if "1" in u:
                # logical or to carry forward
                prev = bin(int(prev, 2) | int(curr, 2))
            else:
                return False

        curr = "".join(["0"] * (n-1) +["1"])
        return "1" in bin(int(prev, 2) & int(curr, 2))

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reachable_i = 0
        for i, e in enumerate(nums):
            if i > max_reachable_i:
                return False
            max_reachable_i = max(max_reachable_i, i + e)

        return True


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/?envType=problem-list-v2&envId=dynamic-programming
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        profit = prices[1] - prices[0] if prices[1] > prices[0] else 0
        minimum1 = prices[0]
        maximum1 = prices[1]
        for e in prices[1:]:
            if e < minimum1:
                minimum1 = e
                maximum1 = 0
            else:
                maximum1 = max(e, maximum1)
                profit1 = maximum1 - minimum1
                profit = max(profit1, profit)

        return profit

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        minimum1 = 10e6
        for e in prices:
            minimum1 = min(minimum1, e)
            profit = max(profit,e-minimum1)

        return profit

# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/?envType=problem-list-v2&envId=dynamic-programming
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minimum = prices[0]+1
        maximum = 0
        profit = 0
        for i in range(len(prices)):
            print("i=",i)

            if prices[i] < minimum:
                print("here")
                minimum = prices[i]
                diff = maximum - minimum
                profit += diff if diff > 0 else 0
                maximum = 0
            else:
                if prices[i] > maximum:
                    print("there")
                    maximum = prices[i]
                    diff = maximum - minimum
                    profit += diff if diff > 0 else 0
                    minimum = prices[i]
                else:
                    print("there")
                    diff = maximum - minimum
                    profit += diff if diff > 0 else 0
                    minimum = prices[i]
                    maximum = 0

            print("maximum=",maximum)
            print("minimum=",minimum)
            print("profit=",profit)
            print()
        # if maximum > minimum:
        #     profit += maximum - minimum
        return profit

# https://leetcode.com/problems/unique-binary-search-trees/?envType=problem-list-v2&envId=dynamic-programming
    # class Solution:
#     def numTrees(self, n: int) -> int:
#         if n <= 1: return 1
#         return sum( [ self.numTrees(i-1) * self.numTrees(n-i) for i in range(1, n+1) ] )
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n+1):
            for j in range(1, i+1):
                dp[i] += dp[j-1] * dp[i-j]
        return dp[n]

class Solution:
    @cache
    def numTrees(self, n: int) -> int:
        if n <= 1: return 1
        return sum(self.numTrees(i-1) * self.numTrees(n-i) for i in range(1, n+1))


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# https://leetcode.com/problems/binary-tree-level-order-traversal/?envType=problem-list-v2&envId=breadth-first-search
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        res = []
        stack = [[root]]

        while stack:
            nodes = stack.pop()
            if not nodes:
                break

            level_vals = []
            next_level_nodes = []
            for node in nodes:
                level_vals.append(node.val)

                if node.left and node.right:
                    next_level_nodes.append(node.left)
                    next_level_nodes.append(node.right)
                elif node.left and not node.right:
                    next_level_nodes.append(node.left)
                elif not node.left and node.right:
                    next_level_nodes.append(node.right)
                else:
                    pass

            res.append(level_vals)
            stack.append(next_level_nodes)

        return res


from collections import deque

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        res = []
        dq = deque()
        dq.append(root)

        while dq:
            level_vals = list()
            current_dq_length = len(dq)

            for _ in range(current_dq_length):
                node = dq.popleft()

                if not node:
                    print("error")
                    break

                level_vals.append(node.val)

                if node.left and node.right:
                    dq.append(node.left)
                    dq.append(node.right)
                elif node.left and not node.right:
                    dq.append(node.left)
                elif not node.left and node.right:
                    dq.append(node.right)
                else:
                    pass

            res.append(level_vals)

        return res


# https://leetcode.com/problems/recover-binary-search-tree/?envType=problem-list-v2&envId=depth-first-search
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """

        stack = [root]

        while stack:
            node = stack.pop()

            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

            if node.left:
                substack_left = [node.left]
            else:
                substack_left = []
            while substack_left:
                sub_node = substack_left.pop()
                if sub_node.val > node.val:
                    temp = node.val
                    node.val = sub_node.val
                    sub_node.val = temp

                if sub_node.left:
                    substack_left.append(sub_node.left)
                if sub_node.right:
                    substack_left.append(sub_node.right)

            substack_right = [node.right] if node.right else []

            while substack_right:
                sub_node = substack_right.pop()
                if sub_node.val < node.val:
                    temp = node.val
                    node.val = sub_node.val
                    sub_node.val = temp

                if sub_node.left:
                    substack_right.append(sub_node.left)
                if sub_node.right:
                    substack_right.append(sub_node.right)

        return root


