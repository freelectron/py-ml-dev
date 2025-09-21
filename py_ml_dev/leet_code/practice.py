#  EASY
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




