#!/usr/bin/env python3
"""Ground-truth verifier pentru asserturile din coding_eval_extreme.py.
Implementari de referinta (care stiu ca sunt corecte) -> valorile reale.
Nu e parte din eval; e doar ca asserturile mele sa fie corecte (lectia dijkstra).
"""
import collections

# ---------- 1. distinct_subsequences (DP 2D, LeetCode 115) ----------
def distinct_subsequences(s, t):
    dp = [0] * (len(t) + 1)
    dp[0] = 1
    for i in range(1, len(s) + 1):
        for j in range(min(i, len(t)), 0, -1):
            if s[i - 1] == t[j - 1]:
                dp[j] += dp[j - 1]
    return dp[len(t)]

# ---------- 2. min_window (sliding window, LeetCode 76) ----------
def min_window(s, t):
    if not s or not t or len(t) > len(s):
        return ""
    need = collections.Counter(t)
    have = collections.Counter()
    required = len(need)
    formed = 0
    l = 0
    best = (float("inf"), 0, 0)
    for r, ch in enumerate(s):
        have[ch] += 1
        if ch in need and have[ch] == need[ch]:
            formed += 1
        while formed == required:
            if r - l + 1 < best[0]:
                best = (r - l + 1, l, r)
            have[s[l]] -= 1
            if s[l] in need and have[s[l]] < need[s[l]]:
                formed -= 1
            l += 1
    return "" if best[0] == float("inf") else s[best[1]:best[2] + 1]

# ---------- 3. word_ladder (BFS, LeetCode 127) ----------
def word_ladder(begin, end, word_list):
    words = set(word_list)
    if end not in words:
        return 0
    q = collections.deque([(begin, 1)])
    seen = {begin}
    while q:
        w, d = q.popleft()
        if w == end:
            return d
        for i in range(len(w)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                nw = w[:i] + c + w[i + 1:]
                if nw in words and nw not in seen:
                    seen.add(nw)
                    q.append((nw, d + 1))
    return 0

# ---------- 4. super_egg_drop (DP cu mutari, LeetCode 887) ----------
def super_egg_drop(k, n):
    moves = 0
    dp = [0] * (k + 1)
    while dp[k] < n:
        moves += 1
        for e in range(k, 0, -1):
            dp[e] = dp[e - 1] + dp[e] + 1
    return moves

# ---------- 5. largest_rectangle (monotonic stack, LeetCode 84) ----------
def largest_rectangle(heights):
    stack = []
    best = 0
    for i, h in enumerate(heights + [0]):
        start = i
        while stack and stack[-1][1] > h:
            idx, hh = stack.pop()
            best = max(best, hh * (i - idx))
            start = idx
        stack.append((start, h))
    return best

# ---------- alternativ: interleave (verific si asta, ca sa am rezerva) ----------
def interleave(s1, s2, s3):
    if len(s1) + len(s2) != len(s3):
        return False
    dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    dp[0][0] = True
    for i in range(len(s1) + 1):
        for j in range(len(s2) + 1):
            if i + j == 0:
                continue
            ok = False
            if i and dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]:
                ok = True
            if j and dp[i][j - 1] and s2[j - 1] == s3[i + j - 1]:
                ok = True
            dp[i][j] = ok
    return dp[len(s1)][len(s2)]


print("=== 1 distinct_subsequences ===")
for s, t in [("rabbbit", "rabbit"), ("babgbag", "bag"), ("abcde", "a"),
             ("abcde", "abd"), ("", ""), ("aaa", "aa"), ("abc", "abcd"),
             ("bagg", "bag")]:
    print(f"  ({s!r}, {t!r}) -> {distinct_subsequences(s, t)}")

print("=== 2 min_window ===")
for s, t in [("ADOBECODEBANC", "ABC"), ("a", "a"), ("a", "aa"), ("aa", "aa"),
             ("aaabaaa", "aab"), ("cabwefgewcwaefgcf", "cae"), ("abc", "d")]:
    w = min_window(s, t)
    print(f"  ({s!r}, {t!r}) -> {w!r}  len={len(w)}")

print("=== 3 word_ladder ===")
print("  hit/cog full      ->", word_ladder("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
print("  hit/cog fara cog  ->", word_ladder("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))
print("  a/c  [a,b,c]      ->", word_ladder("a", "c", ["a", "b", "c"]))
print("  hot/dog [hot,dog] ->", word_ladder("hot", "dog", ["hot", "dog"]))
print("  red/tax           ->", word_ladder("red", "tax", ["ted", "tex", "red", "tax", "tad", "den", "rex", "pee"]))

print("=== 4 super_egg_drop ===")
for k, n in [(2, 10), (1, 1), (2, 100), (3, 14), (1, 5), (4, 50), (2, 3)]:
    print(f"  k={k} n={n} -> {super_egg_drop(k, n)}")

print("=== 5 largest_rectangle ===")
for h in [[2, 1, 5, 6, 2, 3], [2, 4], [], [1, 1, 1, 1], [5, 4, 3, 4, 5], [2, 1, 2], [2, 1, 5, 6, 2, 3, 1]]:
    print(f"  {h} -> {largest_rectangle(h)}")

print("=== alt interleave ===")
for a, b, c in [("aabcc", "dbbca", "aadbbcbcac"), ("aabcc", "dbbca", "aadbbbaccc"),
                ("", "", ""), ("ab", "cd", "acbd"), ("ab", "cd", "abcd")]:
    print(f"  {a!r}+{b!r} vs {c!r} -> {interleave(a, b, c)}")
