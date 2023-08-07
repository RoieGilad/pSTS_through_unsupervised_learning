
def Q1(S):
    counter = dict()
    letters = {'A': 3, 'B': 1, 'N': 2}
    res = float('inf')
    for s in S:
        if s in "BANANA":
            if s not in counter:
                counter[s] = 0
            counter[s] += 1
    for k,v in letters.items():
        if k not in counter:
            return 0
        else:
            res = min(res, counter[k]//v)
    return 0


import heapq
def Q2(A):
    curr_sum = sum(A)
    target = curr_sum / 2
    A = [-1*num for num in A]
    heapq.heapify(A)
    res = 0
    while curr_sum > target:
        curr_max = heapq.heappop(A)
        curr_max = curr_max / 2
        curr_sum += curr_max    # negative number
        heapq.heappush(A, curr_max)
        res += 1
    return res

