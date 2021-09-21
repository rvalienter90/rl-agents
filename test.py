from collections import deque
q =deque()
q.append(1)
q.append(2)
print(q.popleft())
print(q.popleft())

s = []
s.append(1)
s.append(2)
print(s.pop())
print(s.pop())


def dfs(n):
    if n is None:
        return
    print(n.value)
    dfs(n.right)
    print(n.value)
    dfs(n.left)

def bfs(n):
    q = deque()
    q.append(n.value)

    while q:
        temp = q.popleft()

        if temp.left:
            q.append(temp.left)
        if temp.right:
            q.append(temp.right)

