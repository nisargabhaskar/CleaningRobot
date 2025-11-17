import heapq

def a_star(start, goal, visited, weight):
    (gx, gy) = goal
    pq = [(0, start)]  
    came_from = {}
    g_cost = {start: 0}
    
    while pq:
        f, (x, y) = heapq.heappop(pq)
        
        if (x, y) == goal:
            path = []
            cur = goal
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]
        
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            nei = (nx, ny)


            if (x, y) in visited and nei in visited:
                step_cost = 0
            else:
                step_cost = weight(nei)

            new_g = g_cost[(x, y)] + step_cost

            if nei not in g_cost or new_g < g_cost[nei]:
                g_cost[nei] = new_g
                came_from[nei] = (x, y)

                h = abs(nx - gx) + abs(ny - gy)
                f = new_g + h

                heapq.heappush(pq, (f, nei))

    return None 
