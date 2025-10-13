import collections

def get_map():
    return [
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
        ['#', 'F', '.', '.', 'F', '.', '.', '.', '.', 'P', '#'],
        ['#', '.', '#', '#', '.', '.', '.', '#', '#', '.', '#'],
        ['#', '.', '#', '.', '.', '#', 'F', '.', '#', '.', '#'],
        ['#', '.', '.', 'F', '.', '#', '.', '.', '.', '.', '#'],
        ['#', '.', '.', '#', '#', '#', '#', '#', 'F', '.', '#'],
        ['#', '.', '.', '.', '.', '#', 'F', '.', '.', '.', '#'],
        ['#', '.', '#', '.', 'F', '#', '.', '.', '#', '.', '#'],
        ['#', '.', '#', '#', '.', '.', '.', '#', '#', '.', '#'],
        ['#', '.', 'F', '.', '.', '.', '.', '.', '.', '.', '#'],
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
    ]
    
def print_map(map):
    print("Map:")
    for row in map:
        for col in row:
            print(f"{col} ", end="")
        print()
    return

def food_count(map):
    foods = 0
    for row in map:
        for col in row:
            if col == 'F':
                foods += 1
    return foods

def get_pacman_loc(map):
    for i, row in enumerate(map):
        for j, col in enumerate(row):
            if col == 'P':
                return i, j
    return None

def bfs(map, loc):
    q = collections.deque()
    q.append(loc)
    
    visited = {loc}
    
    rows = len(map)
    cols = len(map[0])
    
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    depth = 0
    curr_level_size = len(q)
    
    while q:
        curr = q.popleft()
        curr_i, curr_j = curr
        
        if map[curr_i][curr_j] == 'F':
            return curr, depth
        
        for d_i, d_j in directions:
            nbr_i = curr_i + d_i
            nbr_j = curr_j + d_j
            nbr = nbr_i, nbr_j
            
            if (0 <= nbr_i < rows and 
                0 <= nbr_j < cols and
                map[nbr_i][nbr_j] != '#' and
                nbr not in visited):
                q.append(nbr)
                visited.add(nbr)
        
        curr_level_size -= 1
        if curr_level_size == 0:
            curr_level_size = len(q)
            depth += 1
        
    return None
    
def get_next_nearest_food(map, loc):
    return bfs(map, loc)

def update_pacman_loc(map, pacman_loc, food_loc):
    curr_i, curr_j = pacman_loc
    next_i, next_j = food_loc
    map[curr_i][curr_j] = '.'
    map[next_i][next_j] = 'P'
    return food_loc

def main():
    map = get_map()
    print_map(map)
    
    pacman_loc = get_pacman_loc(map)
    
    if pacman_loc is None:
        print(f"Pacman not found in the map!\n")
        return
    else:
        print(f"Pacman found at: {pacman_loc}")
    
    foods = food_count(map)
    if foods == 0:
        print(f"No food found in the map!\n")
        return
    else:
        print(f"Food found: {foods}")
    
    total_distance = 0
    while foods > 0:
        next_food, distance = get_next_nearest_food(map, pacman_loc)
        print(f"Next food at: {next_food} at distance {distance}")
        
        pacman_loc = update_pacman_loc(map, pacman_loc, next_food)
        print_map(map)
        
        total_distance += distance
        foods -= 1
        print(f"Food remaining: {foods}\n")
    
    print(f"Total distance covered to eat all food: {total_distance}")
    return
        

if __name__ == '__main__':
    main()
    print("\nExiting...\n")