import collections

def get_map():
    return [
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
        ['.', 'F', '.', '.', 'F', '.', '.', '.', '.', 'P', '.'],
        ['.', '.', '#', '#', '.', '.', '.', '#', '#', '.', '.'],
        ['#', '.', '#', '.', '.', '#', 'F', '.', '#', '.', '#'],
        ['#', '.', '.', 'F', '.', '#', '.', '.', '.', '.', '#'],
        ['.', '.', '.', '#', '#', '#', '#', '#', 'F', '.', '.'],
        ['#', '.', '.', '.', '.', '#', 'F', '.', '.', '.', '#'],
        ['#', '.', '#', '.', 'F', '#', '.', '.', '#', '.', '#'],
        ['.', '.', '#', '#', '.', '.', '.', '#', '#', '.', '.'],
        ['.', '.', 'F', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
    ]

def print_map(map):
    print(f"Current map:")
    for row in map:
        for col in row:
            print(f"{col} ", end="")
        print()

def find_pacman(map):
    for i, row in enumerate(map):
        for j, col in enumerate(row):
            if col == 'P':
                return i, j
    return None

def count_foods(map):
    foods = 0
    for row in map:
        for col in row:
            if col == 'F':
                foods += 1  
    return foods

def bfs(map, loc):
    rows = len(map)
    cols = len(map[0])
    
    q = collections.deque()
    q.append((loc, [loc]))
    
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    visited = {loc}
    
    while q:
        curr, path = q.popleft()
        curr_i, curr_j = curr
        
        if map[curr_i][curr_j] == 'F':
            return curr, path
        
        for d_i, d_j in directions:
            nbr_i, nbr_j = curr_i + d_i, curr_j + d_j
            nbr_i, nbr_j = (nbr_i + rows) % rows, (nbr_j + cols) % cols
            nbr = nbr_i, nbr_j
            
            if (map[nbr_i][nbr_j] != '#' and
                nbr not in visited):
                nbr_path = path + [nbr]
                q.append((nbr, nbr_path))
                visited.add(nbr)
                
    return None

def update_map(map, pacman_loc, food_loc):
    loc_i, loc_j = pacman_loc
    map[loc_i][loc_j] = '.'
    
    food_i, food_j = food_loc
    map[food_i][food_j] = 'P'
    
    return food_loc

def eat_foods(map, pacman_loc, foods):
    total_distance = 0
    
    while foods > 0:
        next_food, path = bfs(map, pacman_loc)
        if next_food is None:
            break
        
        distance = len(path) - 1
        total_distance += distance
        print(f"Next food found at {next_food}")
        print(f"Path: {path}")
        print(f"distance: {distance}")
        pacman_loc = update_map(map, pacman_loc, next_food)
        
        foods -= 1
        print_map(map)
        print()
    
    print(f"All foods are eaten, total distance {total_distance}")
    return
    
def main():
    map = get_map()
    print_map(map)
    
    pacman_loc = find_pacman(map)
    if pacman_loc is None:
        print(f"No pacman is found!")
        return
    else:
        print(f"Pacamn found at: {pacman_loc}")
    
    foods = count_foods(map)
    if foods == 0:
        print(f"No food in the map!")
        return
    else:
        print(f"{foods} in the map!")
    
    print()
    eat_foods(map, pacman_loc, foods)

if __name__ == '__main__':
    main()
    print("\nExiting...\n")