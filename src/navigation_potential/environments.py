import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator

class GridEnvironment:
    """Class representing the planning environment with obstacles, start and target."""
    
    def __init__(self, size=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Create empty grid
        self.size = size
        self.grid = np.zeros((size, size), dtype=bool)
        self.start = np.random.randint(0, size, 2)
        self.target = np.random.randint(0, size, 2)

        self._compute_distance_map()
    
    def _compute_distance_map(self):
        """Compute distance to nearest obstacle."""
        self.distance_map = ndimage.distance_transform_edt(~self.grid)
        
    def is_valid_position(self, position, safety_margin=0.5):
        """Check if a position is valid (inside bounds and away from obstacles)."""
        i, j = int(round(position[0])), int(round(position[1]))
        if 0 <= i < self.size and 0 <= j < self.size:
            return self.distance_map[i, j] > safety_margin
        return False
    
    def get_interp_distance(self, position):
        """Get interpolated distance to obstacles."""
        if not hasattr(self, '_dist_interp'):
            y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
            self._dist_interp = RegularGridInterpolator(
                (np.arange(self.size), np.arange(self.size)), 
                self.distance_map, 
                bounds_error=False, 
                fill_value=0
            )
        return self._dist_interp([position[0], position[1]])[0]

class RandomGridEnvironment(GridEnvironment):
    """A grid environment with random obstacles."""
    def __init__(self, size=100, obstacle_density=0.2, min_obstacle_size=3, max_obstacle_size=10, seed=None):
        """Initialize the environment with random obstacles."""
        super().__init__(size, seed)
        
        self._generate_random_grid(obstacle_density, min_obstacle_size, max_obstacle_size)
        self._compute_distance_map()
    
    def _generate_random_grid(self, obstacle_density, min_obstacle_size, max_obstacle_size):
        """Generate random obstacles and place start and target."""
        # Place random obstacles
        num_obstacles = int(obstacle_density * self.size)
        for _ in range(num_obstacles):
            obstacle_size = np.random.randint(min_obstacle_size, max_obstacle_size)
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            
            # Create circular obstacle
            for i in range(max(0, x-obstacle_size), min(self.size, x+obstacle_size)):
                for j in range(max(0, y-obstacle_size), min(self.size, y+obstacle_size)):
                    if np.sqrt((i-x)**2 + (j-y)**2) <= obstacle_size:
                        self.grid[i, j] = True
        
        # Place start and target points (ensure they're not on obstacles)
        valid_points = np.where(~self.grid)
        idx = np.random.choice(len(valid_points[0]), 2, replace=False)
        
        self.start = (valid_points[0][idx[0]], valid_points[1][idx[0]])
        self.target = (valid_points[0][idx[1]], valid_points[1][idx[1]])
    
    def _compute_distance_map(self):
        """Compute distance to nearest obstacle."""
        self.distance_map = ndimage.distance_transform_edt(~self.grid)
        
    def is_valid_position(self, position, safety_margin=0.5):
        """Check if a position is valid (inside bounds and away from obstacles)."""
        i, j = int(round(position[0])), int(round(position[1]))
        if 0 <= i < self.size and 0 <= j < self.size:
            return self.distance_map[i, j] > safety_margin
        return False
    
    def get_interp_distance(self, position):
        """Get interpolated distance to obstacles."""
        if not hasattr(self, '_dist_interp'):
            y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
            self._dist_interp = RegularGridInterpolator(
                (np.arange(self.size), np.arange(self.size)), 
                self.distance_map, 
                bounds_error=False, 
                fill_value=0
            )
        return self._dist_interp([position[0], position[1]])[0]
class MazeEnvironment(GridEnvironment):
    """A maze-like environment with corridors and dead ends."""
    
    def __init__(self, size=100, corridor_width=5, seed=None):
        """Initialize the maze environment."""
        if seed is not None:
            np.random.seed(seed)
        
        self.size = size
        self.corridor_width = corridor_width
        
        # Start with all walls
        self.grid = np.ones((size, size), dtype=bool)
        
        self._generate_maze()
        self._place_start_target()
        self._compute_distance_map()
    
    def _generate_maze(self):
        """Generate a proper maze using a randomized DFS algorithm."""
        # Calculate the number of cells in the maze
        cell_size = self.corridor_width
        cell_count = (self.size // cell_size) - 1
        if cell_count < 3:
            raise ValueError("Size too small for maze with given corridor width")
        
        # Create a grid for the maze cells (True = wall, False = path)
        # We make this smaller than the final grid and will scale it up later
        cells = np.ones((cell_count, cell_count), dtype=bool)
        
        # Create a visited flag array for the maze generation algorithm
        visited = np.zeros((cell_count, cell_count), dtype=bool)
        
        # Pick a random starting cell
        start_x, start_y = np.random.randint(0, cell_count), np.random.randint(0, cell_count)
        cells[start_y, start_x] = False  # Carve the starting cell
        visited[start_y, start_x] = True
        
        # Stack for backtracking
        stack = [(start_y, start_x)]
        
        # Randomized DFS to create the maze
        while stack:
            current_y, current_x = stack[-1]
            
            # Find unvisited neighbors
            neighbors = []
            for dy, dx in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Left, Down, Right, Up
                ny, nx = current_y + dy * 2, current_x + dx * 2
                if 0 <= ny < cell_count and 0 <= nx < cell_count and not visited[ny, nx]:
                    neighbors.append((ny, nx, dy, dx))
            
            if neighbors:
                # Choose a random unvisited neighbor
                next_y, next_x, dy, dx = neighbors[np.random.randint(0, len(neighbors))]
                
                # Remove the wall between current and next
                wall_y, wall_x = current_y + dy, current_x + dx
                cells[wall_y, wall_x] = False
                
                # Mark the new cell as visited and add to stack
                cells[next_y, next_x] = False
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))
            else:
                # Backtrack
                stack.pop()
        
        # Scale up the maze to the actual grid size
        # Start with all walls
        self.grid = np.ones((self.size, self.size), dtype=bool)
        
        # Carve out the passages based on the cell grid
        for y in range(cell_count):
            for x in range(cell_count):
                if not cells[y, x]:  # If this is a passage
                    # Calculate the corresponding region in the actual grid
                    y_start = y * cell_size + 1
                    y_end = (y + 1) * cell_size
                    x_start = x * cell_size + 1
                    x_end = (x + 1) * cell_size
                    
                    # Carve the passage
                    self.grid[y_start:y_end, x_start:x_end] = False
        
        # Add an entrance and exit to the maze
        # Create openings at the edges
        border = self.corridor_width
        # Find a path cell near the top edge
        for x in range(1, self.size - 1):
            y = border
            if not self.grid[y, x]:
                # Create a path to the edge
                self.grid[0:y, x] = False
                break
        
        # Find a path cell near the bottom edge
        for x in range(1, self.size - 1):
            y = self.size - border - 1
            if not self.grid[y, x]:
                # Create a path to the edge
                self.grid[y:self.size, x] = False
                break
    
    def _place_start_target(self):
        """Place start and target at opposite ends of the maze."""
        # Find free cells near the edges
        top_edge = [(y, x) for y in range(self.corridor_width) 
                   for x in range(self.size) if not self.grid[y, x]]
        bottom_edge = [(y, x) for y in range(self.size - self.corridor_width, self.size) 
                      for x in range(self.size) if not self.grid[y, x]]
        left_edge = [(y, x) for y in range(self.size) 
                    for x in range(self.corridor_width) if not self.grid[y, x]]
        right_edge = [(y, x) for y in range(self.size) 
                     for x in range(self.size - self.corridor_width, self.size) if not self.grid[y, x]]
        
        # Try to place start and target at opposite edges
        candidate_pairs = []
        if top_edge and bottom_edge:
            candidate_pairs.append((top_edge, bottom_edge))
        if left_edge and right_edge:
            candidate_pairs.append((left_edge, right_edge))
        
        if candidate_pairs:
            edge1, edge2 = candidate_pairs[np.random.randint(0, len(candidate_pairs))]
            self.start = edge1[np.random.randint(0, len(edge1))]
            self.target = edge2[np.random.randint(0, len(edge2))]
        else:
            # Fallback: find any two distant free points
            free_points = list(zip(*np.where(~self.grid)))
            if len(free_points) < 2:
                raise ValueError("Not enough free space in maze")
            
            # Find two distant points
            max_dist = 0
            start_pos = target_pos = free_points[0]
            
            # Sample pairs if there are too many points
            if len(free_points) > 100:
                sample_indices = np.random.choice(len(free_points), min(100, len(free_points)), replace=False)
                sample_points = [free_points[i] for i in sample_indices]
            else:
                sample_points = free_points
            
            for i, p1 in enumerate(sample_points):
                for p2 in sample_points[i+1:]:
                    dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                    if dist > max_dist:
                        max_dist = dist
                        start_pos, target_pos = p1, p2
            
            self.start = start_pos
            self.target = target_pos

class NarrowPassageEnvironment(GridEnvironment):
    """An environment with narrow passages between obstacles."""
    
    def __init__(self, size=100, num_passages=2, passage_width=5, seed=None):
        """Initialize the environment with narrow passages."""
        if seed is not None:
            np.random.seed(seed)
        
        self.size = size
        self.grid = np.zeros((self.size, self.size), dtype=bool)
        
        self._generate_narrow_passages(num_passages, passage_width)
        self._place_start_target()
        self._compute_distance_map()
    
    def _generate_narrow_passages(self, num_passages, passage_width):
        """Generate environment with narrow passages."""
        # Clear grid
        self.grid = np.zeros((self.size, self.size), dtype=bool)
        
        # Create a central obstacle with narrow passages
        center = self.size // 2
        radius = self.size // 3
        
        # Create a cross shape in the middle with narrow passages
        # Fill the grid with obstacles except for the cross passages
        for i in range(self.size):
            for j in range(self.size):
                # Leave a horizontal and vertical passage
                if abs(i - center) < passage_width or abs(j - center) < passage_width:
                    self.grid[i, j] = False
                else:
                    # Create large obstacle regions
                    distance_to_center = np.sqrt((i - center)**2 + (j - center)**2)
                    if distance_to_center < radius:
                        self.grid[i, j] = True
        
        # Add some additional narrow passages
        for _ in range(num_passages - 1):  # Already have one passage from the cross
            # Choose random angle
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Calculate passage coordinates
            x1 = int(center + radius * np.cos(angle))
            y1 = int(center + radius * np.sin(angle))
            
            x2 = int(center + (self.size//2) * np.cos(angle))
            y2 = int(center + (self.size//2) * np.sin(angle))
            
            # Ensure points are within bounds
            x1 = max(0, min(self.size-1, x1))
            y1 = max(0, min(self.size-1, y1))
            x2 = max(0, min(self.size-1, x2))
            y2 = max(0, min(self.size-1, y2))
            
            # Draw passage line from center outward
            # Use Bresenham's line algorithm
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            
            while True:
                # Create passage (clear obstacles within passage_width/2)
                for i in range(-passage_width//2, passage_width//2 + 1):
                    for j in range(-passage_width//2, passage_width//2 + 1):
                        ni, nj = y1 + i, x1 + j
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            self.grid[ni, nj] = False
                
                if x1 == x2 and y1 == y2:
                    break
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
        
        # Create narrow constrictions at several points in the passages
        for _ in range(3):
            # Select a point in a passage
            free_points = np.where(~self.grid)
            if len(free_points[0]) == 0:
                continue
                
            idx = np.random.randint(0, len(free_points[0]))
            y, x = free_points[0][idx], free_points[1][idx]
            
            # Check if this is in a passage (not all 8 neighbors are free)
            neighbors_free = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.size and 0 <= nx < self.size and not self.grid[ny, nx]:
                        neighbors_free += 1
            
            # If this is in a passage (not in a large open area)
            if neighbors_free < 9:
                # Create a narrow constriction
                constriction_size = passage_width - 2  # Make it narrower
                for i in range(-constriction_size, constriction_size + 1):
                    for j in range(-constriction_size, constriction_size + 1):
                        if abs(i) == constriction_size or abs(j) == constriction_size:
                            ny, nx = y + i, x + j
                            if 0 <= ny < self.size and 0 <= nx < self.size:
                                self.grid[ny, nx] = True
    
    def _place_start_target(self):
        """Place start and target points ensuring they are connected."""
        # Use BFS to find connected components
        def bfs(start):
            visited = np.zeros_like(self.grid, dtype=bool)
            queue = [start]
            visited[start] = True
            connected = [start]
            
            while queue:
                y, x = queue.pop(0)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.size and 0 <= nx < self.size and 
                            not self.grid[ny, nx] and not visited[ny, nx]):
                        queue.append((ny, nx))
                        visited[ny, nx] = True
                        connected.append((ny, nx))
            return connected
        
        # Find all free spaces
        free_points = list(zip(*np.where(~self.grid)))
        if not free_points:
            raise ValueError("No free space in environment")
        
        # Find largest connected component
        largest_component = []
        while free_points:
            start = free_points[0]
            connected = bfs(start)
            if len(connected) > len(largest_component):
                largest_component = connected
            
            # Remove these points from consideration
            free_points = [p for p in free_points if p not in connected]
        
        # Choose start and target from largest connected component
        if len(largest_component) < 2:
            raise ValueError("Not enough connected free space")
        
        # Find two distant points
        max_dist = 0
        start_pos = target_pos = largest_component[0]
        
        # Sample pairs if there are too many points
        if len(largest_component) > 100:
            sample_indices = np.random.choice(len(largest_component), min(100, len(largest_component)), replace=False)
            sample_points = [largest_component[i] for i in sample_indices]
        else:
            sample_points = largest_component
        
        for i, p1 in enumerate(sample_points):
            for p2 in sample_points[i+1:]:
                dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                if dist > max_dist:
                    max_dist = dist
                    start_pos, target_pos = p1, p2
        
        self.start = start_pos
        self.target = target_pos
class ClutteredEnvironment(GridEnvironment):
    """An environment with many small, randomly placed obstacles."""
    
    def __init__(self, size=100, num_obstacles=100, min_size=2, max_size=8, seed=None):
        """Initialize with many small obstacles."""
        super().__init__(size, seed)
        self._generate_cluttered_environment(num_obstacles, min_size, max_size)
        self._place_start_target()
        self._compute_distance_map()
    
    def _generate_cluttered_environment(self, num_obstacles, min_size, max_size):
        """Generate environment with many small obstacles."""
        for _ in range(num_obstacles):
            # Random position and size
            obstacle_size = np.random.randint(min_size, max_size)
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            
            # Random shape: circle, square, or diamond
            shape_type = np.random.choice(['circle', 'square', 'diamond'])
            
            for i in range(max(0, x-obstacle_size), min(self.size, x+obstacle_size)):
                for j in range(max(0, y-obstacle_size), min(self.size, y+obstacle_size)):
                    if shape_type == 'circle':
                        if np.sqrt((i-x)**2 + (j-y)**2) <= obstacle_size:
                            self.grid[i, j] = True
                    elif shape_type == 'square':
                        self.grid[i, j] = True
                    elif shape_type == 'diamond':
                        if abs(i-x) + abs(j-y) <= obstacle_size:
                            self.grid[i, j] = True
    
    def _place_start_target(self):
        """Place start and target points randomly."""
        valid_points = np.where(~self.grid)
        idx = np.random.choice(len(valid_points[0]), 2, replace=False)
        
        self.start = (valid_points[0][idx[0]], valid_points[1][idx[0]])
        self.target = (valid_points[0][idx[1]], valid_points[1][idx[1]])


class RoomsEnvironment(GridEnvironment):
    """An environment with connected rooms."""
    
    def __init__(self, size=100, door_width=5, wall_thickness=3, seed=None):
        """Initialize with connected rooms."""
        super().__init__(size, seed)
        assert door_width > 0 and wall_thickness > 0, "Door width and wall thickness must be positive."
        self.door_width = door_width
        self.wall_thickness = wall_thickness
        
        self._generate_rooms()
        self._place_start_target()
        self._compute_distance_map()
    
    def _generate_rooms(self):
        """Generate environment with connected rooms."""
        # Draw outer walls
        self.grid[0:self.wall_thickness, :] = True  # Bottom wall
        self.grid[-self.wall_thickness:, :] = True  # Top wall
        self.grid[:, 0:self.wall_thickness] = True  # Left wall
        self.grid[:, -self.wall_thickness:] = True  # Right wall
        # Horizontal divider
        h_divider = self.size // 2
        self.grid[h_divider-self.wall_thickness//2:h_divider+self.wall_thickness//2, :] = True
        
        # Vertical divider
        v_divider = self.size // 2
        self.grid[:, v_divider-self.wall_thickness//2:v_divider+self.wall_thickness//2] = True
        
        # Add doors between rooms
        # Horizontal doors
        h_door1 = np.random.randint(self.wall_thickness*2, v_divider-self.door_width-self.wall_thickness)
        h_door2 = np.random.randint(v_divider+self.wall_thickness, self.size-self.door_width-self.wall_thickness*2)
        self.grid[h_divider-self.wall_thickness//2:h_divider+self.wall_thickness//2, h_door1:h_door1+self.door_width] = False
        self.grid[h_divider-self.wall_thickness//2:h_divider+self.wall_thickness//2, h_door2:h_door2+self.door_width] = False
        
        # Vertical doors
        v_door1 = np.random.randint(self.wall_thickness*2, h_divider-self.door_width-self.wall_thickness)
        v_door2 = np.random.randint(h_divider+self.wall_thickness, self.size-self.door_width-self.wall_thickness*2)
        self.grid[v_door1:v_door1+self.door_width, v_divider-self.wall_thickness//2:v_divider+self.wall_thickness//2] = False
        self.grid[v_door2:v_door2+self.door_width, v_divider-self.wall_thickness//2:v_divider+self.wall_thickness//2] = False
    
        # Add some furniture (small obstacles) in each room
        room_size = self.size // 2
        for i in range(2):
            for j in range(2):
                room_x = j * room_size + room_size // 4
                room_y = i * room_size + room_size // 4
                
                # Add 2-3 obstacles per room
                for _ in range(np.random.randint(2, 4)):
                    obstacle_size = np.random.randint(5, 10)
                    ox = room_x + np.random.randint(0, room_size//2)
                    oy = room_y + np.random.randint(0, room_size//2)
                    
                    self.grid[oy:oy+obstacle_size, ox:ox+obstacle_size] = True
    
    def _place_start_target(self):
        """Place start and target in different rooms."""
        # Define room regions
        room_size = self.size // 2
        rooms = [
            (self.wall_thickness, self.wall_thickness, room_size-self.wall_thickness, room_size-self.wall_thickness),  # Bottom-left
            (self.wall_thickness, room_size, room_size-self.wall_thickness, self.size-self.wall_thickness),       # Bottom-right
            (room_size, self.wall_thickness, self.size-self.wall_thickness, room_size-self.wall_thickness),       # Top-left
            (room_size, room_size, self.size-self.wall_thickness, self.size-self.wall_thickness)             # Top-right
        ]
        
        # Select two different rooms
        room_indices = np.random.choice(len(rooms), 2, replace=False)
        
        # Find free space in each selected room
        start_room = rooms[room_indices[0]]
        target_room = rooms[room_indices[1]]
        
        # Find valid points in start room
        start_y, start_x = np.where(~self.grid[start_room[0]:start_room[2], start_room[1]:start_room[3]])
        if len(start_y) > 0:
            idx = np.random.randint(0, len(start_y))
            self.start = (start_y[idx] + start_room[0], start_x[idx] + start_room[1])
        else:
            # Fallback
            free_spaces = np.where(~self.grid)
            idx = np.random.randint(0, len(free_spaces[0]))
            self.start = (free_spaces[0][idx], free_spaces[1][idx])
        
        # Find valid points in target room
        target_y, target_x = np.where(~self.grid[target_room[0]:target_room[2], target_room[1]:target_room[3]])
        if len(target_y) > 0:
            idx = np.random.randint(0, len(target_y))
            self.target = (target_y[idx] + target_room[0], target_x[idx] + target_room[1])
        else:
            # Fallback
            free_spaces = np.where(~self.grid)
            indices = np.arange(len(free_spaces[0]))
            np.random.shuffle(indices)
            for idx in indices:
                point = (free_spaces[0][idx], free_spaces[1][idx])
                if point != self.start:
                    self.target = point
                    break


class MultiPathEnvironment(GridEnvironment):
    """An environment with multiple possible paths of varying difficulty."""
    
    def __init__(self, size=100, seed=None):
        """Initialize with multiple possible paths."""
        super().__init__(size, seed)

        self._generate_multi_path()
        self._place_start_target()
        self._compute_distance_map()
    
    def _generate_multi_path(self):
        """Generate environment with multiple paths."""
        # Create a central obstacle
        center_x, center_y = self.size // 2, self.size // 2
        obstacle_radius = self.size // 3
        
        # Create a circular obstacle in the center
        for i in range(self.size):
            for j in range(self.size):
                if np.sqrt((i - center_y)**2 + (j - center_x)**2) < obstacle_radius:
                    self.grid[i, j] = True
        
        # Create three pathways around the obstacle
        # Path 1: Short but narrow
        path1_width = self.size // 20
        path1_y = center_y - obstacle_radius - path1_width - 5
        self.grid[path1_y:path1_y+path1_width, :] = False
        
        # Add some small obstacles in path 1
        for _ in range(5):
            ox = np.random.randint(0, self.size - 3)
            oy = np.random.randint(path1_y, path1_y+path1_width - 3)
            obstacle_size = np.random.randint(2, min(4, path1_width-1))
            self.grid[oy:oy+obstacle_size, ox:ox+obstacle_size] = True
        
        # Path 2: Medium length, medium width
        path2_width = self.size // 10
        path2_y = center_y + obstacle_radius + 5
        self.grid[path2_y:path2_y+path2_width, :] = False
        
        # Add some small obstacles in path 2
        for _ in range(3):
            ox = np.random.randint(0, self.size - 5)
            oy = np.random.randint(path2_y, path2_y+path2_width - 5)
            obstacle_size = np.random.randint(3, min(6, path2_width-1))
            self.grid[oy:oy+obstacle_size, ox:ox+obstacle_size] = True
        
        # Path 3: Long but wide (goes all the way around)
        # This is implicitly created by not blocking the top and bottom of the grid
        
        # Add some random obstacles in the open spaces
        for _ in range(10):
            # Don't place obstacles in the central region
            quad = np.random.randint(0, 4)
            if quad == 0:  # Bottom-left
                ox = np.random.randint(0, center_x - obstacle_radius)
                oy = np.random.randint(0, center_y - obstacle_radius)
            elif quad == 1:  # Bottom-right
                ox = np.random.randint(center_x + obstacle_radius, self.size - 10)
                oy = np.random.randint(0, center_y - obstacle_radius)
            elif quad == 2:  # Top-left
                ox = np.random.randint(0, center_x - obstacle_radius)
                oy = np.random.randint(center_y + obstacle_radius, self.size - 10)
            else:  # Top-right
                ox = np.random.randint(center_x + obstacle_radius, self.size - 10)
                oy = np.random.randint(center_y + obstacle_radius, self.size - 10)
            
            obstacle_size = np.random.randint(5, 15)
            self.grid[oy:oy+obstacle_size, ox:ox+obstacle_size] = True
    
    def _place_start_target(self):
        """Place start and target points at opposite sides."""
        # Place start on the left and target on the right
        left_margin = self.size // 10
        right_margin = self.size - left_margin
        
        # Find valid points on the left side
        left_points = [(i, j) for i in range(self.size) for j in range(left_margin) if not self.grid[i, j]]
        right_points = [(i, j) for i in range(self.size) for j in range(right_margin, self.size) if not self.grid[i, j]]
        
        if left_points and right_points:
            self.start = left_points[np.random.randint(0, len(left_points))]
            self.target = right_points[np.random.randint(0, len(right_points))]
        else:
            # Fallback
            free_spaces = np.where(~self.grid)
            idx = np.random.choice(len(free_spaces[0]), 2, replace=False)
            self.start = (free_spaces[0][idx[0]], free_spaces[1][idx[0]])
            self.target = (free_spaces[0][idx[1]], free_spaces[1][idx[1]])
