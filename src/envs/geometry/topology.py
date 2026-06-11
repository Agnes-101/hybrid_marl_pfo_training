import math

def generate_hex_positions(num_bs, width=100.0, height=100.0, min_distance_from_center=30.0,  # Minimum distance for small cells
    enforce_center=True):
        """
        Generate base station positions with the macro cell at center (width/2, height/2)
        and small cells arranged in optimal positions around it.
        """
        if num_bs < 1:
            return []
        
        # Always place the first base station (macro cell) at the center
        center = (width/2, height/2)
        positions = [center]
        
        if num_bs == 1:
            return positions
        
        # For small cells, use optimal positioning instead of standard hexagonal grid
        # Calculate the effective radius - half of the smaller dimension
        effective_radius = min(width, height) / 2 * 0.65  # 65% of half-width for optimal small cell placement
        
        # Generate positions on concentric rings around the center
        remaining_positions = []
        
        # First ring - optimal for up to 6 small cells
        if num_bs <= 7:  # 1 macro + 6 small cells
            angle_step = 2 * math.pi / (num_bs - 1)
            for i in range(num_bs - 1):
                angle = i * angle_step
                x = center[0] + effective_radius * math.cos(angle)
                y = center[1] + effective_radius * math.sin(angle)
                remaining_positions.append((x, y))
        else:
            # First ring - 6 cells
            for i in range(6):
                angle = i * math.pi / 3
                x = center[0] + effective_radius * math.cos(angle)
                y = center[1] + effective_radius * math.sin(angle)
                remaining_positions.append((x, y))
            
            # If more cells are needed, add additional rings with increasing radius
            remaining_cells = num_bs - 7  # -1 for macro, -6 for first ring
            if remaining_cells > 0:
                # Second ring
                second_ring_radius = effective_radius * 1.8
                cells_in_second_ring = min(12, remaining_cells)
                angle_step = 2 * math.pi / cells_in_second_ring
                
                for i in range(cells_in_second_ring):
                    angle = i * angle_step + (angle_step / 2)  # offset to stagger from first ring
                    x = center[0] + second_ring_radius * math.cos(angle)
                    y = center[1] + second_ring_radius * math.sin(angle)
                    remaining_positions.append((x, y))
                
                # If still more cells needed, fall back to hexagonal grid for the rest
                remaining_cells -= cells_in_second_ring
                if remaining_cells > 0:
                    # Calculate standard hexagonal grid positions as before
                    area_per_bs = (width * height) / num_bs
                    pitch = math.sqrt((2 * area_per_bs) / math.sqrt(3))
                    pitch = min(pitch, min(width, height) / 2)
                    pitch = max(pitch, 5.0)
                    
                    hex_height = pitch * math.sin(math.radians(60))
                    n_cols = int(math.ceil(width / pitch)) + 2
                    n_rows = int(math.ceil(height / hex_height)) + 2
                    
                    x_offset = (width - (n_cols-1) * pitch) / 2
                    y_offset = (height - (n_rows-1) * hex_height) / 2
                    
                    # Generate grid positions
                    grid_positions = []
                    for row in range(n_rows):
                        y = y_offset + row * hex_height
                        x_start = x_offset + (pitch/2 if row % 2 else 0)
                        
                        for col in range(n_cols):
                            x = x_start + col * pitch
                            pos = (x, y)
                            # Skip the center and any positions too close to already placed cells
                            if 0 <= x <= width and 0 <= y <= height and pos != center:
                                min_distance = min([math.hypot(x-p[0], y-p[1]) for p in positions + remaining_positions], default=float('inf'))
                                if min_distance > pitch * 0.7:
                                    grid_positions.append(pos)
                    
                    # Sort remaining grid positions by distance from center
                    grid_positions.sort(key=lambda p: math.hypot(p[0]-center[0], p[1]-center[1]))
                    
                    # Add as many as needed
                    remaining_positions.extend(grid_positions[:remaining_cells])
        
        # Ensure all positions are within bounds
        remaining_positions = [(max(0, min(width, x)), max(0, min(height, y))) for x, y in remaining_positions]
        
        # Return the center followed by the optimally placed small cells
        return positions + remaining_positions[:num_bs-1]