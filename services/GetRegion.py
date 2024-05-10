def get_region(rectangle, width, height):
    xmin, ymin, xmax, ymax = rectangle
    middle_x = (xmin + xmax) // 2
    middle_y = (ymin + ymax) // 2

    if middle_x < width // 2:
        if middle_y < height // 2:
            return 1
        else:
            return 3
    else:
        if middle_y < height // 2:
            return 2
        else:
            return 4

def get_region_with_middle(rectangle, width, height, middle_width_percent=30):
    xmin, ymin, xmax, ymax = rectangle
    middle_x = (xmin + xmax) // 2
    middle_y = (ymin + ymax) // 2

    middle_width = (middle_width_percent / 100) * width
    left_middle_boundary = (width - middle_width) // 2
    right_middle_boundary = (width + middle_width) // 2

    if left_middle_boundary <= middle_x <= right_middle_boundary:
        return 'middle'
    
    elif middle_x < width // 2:
        if middle_y < height // 2:
            return 1 
        else:
            return 3
    else:
        if middle_y < height // 2:
            return 2  
        else:
            return 4

