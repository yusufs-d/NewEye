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
