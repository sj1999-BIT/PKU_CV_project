def cxcywh_to_xyxy(x_center, y_center, box_width, box_height):
    """
    Convert bounding box from center format (x_center, y_center, box_height, box_width)
    to corner format (x1, y1, x2, y2).

    Parameters:
    - x_center: The x-coordinate of the box center.
    - y_center: The y-coordinate of the box center.
    - box_height: The height of the bounding box.
    - box_width: The width of the bounding box.

    Returns:
    - x1: The x-coordinate of the top-left corner of the box.
    - y1: The y-coordinate of the top-left corner of the box.
    - x2: The x-coordinate of the bottom-right corner of the box.
    - y2: The y-coordinate of the bottom-right corner of the box.
    """
    x1 = (x_center - box_width / 2)
    y1 = (y_center - box_height / 2)
    x2 = (x_center + box_width / 2)
    y2 = (y_center + box_height / 2)

    return x1, y1, x2, y2


def xyxy_cxcywh(x1, y1, x2, y2):
    """
    Convert bounding box from corner format (x1, y1, x2, y2) to center format
    (x_center, y_center, box_height, box_width).

    Parameters:
    - x1: The x-coordinate of the top-left corner of the box.
    - y1: The y-coordinate of the top-left corner of the box.
    - x2: The x-coordinate of the bottom-right corner of the box.
    - y2: The y-coordinate of the bottom-right corner of the box.

    Returns:
    - x_center: The x-coordinate of the box center.
    - y_center: The y-coordinate of the box center.
    - box_height: The height of the bounding box.
    - box_width: The width of the bounding box.
    """
    x_center = x1 + (x2 - x1) / 2
    y_center = y1 + (y2 - y1) / 2
    box_height = y2 - y1
    box_width = x2 - x1

    return x_center, y_center, box_height, box_width


def merge_cxcywh(box1, box2):
    """
    Generate a larger bounding box that covers both input boxes.

    Parameters:
    - box1: A tuple (x_center1, y_center1, box_height1, box_width1) representing the first bounding box.
    - box2: A tuple (x_center2, y_center2, box_height2, box_width2) representing the second bounding box.

    Returns:
    - A tuple (x_center, y_center, box_height, box_width) of the new bounding box that covers both.
    """
    # Unpack the input boxes
    x_center1, y_center1, box_width1, box_height1 = box1
    x_center2, y_center2, box_width2, box_height2 = box2

    # Calculate the corners of the first box and second box
    x1_1, y1_1, x2_1, y2_1 = cxcywh_to_xyxy(x_center1, y_center1, box_width1, box_height1)
    x1_2, y1_2, x2_2, y2_2 = cxcywh_to_xyxy(x_center2, y_center2, box_width2, box_height2)

    # Find the minimum and maximum corners to create the bounding box
    # bounded to 0 and 1
    x1_min = max(0, min(x1_1, x1_2))
    y1_min = max(0, min(y1_1, y1_2))
    x2_max = min(1, max(x2_1, x2_2))
    y2_max = min(1, max(y2_1, y2_2))

    # Calculate the center and size of the new bounding box
    x_center, y_center, box_width, box_height = xyxy_cxcywh(x1_min, y1_min, x2_max, y2_max)

    return x_center, y_center, box_height, box_width
