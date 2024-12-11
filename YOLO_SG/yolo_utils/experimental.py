from collections import defaultdict
from sortedcontainers import SortedList
import numpy as np
from yolo_utils.label_utils import *

class IntervalNode:
    def __init__(self, center):
        self.center = center
        self.intervals = []  # (start, end, index) tuples
        self.left = None
        self.right = None


class IntervalTree:
    def __init__(self):
        self.root = None

    def insert(self, start, end, index):
        if not self.root:
            self.root = IntervalNode((start + end) / 2)
        self._insert(self.root, start, end, index)

    def _insert(self, node, start, end, index):
        # Add interval to current node if it crosses center
        if start <= node.center <= end:
            node.intervals.append((start, end, index))
            return

        # Go left or right based on interval position
        if end < node.center:
            if not node.left:
                node.left = IntervalNode((start + end) / 2)
            self._insert(node.left, start, end, index)
        else:
            if not node.right:
                node.right = IntervalNode((start + end) / 2)
            self._insert(node.right, start, end, index)

    def query(self, point):
        """Find all intervals that contain the point"""
        return self._query(self.root, point) if self.root else []

    def _query(self, node, point):
        if not node:
            return []

        result = []
        # Add intervals from current node that contain the point
        for start, end, index in node.intervals:
            if start <= point <= end:
                result.append(index)

        # Recursively search left or right subtree
        if point < node.center:
            result.extend(self._query(node.left, point))
        else:
            result.extend(self._query(node.right, point))

        return result


def efficient_cluster_algo(obj_label_data, pred_label_data):
    """
    O(n log n) clustering algorithm using sweep line and interval trees

    Args:
        obj_label_data: List of [class, cx, cy, w, h] object labels
        pred_label_data: List of [class, cx, cy, w, h] predicate labels

    Returns:
        dict: Mapping predicate indices to lists of overlapping object indices
    """
    # Convert boxes to intervals with indices
    pred_intervals = []  # [(x_start, x_end, y_start, y_end, pred_idx)]
    for i, (_, cx, cy, w, h) in enumerate(pred_label_data):
        x_start, y_start, x_end, y_end = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        pred_intervals.append((x_start, x_end, y_start, y_end, i))

    obj_intervals = []  # [(x_start, x_end, y_start, y_end, obj_idx)]
    for i, (_, cx, cy, w, h) in enumerate(obj_label_data):
        x_start, y_start, x_end, y_end = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        obj_intervals.append((x_start, x_end, y_start, y_end, i))

    # Sort all x-coordinates for sweep line
    events = []
    for x_start, x_end, _, _, i in pred_intervals:
        events.append((x_start, 0, i))  # 0 for predicate start
        events.append((x_end, 2, i))  # 2 for predicate end
    for x_start, x_end, _, _, i in obj_intervals:
        events.append((x_start, 1, i))  # 1 for object start
        events.append((x_end, 3, i))  # 3 for object end

    events.sort()  # O(n log n)

    # Initialize data structures
    active_predicates = set()
    y_tree = IntervalTree()
    result = defaultdict(list)

    # Sweep line algorithm
    for x, event_type, idx in events:
        if event_type == 0:  # Predicate start
            active_predicates.add(idx)
            y_start, y_end = pred_intervals[idx][2:4]
            y_tree.insert(y_start, y_end, idx)

        elif event_type == 1:  # Object start
            if active_predicates:  # Only check if there are active predicates
                y_point = (obj_intervals[idx][2] + obj_intervals[idx][3]) / 2
                overlapping_preds = y_tree.query(y_point)

                # Check IoU for overlapping predicates
                obj_box = obj_label_data[idx][1:]  # cx, cy, w, h
                for pred_idx in overlapping_preds:
                    pred_box = pred_label_data[pred_idx][1:]
                    if calculate_iou(pred_box, obj_box, is_only_extension=True) >= 0.5:
                        result[pred_idx].append(idx)

        elif event_type == 2:  # Predicate end
            active_predicates.remove(idx)

    return dict(result)
