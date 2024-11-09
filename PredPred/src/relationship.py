

class Bounds:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

    @staticmethod
    def from_center_size(center_x, center_y, w, h):
        return Bounds(
            center_x - w * 0.5,
            center_y - h * 0.5,
            center_x + w * 0.5,
            center_y - h * 0.5,
        )

    @staticmethod
    def from_corner_size(x, y, w, h):
        return Bounds(x, y, x + w, y + h)

    def center(self):
        return [(self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5]

    def size(self):
        return [abs(self.x1 - self.x2), abs(self.y1 - self.y2)]

    def union(self, other):
        return Bounds(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2),
        )

    def normalize(self, other):
        u = self.union(other)
        s = Bounds(
            (self.x1 - u.x1) / u.size()[0],
            (self.y1 - u.y1) / u.size()[1],
            (self.x2 - u.x1) / u.size()[0],
            (self.y2 - u.y1) / u.size()[1],
        )
        o = Bounds(
            (other.x1 - u.x1) / u.size()[0],
            (other.y1 - u.y1) / u.size()[1],
            (other.x2 - u.x1) / u.size()[0],
            (other.y2 - u.y1) / u.size()[1],
        )
        return [s, o]


class Relationship:
    def __init__(self,
                 obj_id,
                 obj_name,
                 obj_bounds: Bounds,
                 subj_id,
                 subj_name,
                 subj_bounds: Bounds,
                 rel_id,
                 rel_name
                 ):
        o, s = obj_bounds.normalize(subj_bounds)
        self.obj_id = obj_id
        self.obj_name = obj_name
        self.obj_bounds = o
        self.subj_id = subj_id
        self.subj_name = subj_name
        self.subj_bounds = s
        self.rel_id = rel_id
        self.rel_name = rel_name
