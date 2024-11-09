import relationship as rel
import json
import logging


class DataSet:
    def __init__(self, rels_json_file_path):
        logging.info(f"Reading the dataset at '{rels_json_file_path}'")
        self.file_path = rels_json_file_path
        f = open(rels_json_file_path, "r")
        js = json.load(f)

        # {
        #     'predicate': 'with',
        #      'object': {
        #         'name': 'windows',
        #         'h': 148,
        #         'object_id': 3798579,
        #         'synsets': ['window.n.01'],
        #         'w': 173,
        #         'y': 4,
        #         'x': 602
        #     },
        #     'relationship_id': 4265927,
        #     'synsets': [],
        #     'subject': {
        #         'name': 'building',
        #         'h': 536,
        #         'object_id': 1058508,
        #         'synsets': ['building.n.01'],
        #         'w': 218,
        #         'y': 2,
        #         'x': 1,
        #     }
        # }
        self.data = []
        image_count = len(js)
        i = 0
        for pic in js:
            for obj in pic["relationships"]:
                self.data.append(
                    rel.Relationship(
                        int(obj["object"]["object_id"]),
                        obj["object"]["name"],
                        rel.Bounds.from_corner_size(
                            int(obj["object"]["x"]),
                            int(obj["object"]["y"]),
                            int(obj["object"]["w"]),
                            int(obj["object"]["h"]),
                        ),
                        int(obj["subject"]["object_id"]),
                        obj["subject"]["name"],
                        rel.Bounds.from_corner_size(
                            int(obj["subject"]["x"]),
                            int(obj["subject"]["y"]),
                            int(obj["subject"]["w"]),
                            int(obj["subject"]["h"]),
                        ),
                        int(obj["relationship_id"]),
                        obj["predicate"],
                    )
                )

            if (i % 200 == 0):
                logging.info(
                    f"Read {i+1}/{image_count} images ({len(self.data)}) relationships")
            i += 1

        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
