import json
import sys
import argparse
import matplotlib.pyplot as plt
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    help="Path to the relationships.json file",
    required=True
)
parser.add_argument(
    "--output",
    help="Directory for the output graphs",
    required=True
)


class DataSetStats:
    def __init__(self, args):
        self.ids = {}

        self.preds = []
        self.triples = []
        self.counts = []

        self.args = args
        with open(args.input, "r") as f:
            objs = json.load(f)
            for i, img in enumerate(objs):
                for rel in img["relationships"]:
                    pred = rel["predicate"]
                    obj = rel["object"]["name"]
                    subj = rel["subject"]["name"]

                    if pred not in self.ids:
                        id = len(self.preds)
                        self.ids[pred] = id
                        self.triples.append([])
                        self.counts.append(0)
                        self.preds.append(pred)

                    id = self.ids[pred]
                    self.triples[id].append((obj, subj))
                    self.counts[id] += 1

                if i % 2000 == 0:
                    logging.info(f"{i}/{len(objs)} Images read")

            logging.info("Finished reading the dataset")
            logging.info(f"==== {sum(self.counts)} triples")
            logging.info(f"==== {len(self.preds)} predicates")

    # Draw a distribution of predicate frequency
    def draw_histogram(self):
        plt.title("Distribution of predicate frequencies")
        plt.xlabel("Frequency of predicates")
        plt.ylabel("Share of predicates")
        plt.yscale("log")
        plt.hist(self.counts, 100)
        plt.show()


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    ds = DataSetStats(args)
    ds.draw_histogram()
