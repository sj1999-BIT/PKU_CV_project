from dataset import DataSet
import logging


def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info("Started")

    ds = DataSet(args["rels_json_file_path"])

    logging.info("Finished")


if __name__ == "__main__":
    main({
        "rels_json_file_path": "relationships.json"
    })
