#!/usr/bin/env python3
import csv
import argparse


def get_author_blizzard2013(train_inventory_f, fiction_name):
    with open(train_inventory_f) as f:
        cf = csv.DictReader(f, fieldnames=["AuthorFirst", "Author", "Title", "HourMinute",
                                           "Hour", "Minute", "HourDecimal", "Prompts", "other"])
        for row in cf:
            title = row["Title"]
            fiction_name = fiction_name.lower()
            title = title.lower().replace(" ", "_")
            if title == fiction_name:
                return row["AuthorFirst"] + " " + row["Author"]
        return "unknow"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inventory', type=str, help='train_inventory_f')
    parser.add_argument('fiction', type=str, help='fiction_name')
    args = parser.parse_args()

    author = get_author_blizzard2013(args.inventory, args.fiction)
    print(author)
