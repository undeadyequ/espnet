import os
with open("author_arts_clean.txt", "w+") as f_new:
    with open("author_arts.txt", "r") as f:
        for l in f.readlines():
            author, arts = l.split(",")
            author = author.strip().lower().replace(" ", "_")
            arts = arts.strip().lower().replace(" ", "_")
            f_new.write(author + "," + arts + "\n")
