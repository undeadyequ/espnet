#!/bin/bash

db=/home/Data/blizzard2013_for_test/new_download_part_for_test
author_art=$db/author_arts_final.txt

function get_author() {
  fiction=$1
  fict_author="unkown"
  while IFS=, read author art; do
    if [[ $fiction == $art ]]; then
      fict_author=$author
    fi
  done < $author_art
  echo $fict_author
}

fiction="the_emerald_city_of_oz"
spk=$(get_author $fiction)
echo $spk