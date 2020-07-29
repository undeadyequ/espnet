#!/bin/bash


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


function get_sr() {
    cmd="} sox -i $1 | sox --i CB-RV-10-114.wav | grample Rate" | awk -F ':' '{print $2}' | tr -d '[:space:]'
"
output=$(eval "$your_command_string")
}