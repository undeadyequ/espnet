#!/bin/bash

#  Create (Arts, Speaker) dictionary (structure: AuthorFirst Author	Title ...)
#inventory=$db/training_inventory.csv
#spk_arr=($(cut -d "," -f 1 $inventory))
#arts_arr=($(cut -d "," -f 3 $inventory))
#for name in "${spk_arr[@]}"; do
#  name=${name/,/ }
#  echo $name
#done
##echo ${spk_arr[@]} ${arts_arr[@]}  # debug
#declare -A arts_spk
#for i in $(seq 0 ${#arts_arr[@]}); do
# arts_spk[${arts_arr[i]}]=spk_arr[i]
#done