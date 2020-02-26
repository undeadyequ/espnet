#!/bin/bash

# Copyright 2020 University of Tokyo (Luo Xuan)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Blizzard Structure
# Segment_version_wav1/arts_name/a.wav   => ./BC2013_segmented_v0_wav1/emma/CB-EM-54-136.wav
# Segment_version_txt1/arts_anme/a.txt
db=$1
data_dir=$2
trans_type=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_dir> <trans_type>"
    exit 1
fi

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text_ori=${data_dir}/text_ori
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${spk2utt} ] && rm ${spk2utt}
[ -e ${text_ori} ] && rm ${text_ori}
[ -e ${text} ] && rm ${text}

# make wav.scp, utt2spk, txt, spk2utt
# 1. show number of wav (83668)
find $db -name "*.wav" | wc -l

# 2. Create Arts - Speaker dictionary
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

#  AuthorFirst	Author	Title ...
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

# 3. Create wav.scp utt2spk  text_ori
lastfiction=""
lastspk=""
find_delim="/"
id_field=$(find ${db} -name "*.wav" | head -1 | awk -F"${find_delim}" '{print NF}')

# sort -t/ -k${id_field}

t1=$SECONDS
echo "id_field:" $id_field
find ${db} -name "*.wav" | sort |  while read -r wav_f; do    # ??? Should sort by id not by path
  #t0=$SECONDS
  id=$(basename $wav_f | sed -e "s/\.[^\.]*$//g")
  fiction=$([[ $wav_f =~ .*/(.*)/.*wav ]] && echo ${BASH_REMATCH[1]})
  txt_f=$(echo $wav_f | sed -e "s/_wav/_txt/g" | sed -e "s/\.wav/\.txt/g")
  txt=$(head -n 1 $txt_f)

  #spk=$(local/get_author.py $db/training_inventory.csv $fiction)
  if [[ $lastfiction == "" ]]; then
    spk=$(get_author $fiction)
    lastfiction=$fiction
    lastspk=$spk
  fi
  if [[ $lastfiction == $fiction ]]; then
    spk=$lastspk
  else
    spk=$(get_author $fiction)
    lastfiction=$fiction
    lastspk=$spk
  fi
  #echo $(($t1-$t0)) $(( $t2-$t1))
  echo $spk"_"$id $wav_f >> ${scp}_tmp
  echo $spk"_"$id"|"$txt >> ${text_ori}_tmp
  #echo $id $id"_"$spk >> $utt2spk
  echo $spk"_"$id $spk >> ${utt2spk}_tmp
done
t2=$SECONDS
echo

echo "finished indexing: " $(($t2-$t1))s

# 3.1 sort file
cat ${scp}_tmp | sort -k1 -g > $scp
cat ${text_ori}_tmp | sort -k1 -g > $text_ori
cat ${utt2spk}_tmp | sort -k1 -g > $utt2spk

rm ${scp}_tmp
rm ${text_ori}_tmp
rm ${utt2spk}_tmp
echo "finished sorting by speaker_id"

# 4. clean text_ori to text
local/clean_text.py $text_ori $trans_type >> $text

# 5. Create spk2utt
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}_bk
cat ${spk2utt}_bk | sort -k1 > ${spk2utt}
rm ${spk2utt}_bk

echo "finished making wav.scp, utt2spk, text, spk2utt."
