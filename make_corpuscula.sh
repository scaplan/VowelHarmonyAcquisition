#!/bin/bash  

corpusdir='/mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/'

declare -a languageList
languageList=("fin" "hun" "tur" "ger" "eng" "uig" "war")
tininess=1000

cd $corpusdir

for currlang in "${languageList[@]}"; do
    currwordlist="wordlist_"$currlang".txt"
    corpusculum="wordlist_"$tininess"_"$currlang".txt"
    echo "corpusculizing "$currlang
    sort -n -k 1 -t ' ' -r $currwordlist | head -n $tininess > $corpusculum
    echo "corpusculized "$currlang
done
