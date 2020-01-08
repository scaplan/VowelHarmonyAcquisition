#!/bin/bash  

outputdir="outputs"
mkdir -p $outputdir #make output dir if it doesn't already exist
wordlistsourcedir='Wordlists/'
vowellistsourcedir='Vowels/'
#wordlistsourcedir='/home1/s/spcaplan/Dropbox/penn_CS_account/UPENN_LORELEI/utils/'

declare -a languageList
tininess=""
#tininess="500_"
languageList=("hun" "tur" "ger" "eng" "fin")
#languageList=("hun" "tur" "ger" "eng" "fin" "uig" "est")

#Generate data sets without word segmentation
#gen finnish unseg
#python wikitext_to_unsegsentences.py /mnt/nlpgridio2/nlp/users/lorelei/wikipedia-xml/text/fiwiki-20160701-pages-articles.xml.bz2_text fin_unsegall.txt
#head -n 78846 /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_finunsegshort.txt > /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_finunseg.txt
#gen turkish unseg
#python wikitext_to_unsegsentences.py /mnt/nlpgridio2/nlp/users/lorelei/wikipedia-xml/text/trwiki-20160701-pages-articles.xml.bz2_text /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_turunsegall.txt
#head -n 58764 /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_turunsegall.txt > /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_turunseg.txt
#gen german unseg
#python wikitext_to_unsegsentences.py /mnt/nlpgridio2/nlp/users/lorelei/wikipedia-xml/text/dewiki-20160701-pages-articles.xml.bz2_text /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_gerunsegall.txt
#head -n 100000 /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_gerunsegall.txt > /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_gerunseg.txt
#gen english unseg
#python wikitext_to_unsegsentences.py flatland.txt /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_engunsegall.txt
#head -n 100000 /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_engunsegall.txt > /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_engunseg.txt
#gen warlpiri unseg
#python wikitext_to_unsegsentences.py /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/raw_warlpiri.txt /mnt/nlpgridio2/nlp/users/lorelei/vowel-harmony-data/wordlist_warunseg.txt

for currLanguage in "${languageList[@]}"; do
	currwordlist="$wordlistsourcedir""wordlist_"$tininess$currLanguage".txt"
	echo $currwordlist
	NERoutputfile=$outputdir/$tininess$currLanguage"_NER_output.txt"
	NERoutputfileLocal=$outputdir/$tininess$currLanguage"_NER_output_local.txt"
	NERoutputfileAntilocal=$outputdir/$tininess$currLanguage"_NER_output_antilocal.txt"
	vowellist=$vowellistsourcedir"harmony_detector_testvowels_"$currLanguage".txt"
	echo "Calculating $tininess$currLanguage Harmony Clusters and NER Candidates"
	python harmony_detector.py $currwordlist $NERoutputfile $vowellist

	#with features mapping
	#python harmony_detector.py $currwordlist $NERoutputfile $vowellist --features "harmony_detector_vowelfeatures_"$currLanguage".txt" --nozipf -c local

	python harmony_detector.py $currwordlist $NERoutputfile $vowellist > $outputdir/$tininess$currLanguage"_harmony_stats_global_simple.txt" 
#	python harmony_detector.py $currwordlist $NERoutputfile $vowellist --features "harmony_detector_vowelfeatures_"$currLanguage".txt" > $outputdir/$tininess$currLanguage"_harmony_stats_global_simple.txt" 
#	sort -n -k 3 -t ',' -r $NERoutputfile > $outputdir/$tininess$currLanguage"_NER_output_sorted.txt"

#	python harmony_detector.py $currwordlist $NERoutputfileLocal $vowellist "-c local" > $outputdir/$tininess$currLanguage"_harmony_stats_local_simple_tp.txt"
#	sort -n -k 3 -t ',' -r $NERoutputfileLocal > $outputdir/$tininess$currLanguage"_NER_output_local_sorted.txt"

#	python harmony_detector.py $currwordlist $NERoutputfileAntilocal $vowellist "-c antilocal" > $outputdir/$tininess$currLanguage"_harmony_stats_antilocal_simple_tp.txt"
#	sort -n -k 3 -t ',' -r $NERoutputfileAntilocal > $outputdir/$tininess$currLanguage"_NER_output_antilocal_sorted.txt"

done
exit

