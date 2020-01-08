# Automatic Vowel Harmony Detection
https://mindmodeling.org/cogsci2018/papers/0281/0281.pdf

requires `numpy` and `scipy`

To run on all provided data sets,
```
./runHarmonyDetector.sh
```

To run on the language of your choice,
```
python2 harmony_detector.py <wordlist or unsegmented text> <harmony scoring file> <vowel list> > <output analysis file>
```
