#!/usr/bin/python -tt

from collections import defaultdict
from math import fabs, log
from numpy import array
import scipy.cluster.vq as scipy
import argparse

DEFAULT_VOWELS = set(['a','e','i','o','u','A','E','I','O','U'])
FREQUENCY_PROPORTION_LOWERBOUND = 0.0000001
GLOBAL = "GLOBAL"
LOCAL = "LOCAL"
ANTILOCAL = "ANTILOCAL"
CONTEXTTYPE = GLOBAL
WINDOWSIZE = 1
ZIPF = True

THRESHOLD_C = 0.5
SIMPLETHRESHOLD = lambda n : THRESHOLD_C/n
FANCYTHRESHOLD = lambda n : (THRESHOLD_C + (49.0/(80*n)) - (1.0/48))/n # assuming numerator should be 0.55 when n = 3 and 0.3 when n = 15 (derived from Hungarian, Charles Incident Language, y=mx+b)
THRESHOLD = SIMPLETHRESHOLD
#ns = range(16)[3:]
#print map(lambda n : round(FANCYTHRESHOLD(n),4), ns)



def load_wordlist(wordlistfile, vowelmap):
	worddict = {}
	with open(wordlistfile, 'r') as f:
		for line in f:
			currline = line.decode('utf8').strip()
			currline = currline.replace(',','\t').replace(' ', '\t')
			linecomponents = currline.split('\t')
			currcount = 1
			if len(linecomponents) == 2:
				currcount = int(linecomponents[0])
                        word = linecomponents[-1]
                        #map all vowels in the vowel map; useful for removing features
                        mappedword = []
                        for char in word:
                            try:
                                mappedword.append(vowelmap[char])
                            except KeyError:
                                mappedword.append(char)
			worddict["".join(mappedword)] = currcount

	return worddict

def eval_proposed_harmony(wordfreqdict, vowelset, harmonysetdict):

    def eval_tolerance(epsilon, n):
        return epsilon <= talk_to_charles(n)

    def talk_to_charles(n):
        if n == 0:
            return sys.maxint
        else:
            return float(n)/log(n)

    topwordsbyfreq = sorted(wordfreqdict.iteritems(), key=lambda (k,v) : (float(v),k), reverse=True)[:len(wordfreqdict)]

    print "\nEvaluating on %s most frequent words" % len(topwordsbyfreq)
    print type(harmonysetdict)
    bigtups = set()
    for vowel, setnum in harmonysetdict.iteritems():
        epsilon = 0
        n = 0
        for word, freq in topwordsbyfreq:
            wordvowelsonly = remove_nonvowels(word, vowelset)
            myvowelcount = wordvowelsonly.count(vowel)
            if myvowelcount == 0:
                continue

            comparisons, violations = eval_disharmony_pairs_on_word(wordvowelsonly, vowel, harmonysetdict)
            n += len(comparisons)*freq
            epsilon += len(violations)*freq
        bigtups.add((vowel.encode('utf8'), epsilon, n, talk_to_charles(n), eval_tolerance(epsilon, n)))


    for bigtup in bigtups:
    	print "vowel: %s\tepsilon: %s\tn: %s\tn/ln(n): %s\ttolerable: %s" % bigtup


    #call function to prune out most disharmonic words
    worddisharmonydict = {}
    for word in wordfreqdict:
        wordvowellen = len(remove_nonvowels(word, vowelset))
        if wordvowellen:
            worddisharmonydict[word] = evaluate_adherence_to_harmony(word,harmonysetdict)/float(wordvowellen)
    percenttoremove = 10
    topwordsbydisharmony = set(map(lambda (k,v) : k, sorted(worddisharmonydict.iteritems(), key=lambda (k,v) : (float(v),k), reverse=True)[int(len(wordfreqdict)*(5/100.0)):]))
    newwordfreqdict = {k:v for k,v in wordfreqdict.iteritems() if k in topwordsbydisharmony}
    return newwordfreqdict

def eval_disharmony_pairs_on_word(wordvowelsonly, myvowel, harmonysetdict):
    
    comparisons = []
    violations = []
    harmonysetdict = dict(harmonysetdict)

    for i, vowel in enumerate(wordvowelsonly):
        if i == 0:
            continue
        neighbor = wordvowelsonly[i-1]
        if (vowel == myvowel or neighbor == myvowel):
            comparisons.append((neighbor,vowel))
            try:
                vowelsetnum = harmonysetdict[vowel]
                neighborsetnum = harmonysetdict[neighbor]
            except KeyError:
                continue
            if (vowelsetnum != neighborsetnum):
                violations.append((neighbor,vowel))
    return comparisons, violations

def remove_nonvowels(word, vowelset, getmap=False):
	wordvowelsonly = []
        vowelmapping = {} #from vowelsonly to original word indices
	for i, char in enumerate(word):
		if char in vowelset:
			wordvowelsonly.append(char)
                        vowelmapping[len(wordvowelsonly)-1] = i
        if getmap:
            return ''.join(wordvowelsonly), vowelmapping
	return ''.join(wordvowelsonly)

def remove_nonharmonizingvowels(word, harmonysetdict):
	wordharmvowelsonly = []
	for i, char in enumerate(word):
		if char in harmonysetdict:
			wordharmvowelsonly.append(char)
	return ''.join(wordharmvowelsonly)


def remove_windowsize_words(worddict, vowelset, windowsize):
    minnumvowels = (windowsize*2)+1 #window * 2 + 1
    trimmedworddict = {}
    for word,value in worddict.iteritems():
        vowelcount = 0
        for char in word:
            if char in vowelset:
                vowelcount += 1
        if vowelcount >= minnumvowels:
            trimmedworddict[word] = value
    return trimmedworddict

def remove_zipfian_tail(worddict):
	trimmedworddict = {}
	totaltokencount = sum (worddict.values())
	for word, freq in worddict.iteritems():
		if (float(freq) / totaltokencount) > FREQUENCY_PROPORTION_LOWERBOUND:
			trimmedworddict[word] = freq
	print 'Trimmed tail from %s to %s, that\'s a reduction of %f! \n'%(len(worddict), len(trimmedworddict), float(len(worddict) - len(trimmedworddict)) / len(worddict))
	return trimmedworddict


def calculate_vowel_conditional_probabilities_global(worddict, vowelset):
	vowelcondfreqdict = defaultdict(lambda : defaultdict(int)) # Assuming the form F(A|B), A is the initial key, B is the nested key, and F(A|B) is the value. i.e. {A:{B:F(A|B)}}
	vowelfreqdict = defaultdict(int)

	for currword, currfreq in worddict.iteritems():
            currwordvowelsonly = remove_nonvowels(currword, vowelset)
            for i,condvowel in enumerate(currwordvowelsonly):
                vowelfreqdict[condvowel] += currfreq
                for j,keyvowel in enumerate(currwordvowelsonly):
                    if i == j:
                        continue
                    vowelcondfreqdict[keyvowel][condvowel] += currfreq
	
	return vowelfreqdict, convert_vowel_conditional_frequencies_to_probabilities(vowelfreqdict, vowelcondfreqdict)

def calculate_vowel_conditional_probabilities_local(worddict, vowelset, windowsize):
	vowelcondfreqdict = defaultdict(lambda : defaultdict(int)) # Assuming the form F(A|B), A is the initial key, B is the nested key, and F(A|B) is the value. i.e. {A:{B:F(A|B)}}
	vowelfreqdict = defaultdict(int)

	for currword, currfreq in worddict.iteritems():
		currwordvowelsonly = remove_nonvowels(currword, vowelset)
		paddingstring = "".join([" "]*windowsize)
		paddedvowelonly = paddingstring + currwordvowelsonly + paddingstring
		localvowelenvironments = []
		for index, value in enumerate(paddedvowelonly):
			if index >= windowsize and index < (len(currwordvowelsonly) + windowsize):
				currvowelenvironment = paddedvowelonly[index-windowsize : index+windowsize+1]
				localvowelenvironments.append(currvowelenvironment.strip())
		
		for currenvironment in localvowelenvironments:
			for i,condvowel in enumerate(currenvironment):
				vowelfreqdict[condvowel] += currfreq
				for j,keyvowel in enumerate(currenvironment):
					if i == j:
						continue
					vowelcondfreqdict[keyvowel][condvowel] += currfreq
	
	return vowelfreqdict, convert_vowel_conditional_frequencies_to_probabilities(vowelfreqdict, vowelcondfreqdict)

def calculate_vowel_conditional_probabilities_antilocal(worddict, vowelset, windowsize):
	vowelcondfreqdict = defaultdict(lambda : defaultdict(int)) # Assuming the form F(A|B), A is the initial key, B is the nested key, and F(A|B) is the value. i.e. {A:{B:F(A|B)}}
	vowelfreqdict = defaultdict(int)

	for currword, currfreq in worddict.iteritems():
		currwordvowelsonly, vowelmap = remove_nonvowels(currword, vowelset, getmap=True)
		for i,condvowel in enumerate(currwordvowelsonly):
			vowelfreqdict[condvowel] += currfreq
                        originalindex = vowelmap[i]
			for j,keyvowel in enumerate(currwordvowelsonly):
                            originalcontextindex = vowelmap[j]
                            if fabs(originalindex-originalcontextindex) <= windowsize:
                                continue
                            vowelcondfreqdict[keyvowel][condvowel] += currfreq

	return vowelfreqdict, convert_vowel_conditional_frequencies_to_probabilities(vowelfreqdict, vowelcondfreqdict)


def convert_vowel_conditional_frequencies_to_probabilities(vowelfreqdict, vowelcondfreqdict):
	
	vowelcondprobdict = defaultdict(lambda : {})
	vowelprobdict = {}

	attestedvowels = vowelcondfreqdict.keys()
	totalvowelcount = sum(vowelfreqdict.values())
#        for vowel in attestedvowels: #we're doing this to initialize the vowel prob dict to all 0s. That way later on all are vectors are the same length
#            vowelprobdict[vowel] = 0
	for vowel, freq in vowelfreqdict.iteritems():
		vowelprobdict[vowel] = float(freq) / totalvowelcount

	for keyvowel, nestvoweldict in vowelcondfreqdict.iteritems():
		# Normalization loop
		normalizednestedvowelfreqdict = {}
		for nestedvowel, freq in nestvoweldict.iteritems():
			normalizednestedvowelfreqdict[nestedvowel] = float(freq) / vowelprobdict[nestedvowel] # F(K|N) / P(N)

		totalkeyvowelcount = sum(normalizednestedvowelfreqdict.values())
		# Prob calc loop
		for nestedvowel, normfreq in normalizednestedvowelfreqdict.iteritems():
			vowelcondprobdict[keyvowel][nestedvowel] = (normfreq / totalkeyvowelcount) #round((normfreq / totalkeyvowelcount) ,6)
		

        padded_vowelcondprobdict = {}
	for keyvowel, nestedvoweldict in vowelcondprobdict.iteritems():
		for attestedvowel in attestedvowels:
			if attestedvowel not in nestedvoweldict:
				nestedvoweldict[attestedvowel] = 0.0
                padded_vowelcondprobdict[keyvowel] = nestedvoweldict
                

        print "Mutual Information:"
	for key, nested in padded_vowelcondprobdict.iteritems():
		print key.encode('utf8'), '\t\t'
		for nestedkey, value in nested.iteritems():
			print nestedkey.encode('utf8'), ':', round(value,3), '\t',
		print '\n'

	return padded_vowelcondprobdict


def load_vowel_set(vowelfile, mapfeatures):
	with open(vowelfile,'r') as f:
            vowels = [line.decode('utf8').strip().split(" ")[0] for line in f]
            if not mapfeatures:
		return set(vowels)
            else: #remove header row
                return set(vowels[1:])

def load_vowel_map(vowelfile):
	with open(vowelfile,'r') as f:
		vowelmap = {line.decode('utf8').strip().split(" ")[0]:line.decode('utf8').strip().split(" ")[-1] for line in f}
                hasdiffs = False
                for k,v in vowelmap.iteritems():
                    if k != v:
                        hasdiffs = True
                        break
                if hasdiffs:
                    print "\nVowel Mappings:"
                    for k,v in vowelmap.iteritems():
                        print "%s -> %s" % (k.encode("utf8"), v.encode("utf8"))
                    print ""
                return vowelmap

def physically_remove_outliervowels(outliervowelset, vowelset):
    print "Removing the following vowels:"
    for vowel in outliervowelset:
        print vowel.encode("utf8")
    print "\n"
    return vowelset.difference(outliervowelset)

def detect_outlier_vowels(vowelcondprobdict, vowelfreqdict):
        totalvowels = sum(vowelfreqdict.values())

        print "Vowel absolute and relative frequencies"
        for vowel, freq in vowelfreqdict.iteritems():
            print "vowel: %s\traw freq: %s\trelative freq: %s" % (vowel.encode("utf8"), freq, float(freq)/totalvowels)
        print "\n"

	nonprodvowelset = set()
	lowerfreqthreshold = 1 / (2*float(len(vowelfreqdict)))
        vowelrelfreqdict = {vowel:freq/float(totalvowels) for vowel, freq in vowelfreqdict.iteritems()}
        vowelsbyrelfreq = map(lambda (k,v) : k, sorted(vowelrelfreqdict.iteritems(), key=lambda (k,v) : (float(v),k)))
        killcandidatelist = set()
        killlist = set()
	for keyvowel, nestedvoweldict in vowelcondprobdict.iteritems():
            if nestedvoweldict[keyvowel] == max(nestedvoweldict.values()):
                if vowelrelfreqdict[keyvowel] < lowerfreqthreshold:
                    killcandidatelist.add(keyvowel)

        for vowel in vowelsbyrelfreq:
            if vowel in killcandidatelist:
                killlist.add(keyvowel)
            else:
                break

        return killlist

def detect_neutral_vowels(vowelcondprobdict):

	def renormalize_list(entry):
		total = sum(entry)
		entry = [elem/total for elem in entry]
		return entry


	vowelskewdict = {}
	n = len(vowelcondprobdict.keys())
	uniformdistro = [1.0/n]*n

	print "Vowel cooccurrence skew (KL from uniform):"
	indextovoweldict = {}
	for keyvowel, nestvoweldict in vowelcondprobdict.iteritems():
		indextovoweltuplelist = nestvoweldict.items()
		keyvowelcondproblist = [0]*n
		for i, pair in enumerate(indextovoweltuplelist):
                    indextovoweldict[i] = pair[0]
                    keyvowelcondproblist[i] = pair[1]
		for index,element in enumerate(keyvowelcondproblist):
			if element == 0.0:
				keyvowelcondproblist[index] = 0.0001
		keyvowelcondproblist = renormalize_list(keyvowelcondproblist)

		keyvowelskew = 0.0
		for i in range(len(uniformdistro)):
			currquot = uniformdistro[i] / keyvowelcondproblist[i]
			currquot = (fabs(log(currquot)) * uniformdistro[i])
			keyvowelskew += currquot

		vowelskewdict[keyvowel] = keyvowelskew

	sortedvowelskewdict = sorted(vowelskewdict.iteritems(), key=lambda (k,v) : (float(v),k) )
	for k, v in sortedvowelskewdict:
		print k.encode('utf8'), v

	print "\n"



	harmonizingvowelset = set()
#        threshhold = THRESHOLD(len(vowelcondprobdict))
        threshhold = 0.55 / float(len(vowelcondprobdict))
        print "Neutrality Threshold: %s\n(every entry in a vowel's co-occurrence vector must exceed this value)\n" % threshhold
	for keyvowel, nestvoweldict in vowelcondprobdict.iteritems():
		keyvowelcondproblist = nestvoweldict.values()
		for index,element in enumerate(keyvowelcondproblist):
			if element < threshhold:
				# We know that this is not a neutral vowel
				harmonizingvowelset.add(keyvowel)
				continue

	
	
	neutralvowelset = set(vowelcondprobdict.keys()).difference(harmonizingvowelset)
	print "neutralvowelset: "
	for vowel in neutralvowelset:
		print vowel.encode('utf8')
	return neutralvowelset

	#return vowelskewset


def classify_harmonizing_vowels(harmonizingvowelset, vowelcondprobdict):
	harmonizingvowellist = list(harmonizingvowelset)
	totalvowellist = list(vowelcondprobdict.keys())
	featureslist = []
	for harmonizingvowel in harmonizingvowellist:
		currvowelcondprobdict = vowelcondprobdict[harmonizingvowel]
                currvowelcondproblist = [currvowelcondprobdict[vowel] for vowel in totalvowellist]
		featureslist.append(currvowelcondproblist)
	featurearray = array(featureslist)

	featurearray = scipy.whiten(featurearray)

	#print "Debug featurearray: "
	#print featurearray

	centroids, distortion = scipy.kmeans(featurearray, 2)
	clusters, distortion = scipy.vq(featurearray, centroids)

	#print "Debug centroids: "
	#print centroids
	#print "Debug clusters: "
	#print clusters


	print 'harmonizingvowellist: ', len(harmonizingvowellist)
	#print 'clusters: ', len(clusters)
	harmonyclusterdict = defaultdict(int)
	for i, vowel in enumerate(harmonizingvowellist):
		print vowel.encode('utf8'), clusters[i]
		harmonyclusterdict[vowel] = clusters[i]

	return harmonyclusterdict


def detect_harmony(inputfile, outputfile, vowelset, vowelmap, vowelfeaturedict=None):
    windowsize = WINDOWSIZE

    worddict = load_wordlist(inputfile, vowelmap)

    mappedvowelset = vowelmap.values()
    if CONTEXTTYPE == ANTILOCAL:
        worddict = remove_windowsize_words(worddict, vowelset, windowsize)
    if ZIPF:
        worddict = remove_zipfian_tail(worddict)

    vowelcondprobdict = {}
    if CONTEXTTYPE == LOCAL:
        vowelfreqdict, vowelcondprobdict = calculate_vowel_conditional_probabilities_local(worddict, vowelset, windowsize)
    elif CONTEXTTYPE == GLOBAL:
        vowelfreqdict, vowelcondprobdict = calculate_vowel_conditional_probabilities_global(worddict, vowelset)
    else:
    	vowelfreqdict, vowelcondprobdict = calculate_vowel_conditional_probabilities_antilocal(worddict, vowelset, windowsize)

    outliervowelset = detect_outlier_vowels(vowelcondprobdict, vowelfreqdict)
    if outliervowelset:
        vowelset = physically_remove_outliervowels(outliervowelset, vowelset)
        if CONTEXTTYPE == LOCAL:
            vowelfreqdict, vowelcondprobdict = calculate_vowel_conditional_probabilities_local(worddict, vowelset, windowsize)
        elif CONTEXTTYPE == GLOBAL:
            vowelfreqdict, vowelcondprobdict = calculate_vowel_conditional_probabilities_global(worddict, vowelset)
        else:
            vowelfreqdict, vowelcondprobdict = calculate_vowel_conditional_probabilities_antilocal(worddict, vowelset, windowsize)

    neutralvowelset = detect_neutral_vowels(vowelcondprobdict)
    harmonizingvowelset = set(vowelcondprobdict.keys()).difference(neutralvowelset) # harmonizingvowelset = totalvowelset - neutralvowelset
  #  harmonizingvowelset = set(vowelcondprobdict.keys())


    harmonyclusterdict = {}
    if len(harmonizingvowelset) > 1:
        #if featurevwoeldict:
            #do use it with classify_harmonizing_vowels somehow
        harmonyclusterdict = classify_harmonizing_vowels(harmonizingvowelset, vowelcondprobdict)
    elif len(harmonizingvowelset) == 1:
        # else there is only one supposedly non-neutral vowel, we know that's a false positive (because what would it alternate with...)
        print list(harmonizingvowelset)[0] + " *"
        harmonyclusterdict = {}

#    for i in range(0,1):
#        print "Iteration: %s\t\tNum Words: %s" % (i, len(worddict.keys())) 
#        worddict = eval_proposed_harmony(worddict, vowelset, harmonyclusterdict)

    return harmonyclusterdict
    

def search_NER_candidates_for_harmony_violations(inputfile, harmonyclusterdict):
	possiblenercandidates = []
	with open(inputfile, 'r') as f:
		for line in f:
			currline = line.decode('utf8').strip()
			currline = currline.replace(',','\t').replace(' ', '\t')
			linecomponents = currline.split('\t')
			currword = linecomponents[-1]
			numviolations = evaluate_adherence_to_harmony(currword, harmonyclusterdict)
			wordvowellen = len(remove_nonvowels(currword, harmonyclusterdict.keys()))
			if (numviolations > 0):
				currwordinfo = (currword, numviolations, numviolations / float(wordvowellen))
				possiblenercandidates.append(currwordinfo)

	return possiblenercandidates

def evaluate_adherence_to_harmony(word, harmonyclusterdict):
	currwordvowelclusterlist = []
	for char in word:
		if char in harmonyclusterdict:
			currwordvowelclusterlist.append(harmonyclusterdict[char])

	numviolations =  0
	for index, value in enumerate(currwordvowelclusterlist):
		if index > 0:
			if currwordvowelclusterlist[index] != currwordvowelclusterlist[index-1]:
				numviolations += 1

	return numviolations

def read_featurefile(featurefile):
    vowelfeaturedict = {}
    with open(featurefile, "r") as f:
        header = f.readline().decode("utf8").strip()
        indexfeaturedict = {i:feat for i, feat in enumerate(header.split("\t")) if i > 0}
        for line in f:
            feats = line.decode("utf8").strip().split("\t")
            vowel = feats[0]
            featdict = {indexfeaturedict[i]:val for i, val in enumerate(feats) if i > 0}
            vowelfeaturedict[vowel] = featdict
    return vowelfeaturedict
                
def load_vowelfeatures_as_maps(vowelfile):
    vowelmaps = {}
    with open(vowelfile,'r') as f:
        #replace file 0s and 1s with -s and +s
        lines = [line.decode('utf8').strip().replace("0","-").replace("1","+").split("\t") for line in f]
        features = lines[0]
        numfeatures = len(features)-1

        featurelines = lines[1:]
        
        for i in range(1,numfeatures+1):
            vowelmap = {}
            for line in featurelines:
                vowel = line[0]
                featureval = line[i]
                vowelmap[vowel] = featureval
            vowelmaps[features[i]] = vowelmap
    return vowelmaps

def feature_postanalysis(harmonysetdict, featurefile):
    vowelfeaturedict = read_featurefile(featurefile)
    features = list(vowelfeaturedict.values())[0].keys()
    harmonyset0 = map(lambda kv : kv[0], filter(lambda kv : kv[1] == 0, harmonysetdict.items()))
    harmonyset1 = map(lambda kv : kv[0], filter(lambda kv : kv[1] == 1, harmonysetdict.items()))

    #first figure out which feature is harmonizing
    harmonizingfeat = None
    for feat in features:

        feat0 = map(lambda vowel : vowelfeaturedict[vowel][feat],harmonyset0)
        feat1 = map(lambda vowel : vowelfeaturedict[vowel][feat],harmonyset1)
        same0 = feat0.count(feat0[0]) == len(feat0)
        same1 = feat1.count(feat1[0]) == len(feat1)
        print feat, feat0, feat1, same0, same1
        if same0 and same1 and feat0[0] != feat1[0]:
            harmonizingfeat = feat

    if not harmonizingfeat:
        print "No obvious harmonizing feature\n"
        return

    print "Harmonizing feature: %s\n" %  harmonizingfeat

    #then get harmony pairs

    pairs = set()
    for v0 in harmonyset0:
        for v1 in harmonyset1:
            paired = True
            for feat,val in vowelfeaturedict[v0].iteritems():
                if vowelfeaturedict[v1][feat] != val and feat != harmonizingfeat:
                    paired = False
                    continue
            if paired:
                pairs.add((v0,v1))

    print "Harmonizing Pairs:"
    for pair in pairs:
        print "%s ~ %s" % pair

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Detect and analyze vowel harmony processes in a given language")
    parser.add_argument("inputfile", help="input wordlist to evaluate over. Format must either be one word per line, or count followed by word on each line (space, comma, or tab delimited)")
    parser.add_argument("outputfile", help="Output file with harmony process analysis")
    parser.add_argument("vowelfile", help="input file with list of vowels to consider. Assumed to be unicode, with each entry on its own line", nargs="?") # all other characters are assumed to be consonants (independent of the harmony process)
    parser.add_argument("-c","--contexttype", help="Indicates context to consider: Options: global [default], local, antilocal. Local directs the program to search for cooccurrences only strictly locally (adjeacent on the vowel tier) rather than over the whole word. Antilocal searches the complement of local.", type=str)
    parser.add_argument("--mapfeatures", help="Assume vowelfile is in the format of a vowel feature file. Map these in succession rather than learning on the orthographic vowels", action="store_true")
    parser.add_argument("--features", help="Vowel feature file. Tab delimited. One vowel per line followed by 1/0 values for features. Header row gives feature names. Don't use with --mapfeatures", type=str)
    parser.add_argument("--windowsize", help="Window size for local and antilocal contexts. Does nothing with global context. Default=1.", type=int)
    parser.add_argument("--nozipf", help="don't trim wordlist tail", action="store_true")
    args = parser.parse_args()

    if args.contexttype:
        if args.contexttype.lower().strip() == "global":
            CONTEXTTYPE = GLOBAL
        elif args.contexttype.lower().strip() == "local":
            CONTEXTTYPE = LOCAL
        elif args.contexttype.lower().strip() == "antilocal":
            CONTEXTTYPE = ANTILOCAL
        
    if args.windowsize:
        WINDOWSIZE = args.windowsize

    if args.nozipf:
        ZIPF = False

    vowelset = DEFAULT_VOWELS
    vowelmapdict = {}
    if args.vowelfile:
    	vowelset = load_vowel_set(args.vowelfile, args.mapfeatures)
        vowelmap = load_vowel_map(args.vowelfile)

    if args.features:
        print "got features"
        vowelfeaturedict = read_featurefile(args.features)
        harmonyclusterdict = detect_harmony(args.inputfile, args.outputfile, vowelset, vowelmap, vowelfeaturedict)
    else:
        print "no features"
        harmonyclusterdict = detect_harmony(args.inputfile, args.outputfile, vowelset, vowelmap)


    
#    nercandidatelist = search_NER_candidates_for_harmony_violations(args.inputfile, harmonyclusterdict)

#    if args.features:
#        feature_postanalysis(harmonyclusterdict, args.features)

#    with open(args.outputfile, 'w') as f:
#	    for value in nercandidatelist:
#	    	outputstring = value[0].encode('utf8') + "," + str(value[1]) + "," + str(value[2]) + "\n"
#	    	f.write(outputstring)
