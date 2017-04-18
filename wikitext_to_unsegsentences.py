#!/usr/bin/python -tt

import re
import argparse

def cleanline(line):
    remove = re.compile(r"\d|\s|[,\.\(\)\{\}\[\]-]")
    sentences = line.split(".")
    unsegs = [remove.sub("",sent.strip().lower()) for sent in sentences]
    return unsegs

def getlines(inputfile):
    sentences = []
    with open(inputfile,'r') as f:
        lines = [cleanline(line.decode('utf8')) for i, line in enumerate(f) if i < 100000]
        sentences = [sent for sents in lines for sent in sents]
    return sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Wikipeida text to unsegmented sentences")
    parser.add_argument("inputfile", help="cleaned wikipedia text")
    parser.add_argument("outputfile", help="output file")

    args = parser.parse_args()
    
    sentences = getlines(args.inputfile)
    with open(args.outputfile, 'w') as f:
        for sent in sentences:
            if sent:
                f.write(sent.encode("utf8") + "\n")
