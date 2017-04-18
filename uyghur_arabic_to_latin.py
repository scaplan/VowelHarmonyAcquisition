#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import argparse

ARATOLATINDICT = {u"ا":u"a",u"ە":u"e",u"ب":u"b",u"پ":u"p",u"ت":u"t",u"ج":u"j",u"چ":u"C",u"خ":u"x",u"د":u"d",u"ر":u"r",u"ز":u"z",u"ژ":u"Z",u"س":u"s",u"ش":u"S",u"غ":u"G",u"ف":u"f",u"ق":u"q",u"ك":u"k",u"گ":u"g",u"ڭ":u"N",u"ل":u"l",u"م":u"m",u"ن":u"n",u"ھ":u"h",u"و":u"o",u"ۇ":u"u",u"ۆ":u"ö",u"ۈ":u"ü",u"ۋ":u"w",u"ې":u"ë",u"ى":u"i",u"ي":u"y",u"ئ":u"@"}


def convert_text(intext):
    outtext = []
    for char in intext:
        try:
            outtext.append(ARATOLATINDICT[char])
        except KeyError:
            outtext.append(char)
    return "".join(outtext)

def read_file_as_lines(inputfile):
    with open(inputfile, "r") as f:
        for line in f:
            decodedline = line.decode('utf8').strip()
            print convert_text(decodedline).encode("utf8")
#           print decodedline, convert_text(decodedline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Convert Uyghur Perso-Arabic script to Latin Script")
    parser.add_argument("inputfile", help="input text file with Perso-Arabic text")
#    parser.add_argument("outputfile", help="Same file converted to Latin text")
    
    args = parser.parse_args()

    read_file_as_lines(args.inputfile)
    
