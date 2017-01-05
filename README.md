# obf

Converts plain text into other plain text, and back again.

Needs `dictionary.txt` (a simple list of words like
[this](http://www.mit.edu/~ecprice/wordlist.10000) or
[this](https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt),
though a list including contractions would be far more useful here) and
`corpus.txt` (any large volume of English text from which to gather frequency
data) to run.

## WTF?

The idea, here, is to create a bijective compression scheme aimed at producing
"plausible" text output for any input bitstream.

This makes it possible to compress, permute, and decompress text into other
text in a reversible way; without it being immediately obvious that the
resulting text represents a different (secret!) message.  If posted in a
suitable context (spam, drunk posts, YouTube comments) it may less obvious
still.

Moreover, this offers the possibility of improving and perhaps even clarifying
such posts by attempting to mechanically de-obfuscate them.  "Obfuscation" at
this stage is to simply exclusive-or the bitstream with 0x55.

This project combines Markov chains with range coding to define a decompression
scheme that implements the popular use of Markov chains to synthesise English
text.

The tricky part is keeping that bijective while accepting a broad range of
input (ideally any text you care to push into it, no matter how bad the
spelling) and at the same time striving to produce plausible output.  Random
synthesis does not face this challenge as it has no requirement to be able to
produce arbitrary output, which is an objective that works against producing
_plausible_ output.

To help promote coherent output the basic tokens are dictionary words (plus a
bit of punctuation), but with an escape word that allows non-dictionary words
to be spelled out letter-by-letter.  To keep this bijective, those spelled-out
words must be incapable of duplicating any dictionary word (two ways to produce
the same output implies an ambiguous case in the reversal process).  The
spelled word is finalised with an end token, and whenever the spelled-so-far
word coincides with a dictionary entry the end token is suppressed in the set
of legal codepoints so that the word must grow by at least another letter to
avoid the conflict.

There's a bunch of stuff like that, and also a bunch of hacking around (really
not fit to deploy) trying to keep all the statistics in a huge array, and to
combine multiple-length-chains to add detail to sequences that appeared rarely
(or not at all) in the seed corpus.  And it's in the middle of being refactored
in a way that makes it easier to do more hacking.
