# obf

Converts plain text into other plain text, and back again.

It's not going to work out of the box.  It needs `dictionary.txt` (a simple
list of words like [this](http://www.mit.edu/~ecprice/wordlist.10000) or
[this](https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt),
though a list including contractions would be far more useful here) and
`corpus.txt` (any large volume of English text from which to gather frequency
data) to run.  But even then, check [status](#status) below.

## WTF?

The idea, here, is to create a bijective compression scheme aimed at producing
"plausible" text output for any input bitstream.

This makes it possible to compress, permute, and decompress text into other
text in a reversible way; without it being immediately obvious that the
resulting text represents a different (secret!) message.  If posted in a
suitable context (spam, drunk posts, drunk IRC, drunk SMS, comments sections,
drunk comments in comments sections, etc.) it may be less obvious still.

Moreover, this offers the chance to improve and perhaps even clarify such posts
by attempting to mechanically de-obfuscate them.  "Obfuscation" at this stage
is to simply exclusive-or the encoded bitstream with 0x55.

This project combines Markov chain models with range coding to define a
decompression scheme that implements the popular use of Markov chains to
synthesise English text.

The tricky part is keeping that bijective while accepting a broad range of
input (ideally any text you care to push into it, no matter how bad the
spelling) and at the same time striving to produce plausible output.  Random
synthesis does not face this challenge as it has no requirement to be able to
produce arbitrary output, which is an objective that works against producing
_plausible_ output.

Numerous hacks have accreted here to help try to promote coherent output.

For example, the basic tokens are dictionary words (plus a bit of punctuation),
but with an escape word that allows non-dictionary words to be spelled out
letter-by-letter.  This will not remain bijective on its own, as it becomes
possible to either spell or directly state dictionary words.  If there are two
ways to produce the same output text then the reversal process becomes
ambiguous, so the coding scheme for those spelled-out words must be made
incapable of duplicating any dictionary word.  To achieve this the spelled word
is finalised with an end token, and whenever the spelled-so-far word coincides
with a dictionary entry the end token is suppressed in the set of legal
codepoints so that the word must grow by at least another letter to avoid the
conflict.

Similar problems exist with trying to allow the encoding of numbers and
arbitrary punctuation without letting random output become _that_ random.  With
enough data this might be expected to resolve itself, but I don't have that
much data.  More hacking is required, there, and patterns like URLs should
probably get special treatment (but then we must be certain that a URL cannot
be expressed any other way; and on it goes...).

Also, it's a WIP.

## Status

It doesn't work very well right now.  I don't expect it to work at all out of
the box, because I have other priorities.

And also it uses way too much memory.  If you manage to get it running it may
well run out of memory before it does anything interesting.  Or you'll be
forced to kill it before your machine becomes unusable (how can this happen
with just one thread?).  There are a few things that can be tweaked but what's
really needed are some more sensible data structures.

Even when it does run to completion, results are likely to be disappointing.
It used to work better, but I've been hacking around with the way I do things
which de-tuned parameters somewhat (parameters I want to do away with
altogether), and that tuning was provisional and only ever worked well for
specific input in the first place.

I expect to oscillate between working the way I want and working well for at
least some input; hopefully one day converging on the former while broadening
the latter at the same time.
