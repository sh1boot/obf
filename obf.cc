#include <cstdio>
#include <cassert>
#include <cctype>
#include <cstdarg>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <climits>
#include <fcntl.h>
#include <unistd.h>

#include <unordered_map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <functional>
#include <type_traits>


/* Some kind of enumeration of the fragments we deal with.
 */
class Token {
 protected:
  uint16_t value;
  static const uint16_t nothingValue = UINT16_MAX;
  bool inRange(uint16_t s, uint16_t n) const { return s <= value && value < (s + n); }
  explicit constexpr Token(uint16_t i = nothingValue) : value(i) {}

 public:
  static constexpr Token nothing() { return Token(nothingValue); }
  bool isNothing() const { return value == nothingValue; }

  uint16_t toInt() const { return value; }
  uint16_t hash() const { return value; }

  bool operator<(Token t) { return value < t.value; }
  bool operator<=(Token t) { return value <= t.value; }
  bool operator==(Token t) { return value == t.value; }
  bool operator!=(Token t) { return !(value == t.value); }
  bool operator>(Token t) { return !(value <= t.value); }
  bool operator>=(Token t) { return !(value < t.value); }
};

/* Slightly more specific enumerations of single-character (ASCII) fragments.
 */
template<char const* alphabet, bool endToken = true>
class LetterlikeToken : public Token {
  explicit LetterlikeToken(uint16_t i) : Token(i) { assert(isNothing() || value < range()); }

 public:
  explicit LetterlikeToken(char const* c) {
    if (c && *c && strchr(alphabet, *c) != NULL) {
      uint16_t i = strchr(alphabet, *c) - alphabet;
      value = i;
    } else {
      value = nothingValue;
    }
  }
  static constexpr LetterlikeToken fromChar(char c) {
    return LetterlikeToken(&c);
  }
  static constexpr LetterlikeToken fromInt(uint16_t i) {
    return LetterlikeToken(i);
  }
  static constexpr LetterlikeToken nothing() { return LetterlikeToken(nothingValue); }

  char toChar() const { if (isNothing()) return '\0'; return alphabet[value]; }

  static constexpr uint16_t range() { return strlen(alphabet) + (endToken ? 1 : 0); }
  static constexpr LetterlikeToken end() { return LetterlikeToken(range() - 1); }

  constexpr bool isEnd() const { return value == end().value; }
  constexpr bool isDigit() const { return isdigit(toChar()); }
  constexpr bool isXDigit() const { return isxdigit(toChar()); }
  constexpr bool isAlpha() const { return isalpha(toChar()); }
  constexpr bool isAlnum() const { return isalnum(toChar()); }
  constexpr bool isPunct() const { return ispunct(toChar()); }
};

/* These alphabets are ordered such that under specific conditions (defined by
 * classes within WordParser) the alphabet can be truncated and the
 * range-coding parameters can be easily adjusted to account for that.
 */
static char constexpr letterAlphabet[] = "abcdefghijklmnopqrstuvwxyz'";
static char constexpr separatorAlphabet[] = ".!?,;:()[]-/\"'*";
static char constexpr numberAlphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789$.,:";
typedef LetterlikeToken<letterAlphabet> LetterToken;
typedef LetterlikeToken<separatorAlphabet, false> SeparatorToken;
typedef LetterlikeToken<numberAlphabet> NumberToken;


/* Only some of the tokens represented here are words (the dictionary words).
 * Why are the others in here?  I'm not sure, anymore.  This is where life
 * began so there's a lot of history.
 *
 * Should I call this MacroToken, maybe?
 *
 * We have in here a list of dictionary words, separators (all the
 * SeparatorTokens, above), escapes to numbers and words we have to spell out
 * letter-by-letter, and suffixes.
 *
 * The suffixes were going to be words to be spelled out letter-by-letter but
 * with pre-defined endings.  By enumerating these separately, we get a
 * compromise token for many words we can't fit in the dictionary which might
 * (maybe) be interchangeable options in a markov chain, as in:
 *     "At school Albert studies (psych|astr|fren|thdrbroeptp)ology"
 * but that represents a lot of extra work, so I haven't done it yet.
 */
class WordToken : public Token {
  static std::unordered_map<std::string, uint16_t> wordMap;
  static std::unordered_map<std::string, uint16_t> suffixMap;
  static std::vector<std::string const*> wordList;
  static std::vector<std::string const*> suffixList;
  static std::vector<std::string> separatorStrings;
  static uint16_t firstWord,
                  firstSuffix,
                  firstSeparator,
                  firstNumber,
                  firstOther;
  static uint16_t periodValue;
  static const uint16_t otherMask = 0x3fff;

  /* hashValue should be the same as value except for tokens that are escapes
   * to unknown words and numbers, in which case we use a hash of the spelling.
   * This allows the context hash to respond to the specific (assuming no
   * collisions) word even if it's not in the dictionary.
   */
  uint16_t hashValue;

 protected:
  WordToken(uint16_t i, uint16_t h) : Token(i), hashValue(h) {}

 public:
  WordToken() : WordToken(nothingValue, nothingValue) {}
  operator std::string const& () { return toString(*this); }

  static WordToken nothing() { return WordToken(nothingValue, nothingValue); }

  static WordToken fromInt(uint16_t i) { return WordToken(i, i); }

  static size_t numWords() { return wordList.size(); }
  static size_t numSuffixes() { return suffixList.size(); }
  static size_t numOtherWords() { return otherMask + 1; }
  static size_t numNumbers() { return 1; }
  static size_t numSeparators() { return SeparatorToken::range(); }

  static WordToken dictionaryWord(unsigned i) { assert(i < numWords()); return fromInt(firstWord + i); }
  static WordToken suffixWord(unsigned i) { assert(i < numSuffixes()); return fromInt(firstSuffix + i); }
  static WordToken otherWord(uint32_t h) { return WordToken(firstOther, firstOther + (h & otherMask)); }
  static WordToken separatorWord(unsigned i) { assert(i < numSeparators()); return fromInt(firstSeparator + i); }
  static WordToken numberWord(unsigned i) { assert(i < numNumbers()); return fromInt(firstNumber + i); }
  static WordToken period() { return WordToken(periodValue, periodValue); }

  uint16_t hash() const { return hashValue; }

  bool isDictionaryWord() const { return inRange(firstWord, numWords()); }
  bool isSuffix() const { return inRange(firstSuffix, numSuffixes()); }
  bool isSeparator() const { return inRange(firstSeparator, numSeparators()); }
  bool isNumber() const { return inRange(firstNumber, numNumbers()); }
  bool isOtherWord() const { return inRange(firstOther, numOtherWords()); }
  bool needsLetters() const { return isSuffix() || isOtherWord(); }
  bool needsDigits() const { return isNumber(); }

  std::string const& toString(WordToken t) const {
    if (isDictionaryWord()) return *wordList[value - firstWord];
    if (isSuffix()) return *suffixList[value - firstSuffix];
    if (isSeparator()) return separatorStrings[value - firstSeparator];
    static const std::string number("<999>");
    if (isNumber()) return number;
    static const std::string nothing("<nothing>");
    if (isNothing()) return nothing;
    static const std::string other("<smurf>");
    return other;
  }

  std::string const& toString() const {
    return toString(*this);
  }

  char const* c_str() const {
    return toString(*this).c_str();
  }

  static uint16_t range() { return firstOther + 1; }

  static ssize_t init(char const* dict = "dictionary.txt",
                      bool verbose = false);

 public:
  /* TODO: eliminate this... */
  static ssize_t lookupWord(std::string const& s) {
    auto it = wordMap.find(s);
    if (it != wordMap.end()) {
      return it->second;
    }
    return -1;
  }
};

/* A utility class to incrementally classify strings (words, if you will) and
 * report what characters would be legal extensions on what's been read so far.
 * This manages things like prohibiting the end-of-word token in an "other"
 * word when the word parsed so far is already a dictionary word (because we
 * can never allow two encodings for the same string).
 *
 * This is all very ad-hoc and needs to be replaced by something less so.  Also
 * I started messing around with C++ more than I really needed to, here, and it
 * probably shows.
 *
 * TODO: Start again.
 */
class WordParser {
  template<typename T>
  class ParserBase {
   public:
    using TokenType = T;
    ParserBase() : illegalTokens(T::nothing()) {}
    bool needMore() const { return !illegalTokens.isNothing(); }
    bool isFailed() { return fail; }
    TokenType getIllegalTokens() const { return illegalTokens; }

   protected:
    TokenType illegalTokens = TokenType::nothing();
    bool fail = true;
  };
  template<typename T>
  struct ParserTop : public T {
    using typename T::TokenType;
    using T::fail;
    using T::illegalTokens;
    using T::needMore;
    using T::reset;
    using T::update;
    using T::get;
    //std::static_assert(std::is_base_of<ParserBase<T::TokenType>, T>::value, "bad T");

    ParserTop() : T() { start(); }
    ParserTop(bool b) : T(b) { start(); }
    void start() {
      illegalTokens = TokenType::end();
      fail = false;
      reset();
    }
    WordToken append(TokenType t, uint32_t hash, bool reserved) {
      if (fail) return WordToken::nothing();
      if (t.isNothing() || t >= illegalTokens) {
        fail = true;
        return WordToken::nothing();
      }
      illegalTokens = reserved ? TokenType::end() : TokenType::nothing();
      update(t);
      if (fail || needMore()) return WordToken::nothing();
      return get(hash);
    }
    WordToken append(char c, uint32_t hash, bool reserved) {
      TokenType t = TokenType::fromChar(c);
      return append(t, hash, reserved);
    }
    TokenType getIllegalTokens() const { return illegalTokens; }
  };

  class OtherWordParser_Middle : public ParserBase<LetterToken> {
   protected:
    void reset() {
      illegalTokens = LetterToken::fromChar('\'');
    }
    void update(TokenType t) {
      if (t == TokenType::fromChar('\'')) {
        illegalTokens = TokenType::fromChar('\'');
      }
    }
    WordToken get(uint32_t hash) const {
      return WordToken::otherWord(hash);
    }
  };

  class DictWordParser_Middle : public OtherWordParser_Middle {
    std::string word;  // TODO: eliminate
    const bool enabled;

   protected:
    void reset() {
      if (enabled) OtherWordParser_Middle::reset();
      else fail = true;
    }

    void update(TokenType t) {
      word += t.toChar();
      OtherWordParser_Middle::update(t);
    }

    WordToken get(uint32_t hash) const {
      // TODO: this, incrementally, with a triedawg/whatever in update()
      int wordIndex = WordToken::lookupWord(word);
      return (wordIndex >= 0) ? WordToken::dictionaryWord(wordIndex)
                              : WordToken::nothing();
    }

   public:
    DictWordParser_Middle(bool enable = true) : enabled(enable) {}
  };

  class NumberParser_Middle : public ParserBase<NumberToken> {
    bool seenNonAlpha;

   protected:
    void reset() {
      seenNonAlpha = false;
      /* can't have a . or , before we see a digit. */
      illegalTokens = NumberToken::fromChar('.');
    }
    bool isNotLetter(NumberToken t) {
      bool r = !t.isAlpha();
      assert( r == LetterToken::fromChar(t.toChar()).isNothing());
      return r;
    }
    void update(NumberToken t) {
      seenNonAlpha = seenNonAlpha || isNotLetter(t);
      if (!seenNonAlpha) {
        illegalTokens = NumberToken::fromChar('.');
      } else if (!t.isAlnum()) {
        illegalTokens = NumberToken::end();
      }
    }
    WordToken get(uint32_t hash) const {
      if (seenNonAlpha) return WordToken::numberWord(0);
      else return WordToken::nothing();
    }
  };

  class SeparatorParser_Middle : public ParserBase<SeparatorToken> {
    TokenType value = TokenType::nothing();

   protected:
    void reset() { value = TokenType::nothing(); }
    void update(SeparatorToken t) {
      if (value.isNothing()) value = t;
      else fail = true;
    }
    WordToken get(uint32_t) const {
      return WordToken::separatorWord(value.toInt());
    }
  };

 public:
  using OtherWordParser = ParserTop<OtherWordParser_Middle>;
  using DictWordParser = ParserTop<DictWordParser_Middle>;
  using NumberParser = ParserTop<NumberParser_Middle>;
  using SeparatorParser = ParserTop<SeparatorParser_Middle>;

  DictWordParser dict;
  OtherWordParser other;
  NumberParser number;
  SeparatorParser separator;
  WordToken dictToken;
  WordToken otherToken;
  WordToken numberToken;
  WordToken separatorToken;
  uint32_t hash;
  size_t position;
  WordToken goodToken;
  size_t goodPosition;

 public:
  WordParser(bool useDict = true) : dict(useDict) {}

  void start() {
    hash = 0xdeadbeef;
    position = 0;
    goodToken = WordToken::nothing();
    goodPosition = 0;
    dict.start();
    other.start();
    separator.start();
    number.start();
  }

  bool nextChar(char c) {
    struct WordTokenAcc {
      WordToken v;
      WordTokenAcc() : v(WordToken::nothing()) {}
      WordTokenAcc(WordToken t) : v(t) {}
      WordTokenAcc operator |=(WordToken t) {
        if (isNothing()) v = t;
        return *this;
      }
      operator WordToken() { return v; }
      bool isNothing() { return v.isNothing(); }
      bool toInt() { return v.toInt(); }
      std::string const& toString() { return v.toString(); }
    } r;
    hash = ((hash ^ (hash >> 15)) + c * 0x12345) * 31;
    position++;
    uint32_t h = (hash ^ (hash >> 17));
    r |= (dictToken = dict.append(c, h, !r.isNothing()));
    r |= (otherToken = other.append(c, h, !r.isNothing()));
    r |= (numberToken = number.append(c, h, !r.isNothing()));
    r |= (separatorToken = separator.append(c, h, !r.isNothing()));

    if (!r.isNothing()) {
      goodToken = r;
      goodPosition = position;
    }
    bool done = dict.isFailed()
              && other.isFailed()
              && number.isFailed()
              && separator.isFailed();
    return !done;
  }
  WordToken getResult(size_t& length) {
    length = goodPosition;
    return goodToken;
  }

  template<typename T> inline T getIllegalTokens() const;
  template<typename T> inline WordToken getToken() const;
  template<typename T>
  bool isOk() const {
    return !getToken<T>().isNothing();
  }
};

template<> inline
LetterToken WordParser::getIllegalTokens<LetterToken>() const { return other.getIllegalTokens(); }
template<> inline
NumberToken WordParser::getIllegalTokens<NumberToken>() const { return number.getIllegalTokens(); }
template<> inline
SeparatorToken WordParser::getIllegalTokens() const { return separator.getIllegalTokens(); }
template<> inline
WordToken WordParser::getToken<LetterToken>() const { return otherToken; }
template<> inline
WordToken WordParser::getToken<NumberToken>() const { return numberToken; }
template<> inline
WordToken WordParser::getToken<SeparatorToken>() const { return separatorToken; }


/* Converts a std::istream to a series of WordTokens through calls to next().
 */
class Tokenizer {
  std::istream& in;
  std::string buffer;
  std::string ungets;
  bool lineNums = true;
  int line = 1;
  bool useDictionary;

  int get() {
    if (ungets.size() > 0) {
      int r = ungets.back();
      ungets.pop_back();
      return r;
    }
    return in.get();
  }

  void putback(int c) {
    assert(c != EOF);
    ungets.push_back(c);
  }

 public:
  Tokenizer(std::istream& stream, bool useDict = true, bool verbose = false)
    : in(stream), lineNums(verbose), useDictionary(useDict) {
    buffer.reserve(1024);
    ungets.reserve(1024);
  }

  WordToken next() {
    buffer.clear();
    WordParser wp(useDictionary);
    int c;
    for (;;) {
      while ((c = get()) != EOF && (isspace(c) || iscntrl(c))) {
        if (c == '\n' && lineNums && ++line % 10000 == 0) {
          printf("line %d\r", line);
          fflush(stdout);
        }
      }
      if (c == EOF) {
        if (lineNums) printf("%d lines.\n", line);
        return WordToken::nothing();
      }

      buffer.clear();
      wp.start();
      while (c != EOF && wp.nextChar(tolower(c))) {
        buffer += tolower(c);
        c = get();
      }
      size_t len;
      auto t = wp.getResult(len);
      if (!t.isNothing()) {
        assert(len > 0);
        if (c != EOF) putback(c);
        while (buffer.length() > len) {
          putback(buffer.back());
          buffer.pop_back();
        }
        assert(!t.isNothing());
        return t;
      } else {
        assert(len == 0);
      }
    }
  }
  std::string const& spellThat() {
    return buffer;
  }
};

std::vector<std::string const*> WordToken::wordList;
std::vector<std::string const*> WordToken::suffixList;
std::unordered_map<std::string, uint16_t> WordToken::wordMap;
std::unordered_map<std::string, uint16_t> WordToken::suffixMap;
std::vector<std::string> WordToken::separatorStrings;
uint16_t  WordToken::firstWord,
          WordToken::firstSuffix,
          WordToken::firstSeparator,
          WordToken::firstNumber,
          WordToken::firstOther;
uint16_t  WordToken::periodValue = Token::nothingValue;

ssize_t WordToken::init(char const* dict, bool verbose) {
  std::filebuf fb;
  if (!fb.open(dict, std::ios::in)) return -1;

  if (verbose) {
    printf("reading %s...\n", dict);
    fflush(stdout);
  }

  std::istream is(&fb);
  Tokenizer tok(is, false, verbose);
  int count = 0;

  while (!tok.next().isNothing()) {
    auto& s = tok.spellThat();
    if (isalpha(s[0]) && wordMap.emplace(s, count).second) count++;
  }
  fb.close();

  if (verbose) printf("dictionary is %zu words.\n", wordMap.size());
  /* TODO: suffixMap */

  for (auto c : separatorAlphabet) {
    if (c != '\0') {
      separatorStrings.emplace_back(std::string(1, c));
    }
  }

  wordList.resize(wordMap.size());
  for (auto& it : wordMap) wordList[it.second] = &it.first;
  suffixList.resize(suffixMap.size());
  for (auto& it : suffixMap) suffixList[it.second] = &it.first;

  firstWord = 0;
  firstSuffix = firstWord + numWords();
  firstSeparator = firstSuffix + numSuffixes();
  firstNumber = firstSeparator + numSeparators();
  firstOther = firstNumber + numNumbers();
  periodValue = firstSeparator;

  if (verbose) printf("dictWords: %zu, suffWords: %zu, sepWords: %zu, numWords: %zu\n",
      numWords(), numSuffixes(), numSeparators(), numNumbers());
  return 0;
}


/* Class needed by RangeCoder to get ranges to code. */
struct RangeCoderProb {
  uint64_t getTotal() const { return total; }
  virtual uint64_t getSpan(int symbol, uint64_t& base, uint64_t& size) const = 0;
  virtual int getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const = 0;

 protected:
  uint64_t total;
};

/* As advertised.
 */
class RangeCoder {
  static const bool debug = true;
  uint64_t low = 0;
  uint64_t range = 0x8000000000000000;
  __extension__ typedef unsigned __int128 uint128_t;
  /* TODO: keep a running EOF stream value to detect/emit at EOF. */

  void encode(uint64_t base, uint64_t size, uint64_t total) {
    assert(size > 0);
    assert(base + size <= total);

    /* TODO: handle small ranges with deferred carry or suchlike. */
    assert(total <= range);

    low += base * (uint128_t)range / total;
    range = size * (uint128_t)range / total;

    dbg("(%08x-%08x ", (unsigned)(low >> 32), (unsigned)((low + range) >> 32));

    if (bitsAvailable() > 0) {
      if (debug) for (int i = 0; i < bitsAvailable(); i++)
        dbgchar(((low << i) >> 63) ? '1' : '0');
      advance(bitsAvailable());
    }
    if (debug) {
      dbgchar(' ');
      for (int i = 0; i < 12; i++) dbgchar(((low << i) >> 63) ? '1' : '0');
      dbgchar('-');
      for (int i = 0; i < 12; i++) dbgchar((((low + range - 1) << i) >> 63) ? '1' : '0');
      dbg(" %08x-%08x)", (unsigned)(low >> 32), (unsigned)((low + range) >> 32));
    }
  }

 protected:
  uint64_t bs = 0;

 public:
  RangeCoder() { consume(63); }

  int decode(RangeCoderProb const& p) {
    uint64_t total = p.getTotal();
    assert(total <= range);
    //TODO: should be: `uint64_t off = ((bs - low) * (uint128_t)total + range - 1) / range;`, right?
    uint64_t off = (bs - low) * (uint128_t)total / range;
    uint64_t base, size;
    int r = p.getSymbol(off, base, size);
    dbg("dec: s:%08x-%08x ", (unsigned)(base >> 0), (unsigned)((base + size) >> 0));
    encode(base, size, total);
    assert(low <= bs && bs < (low + range));
    return r;
  }

  void encode(RangeCoderProb const& p, int symbol) {
    uint64_t base, size, total;
    total = p.getSpan(symbol, base, size);
    dbg("enc: s:%08x-%08x ", (unsigned)(base >> 0), (unsigned)((base + size) >> 0));
    encode(base, size, total);
  }

 protected:
  int bitsAvailable() const {
    uint64_t test = low ^ (low + range);
    return __builtin_clzll(test);
  }

  virtual void advance(int bits) {
    emit(bits);
    consume(bits);
    low <<= bits;
    range <<= bits;
  }

  virtual void emit(int bits) {
    uint64_t bs = low;
    for (int i = 0; i < bits; i++) {
      putbit(bs >> 63);
      bs <<= 1;
    }
  }
  virtual void consume(int bits) {
    for (int i = 0; i < bits; i++) {
      bs = (bs << 1) | getbit();
    }
  }
  virtual void putbit(int b) {
    putchar('0' | b);
  }
  virtual int getbit() {
    return random() & 1;
  }

 public:
  void setLog(std::string* s) {
    debuglog = s;
  }

  std::string log() {
    if (debuglog) {
      std::string r = *debuglog;
      debuglog->clear();
      return r;
    }
    return "";
  }

 private:
  std::string* debuglog = NULL;

  void dbgchar(char c) const {
    if (debug && debuglog) *debuglog += c;
  }

  void dbg(char const* fmt, ...) const {
    if (debug && debuglog) {
      char buf[4096];
      va_list ap;
      va_start(ap, fmt);
      vsnprintf(buf, sizeof(buf), fmt, ap);
      va_end(ap);
      *debuglog += buf;
    }
  }
};

/* Keep a hash of the last n tokens seen.  Also keep 'alpha' and 'beta', hashes
 * of the last 1 and 2 tokens seen.
 *
 * TODO(think): Is a distance of 1 pointless?  The dictionary words are
 * probably the commonest words and so that should be pretty flat -- there are
 * only a couple of tokens where we should expect meaningfully un-flat values.
 * Maybe distances of 2 and 3 make more sense.
 *
 * TODO: Make it not a template, with static history size (power of two) and
 * programmable history distance.
 */
template <unsigned D, const uint64_t m = 0x98b5892bb6fb97d9>
class HistoryHash {
  uint16_t history[D];
  unsigned histpos = 0;
  uint64_t hash = 0, alpha = 0, beta = 0;

  uint64_t FullHash(void) {
    uint64_t h = 0;
    for (unsigned i = 0; i < D; i++) {
      h = (h + history[(histpos + i) % D]) * m;
    }
    return h;
  }

  static uint64_t lag(unsigned d = D) {
    uint64_t e = 1;
    for (unsigned i = 0; i < d; i++) e *= m;
    return e;
  }

 public:
  HistoryHash(void) { reset(); }
  void reset(void) {
    memset(history, 0, sizeof(history));
    histpos = 0;
    hash = 0;
  }
  void integrate(uint32_t t) {
    hash = hash - history[histpos] * lag();
    t *= 0xb16d5a03;
    t ^= t >> 15;
    history[histpos] = t;
    hash = (hash + history[histpos]) * m;
    beta = (alpha + t) * m;
    alpha = t * m;
    histpos = (histpos + 1) % D;

    assert(hash == FullHash());
  }
  void reset(uint32_t s) {
    reset();
    if (s) integrate(s);
  }
  uint32_t get(unsigned bits) const {
    return (uint32_t)(hash ^ (hash >> 32)) >> (32 - bits);
  }
  uint32_t getAlpha(unsigned bits) const {
    return (uint32_t)(alpha ^ (alpha >> 32)) >> (32 - bits);
  }
  uint32_t getBeta(unsigned bits) const {
    return (uint32_t)(beta ^ (beta >> 32)) >> (32 - bits);
  }
};

/* Manage accumulation of statistics and return probabilities for given context.
 */
template<typename T, int hashDist = 4,
        uint64_t fscale = 1024, uint64_t fscale_a = 128, uint64_t fscale_b = 32>
class Mumble {
 public:
  using TokenType = T;  // TODO: Pretty sure this isn't necessary.
  using ContextHash = HistoryHash<hashDist>;  // TODO: Can probably squash this, too.

  class RCProb : public RangeCoderProb {
    Mumble const& source;
    ContextHash const& hashes;


    uint64_t getSpan_(int symbol, uint64_t& base, uint64_t& size) const {
      base = 0;
      for (int i = 0; i < symbol; i++) base += source.getScore(hashes, i);
      size = source.getScore(hashes, symbol);
      return total;
    }

    int getSymbol_(uint64_t off, uint64_t& base, uint64_t& size) const {
      assert(off < total);
      base = 0;
      int symbol = 0;
      for (base = 0; base < total; base += size, symbol++) {
        size = source.getScore(hashes, symbol);
        if (off < base + size) {
          return symbol;
        }
      }
      assert(!"findToken() failed");
      return -1;
    }

   public:
    RCProb(Mumble const& src, ContextHash const& h, TokenType illegal)
      : source(src), hashes(h) {
      total = source.getTotal(hashes, illegal);
    }

    virtual uint64_t getSpan(int symbol, uint64_t& base, uint64_t& size) const override;
    virtual int getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const override;
  };
  RCProb getStats(ContextHash const& history, TokenType illegal) const {
    return RCProb(*this, history, illegal);
  }

 private:
  uint16_t tableSize;

  template<uint32_t (*getHash)(ContextHash const& ctx)>
  class FreqData {
#if 0
    /* good for sparse tables,
     * high memory overhead in dense tables (but overall win),
     * terrible CPU overhead
     */
    struct FreqTab : public std::multiset<uint16_t> {
      FreqTab(uint32_t, uint16_t) {}
    };
#else
    /* bad for sparse tables,
     * low memory overhead for dense tables (overall lose),
     * minimal CPU overhead
     */
    class FreqTab {
      uint32_t total = 0;
      std::vector<uint32_t> counts;

     public:
      FreqTab(uint32_t h, uint16_t sz) : total(0), counts(sz, 0) { }

      void insert(uint16_t i) {
        counts[i]++;
        total++;
      }
      size_t count(uint16_t i) const { return counts[i]; }
      size_t size(void) const { return total; }
      using iterator = std::vector<uint32_t>::iterator;
      using const_iterator = std::vector<uint32_t>::const_iterator;
    };
#endif
    std::unordered_map<uint32_t, FreqTab> freqMap;
    uint64_t total = 0;
    uint16_t tableSize;

   public:
    FreqData(uint16_t sz) : tableSize(sz) {}

    void insert(ContextHash const& ctx, uint16_t i) {
      total++;
      auto set = freqMap.emplace(getHash(ctx), FreqTab(getHash(ctx), tableSize)).first;
      set->second.insert(i);
    }
    size_t count(ContextHash const& ctx, uint16_t i) const {
      auto set = freqMap.find(getHash(ctx));
      if (set == freqMap.end()) return 0;
      return set->second.count(i);
    }
    size_t size(ContextHash const& ctx) const {
      auto set = freqMap.find(getHash(ctx));
      if (set == freqMap.end()) return 0;
      return set->second.size();
    }
    size_t size() const {
      return total;
    }
  };

  /* TODO: hashBits should be 32 (assumed to be sufficient for no collisions),
   * but we're just going to cut it short and endure the collisions as "part of
   * the fun", because we don't have a great underlying data structure right
   * now (something that scales between sparse and dense while keeping the
   * memory footprint minimal).
   */
  static const int hashBits = 18;
  static uint32_t getHash(ContextHash const& ctx) { return ctx.get(hashBits); }
  static uint32_t getHash_a(ContextHash const& ctx) { return ctx.getAlpha(hashBits); }
  static uint32_t getHash_b(ContextHash const& ctx) { return ctx.getBeta(hashBits); }
  FreqData<getHash> freq;
  FreqData<getHash_a> freq_a;
  FreqData<getHash_b> freq_b;

  void inc(ContextHash const& ctx, int i) {
    freq.insert(ctx, i);
    freq_a.insert(ctx, i);
    freq_b.insert(ctx, i);
  }

  /* Currently histogram data is synthesised from three frequency tables.  The
   * stats for the full hashDist-length hash (very sparse), and the alpha and
   * beta hashes (much denser).  These are weighted by fscale* to give priority
   * to the full-length hash when it's available, and to use the shorter hashes
   * as low-weight tiebreakers.  All that plus one for each bucket to ensure
   * that every outcome is attainable.
   *
   * This method is really awful, and there's quite a lot obviously wrong with
   * it, but it kind of worked a little bit at some point while I was hacking.
   *
   * Really the tables represent only weightings, and the absolute magnitudes
   * reflect only the coverage of the sampling.  These should be mixed in a
   * more insightful way and the combined result distilled into something
   * compact and easy to parse.
   *
   * TODO(quick-and-dirty): Throw away tables with low populations, and retain
   * only the longest-match table that has adequate coverage.  Refer to that
   * alone.
   *
   * What I think we really want is a selection of typical sorted distributions
   * (maybe just the one curve, in fact) and for each hash a pointer to a list
   * of tokens in the order they appear on that curve under that hash.  That's
   * a fuzzy match already, and the order can probably afford to be fairly
   * fuzzy too (especially for small values), so many hashes should share the
   * same order and/or the same curve.
   *
   * TODO: Make that happen.
   *
   * TODO(interim): Scan the tables more sensibly (particularly when using
   * multisets or similar), retaining results of past look-ups, iterators,
   * etc..
   */

  /* Answer queries from RCProb, for the range coder to use.
   */
  uint64_t getScore(ContextHash const& ctx, int i) const {
    return 1 + fscale   * freq.count(ctx, i)
             + fscale_a * freq_a.count(ctx, i)
             + fscale_b * freq_b.count(ctx, i);
  }

  uint64_t getTotal(ContextHash const& ctx, TokenType illegal = TokenType::nothing()) const {
    uint64_t total = tableSize + fscale   * freq.size(ctx)
                               + fscale_a * freq_a.size(ctx)
                               + fscale_b * freq_b.size(ctx);
    if (!illegal.isNothing()) {
      for (int i = illegal.toInt(); i < tableSize; i++)
        total -= getScore(ctx, i);
    }
    return total;
  }

  /* TODO: Could probably accept fscale* values here, and hash distance too.
   * Not sure there's any need to make this a template at all... have to check
   * how StringContext turns out to be sure.
   */
 public:
  Mumble(int tmax) : tableSize(tmax), freq(tmax), freq_a(tmax), freq_b(tmax) {
  }

  void integrate(ContextHash const& ctx, Token t) {
    if (t.isNothing()) return;
    uint16_t i = t.toInt();
    assert(i < tableSize);
    inc(ctx, i);
  }
};

typedef Mumble<WordToken> WordMumble;
typedef Mumble<LetterToken, 6, 16, 1, 1> LetterMumble;
typedef Mumble<NumberToken, 6, 16, 1, 1> NumberMumble;


/* TODO: Make a better effort to expose what I really want in WordParser and
 * then making a WordParser out of that, rather than throwing it all inside
 * WordParser and ignoring the redundancy.
 */
template<typename T>
struct TokenParser : public WordParser {
  void reset() { WordParser::start(); }
  bool nextChar(T t) { return WordParser::nextChar(t.toChar()); }
  T getIllegalTokens() const { return WordParser::getIllegalTokens<T>(); }
  T end() const { assert(WordParser::isOk<T>()); return T::end(); }
  T end(bool dictWord) const { assert(dictWord != WordParser::isOk<T>()); return T::end(); }
  WordToken getToken() const { return WordParser::getToken<T>(); }
};

template<>
struct TokenParser<WordToken> {
  WordToken v;
  static void reset() {}
  void nextChar(WordToken t) { v = t; }
  static WordToken getIllegalTokens() { return WordToken::nothing(); }
  WordToken end() const { return WordToken::nothing(); }
  WordToken getToken() const { return v; }
};

/* Maintain rolling hashes and legal states and stuff...
 */
template<typename M>
class StringContext : public TokenParser<typename M::TokenType> {
 protected:
  using super = TokenParser<typename M::TokenType>;
  M const& mumble;
  typename M::ContextHash history;

 public:
  using TokenType = typename M::TokenType;
  using ContextHash = typename M::ContextHash;
  using RCProb = typename M::RCProb;

  void reset() {
    super::reset();
    history.reset();
  }

  void reset(uint32_t h) {
    super::reset();
    history.reset(h);
  }

  StringContext(M const& m) : super(), mumble(m) { reset(); }
  StringContext(M const& m, uint32_t h) : super(), mumble(m) { reset(h); }

  RCProb getStats() const {
    return mumble.getStats(history, super::getIllegalTokens());
  }

  void updateState(TokenType t) {
    assert(!t.isNothing());
    super::nextChar(t);
    history.integrate(t.hash());
  }
};

/* As above, but modifying M as we go.  Used during the analysis phase.
 */
template<typename M>
class StringContextRW : public StringContext<M> {
  M& mumble;
  using StringContext<M>::history;

 public:
  StringContextRW(M& m) : StringContext<M>(m), mumble(m) {}
  StringContextRW(M& m, WordToken t) : StringContext<M>(m, t), mumble(m) {}

  /* Call this _before_ updateState(), because the state tells us what stats
   * will be affected by the transition to the new state.
   */
  void updateStats(Token t) {
    mumble.integrate(history, t);
  }
};

using  WordContext = StringContext<WordMumble>;
using  LetterContext = StringContext<LetterMumble>;
using  NumberContext = StringContext<NumberMumble>;

using  WordContextRW = StringContextRW<WordMumble>;
using  LetterContextRW = StringContextRW<LetterMumble>;
using  NumberContextRW = StringContextRW<NumberMumble>;

template<>
uint64_t WordContext::RCProb::getSpan(int symbol, uint64_t& base, uint64_t& size) const {
  return getSpan_(symbol, base, size);
}
template<>
uint64_t LetterContext::RCProb::getSpan(int symbol, uint64_t& base, uint64_t& size) const {
  return getSpan_(symbol, base, size);
}
template<>
uint64_t NumberContext::RCProb::getSpan(int symbol, uint64_t& base, uint64_t& size) const {
  return getSpan_(symbol, base, size);
}
template<>
int WordContext::RCProb::getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const {
  return getSymbol_(off, base, size);
}
template<>
int LetterContext::RCProb::getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const {
  return getSymbol_(off, base, size);
}
template<>
int NumberContext::RCProb::getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const {
  return getSymbol_(off, base, size);
}


/* Somewhere to stick my printf debugging so it can be deferred until something
 * bad happens.
 */
class DebugLog {
  std::vector<std::string>* log;
  bool verbose = false;

 public:
  DebugLog(std::vector<std::string>* l) : log(l) {}
  void setVerbose(bool v) { verbose = v; }
  void push_back(std::string const&& s) {
    if (log) log->emplace_back(s);
    if (verbose) {
      printf("%s\n", s.c_str());
      fflush(stdout);
    }
  }
};


template<class Context>
static WordToken decodeWord(std::string& s, RangeCoder& dec,
    Context ctx, DebugLog& dbg) {
  s.clear();
  for (;;) {
    int i = dec.decode(ctx.getStats());
    auto t = Context::TokenType::fromInt(i);
    if (t.isEnd()) {
      dbg.push_back(dec.log() + " \"" + s + "\" " + std::to_string(ctx.getToken().hash()));
      break;
    }
    s += t.toChar();
    ctx.updateState(t);
  }
  assert(s.length() > 0);

  /* For hashing we integrate the hash of the unknown word, rather than
   * just the constant 'other' token.  We can do this only after we've
   * decoded the word, so we need to update the caller now.
   */
  return ctx.getToken();
}

template <class Context>
static WordToken encodeWord(RangeCoder& enc, std::string const& s,
      Context ctx, DebugLog& dbg) {
  for (auto c : s) {
    auto t = Context::TokenType::fromChar(c);
    assert(!t.isNothing() && t < ctx.getIllegalTokens());
    enc.encode(ctx.getStats(), t.toInt());
    ctx.updateState(t);
  }
  assert(!ctx.getToken().isNothing());
  enc.encode(ctx.getStats(), ctx.end().toInt());
  dbg.push_back(enc.log() + " \"" + s + "\" " + std::to_string(ctx.getToken().hash()));
  return ctx.getToken();
}

/* Not really used the way it was written, anymore.  Was meant to do the
 * synthesis of text from random input (and should really be parsing bitstreams
 * to do the regular decode of encoded input), but that got left behind and now
 * it's just filling in for the pretty-printer I really need to write.
 *
 * TODO: Throw this away.
 */
class MumbleStream {
  WordMumble const& words;
  LetterMumble const& letters;
  NumberMumble const& numbers;

  bool startSentence = true;
  bool flipCase = false;
  bool inSQuotes = false;
  bool inDQuotes = false;
  std::string ifWord = "";
  bool wasWord = false;
  int inParentheses = 0;
  bool endParagraph = false;
  int lineWidth = 0;

 public:
  MumbleStream(WordMumble const& w, LetterMumble const& l, NumberMumble const& n)
    : words(w), letters(l), numbers(n) {}

  void resetDecor() {
    startSentence = true;
    flipCase = false;
    inSQuotes = false;
    inDQuotes = false;
    ifWord = "";
    wasWord = false;
    inParentheses = 0;
    endParagraph = false;
    lineWidth = 0;
  }

  std::string decorate(std::string word, bool isParagraph) {
    std::string string;
    /* TODO: use the original token or something, because this will eventually
     * break proper spacing between tokens with similar alphabets
     */
    if (isalnum(word[0]) || word[0] == '$' || word[0] == '\033') {
      bool upperCase = startSentence ^ flipCase;
      string = ifWord;
      if (upperCase) {
        string += toupper(word[0]);
        string += word.c_str() + 1;
      } else {
        string += word;
      }
      wasWord = true;
      ifWord = " ";
      startSentence = false;
    } else {
      std::string oldIfWord = wasWord ? ifWord : "";
      ifWord = "";
      switch (word[0]) {
        case '.':
        case '!':
        case '?':
          startSentence = true;
          string += word;
          endParagraph = isParagraph;
          ifWord = endParagraph ? "\n\n" : "  ";
          break;
        case '(':
          string += oldIfWord + word;
          ++inParentheses;
          break;
        case ')':
          --inParentheses;
          /*@fallthrough@*/
        case ',': case ';': case ':':
          string += word;
          ifWord = " ";
          break;
        case '\'':
          if (inSQuotes) {
            string += word;
            ifWord = " ";
          } else {
            string += oldIfWord + word;
          }
          inSQuotes = !inSQuotes;
          break;
        case '"':
          if (inDQuotes) {
            string += word;
            ifWord = " ";
          } else {
            string += oldIfWord + word;
          }
          inDQuotes = !inDQuotes;
          break;
        default:
          string += word;
      }
      wasWord = false;
    }
    if (lineWidth + string.length() >= 78) {
      lineWidth = string.length();
      bool fail = true;
      for (auto& c : string) {
        lineWidth--;
        if (c == ' ') {
          c = '\n';
          fail = false;
          break;
        }
      }
      if (fail) {
        lineWidth = string.length();
        string = "\n" + string;
      }
    } else {
      lineWidth += string.length();
    }
    return string;
  }

  template<class Context>
  WordToken synthesizeWord(std::string& string, RangeCoder& rng, Context ctx) {
    DebugLog dbg(NULL);
    return decodeWord<Context>(string, rng, ctx, dbg);
  }

  std::string synthesizeParagraph(RangeCoder& rng, uint32_t seed = 0) {
    std::string string;
    WordContext ctx(words, seed);
//    ctx.setCeiling(WordToken::period());  // TODO: this is cack!
    while (!endParagraph) {
      auto t = WordToken::fromInt(rng.decode(ctx.getStats()));
      std::string word;
      if (t.needsLetters()) t = synthesizeWord(word, rng, LetterContext(letters));
      else if (t.needsDigits()) t = synthesizeWord(word, rng, NumberContext(numbers));
      else word = t.toString();
      string += decorate(word, isParagraph(string));
      ctx.updateState(t);
    }
    resetDecor();
    return string;
  }

  virtual bool isParagraph(std::string const& string) {
    return string.length() > 80 && (random() % 4) == 0;
  }
  virtual uint64_t choose(uint64_t range) {
    return random() % range;
  }
};


int analyze(WordMumble& words, LetterMumble& letters, NumberMumble& numbers,
    std::istream& is, bool verbose = false) {
  Tokenizer tok(is, true, verbose);
  WordContextRW ctx(words);
  for (;;) {
    auto t = tok.next();
    if (t.isNothing()) break;
    if (t.isDictionaryWord() || t.isOtherWord()) {
      auto& s = tok.spellThat();
      LetterContextRW ctx(letters);  // TODO: think about adding `, t` here
      for (auto c : s) {
        auto t = LetterContextRW::TokenType::fromChar(c);
        ctx.updateStats(t);
        ctx.updateState(t);
      }
      ctx.updateStats(ctx.end(t.isDictionaryWord()));
    }
    if (t.isNumber()) {
      auto& s = tok.spellThat();
      NumberContextRW ctx(numbers);  // TODO: think about adding `, t` here
      for (auto c : s) {
        auto t = NumberToken::fromChar(c);
        ctx.updateStats(t);
        ctx.updateState(t);
      }
      ctx.updateStats(ctx.end());
    }
    ctx.updateStats(t);
    ctx.updateState(t);
  }
  return 0;
}

int analyze(WordMumble& words, LetterMumble& letters, NumberMumble& numbers,
    char const* text, bool verbose = false) {
  std::filebuf fb;
  if (!fb.open(text, std::ios::in)) return -1;

  if (verbose) {
    printf("analyzing %s...\n", text);
    fflush(stdout);
  }

  std::istream is(&fb);
  int r = analyze(words, letters, numbers, is, verbose);
  fb.close();
  return r;
}

void synthesize(WordMumble const& words, LetterMumble const& letters, NumberMumble const& numbers) {
  class RNG : public RangeCoder {
    virtual void putbit(int b) override {}
  } rng;
  MumbleStream ms(words, letters, numbers);

  for (int i = 0; i < 100; i++) {
    std::string word;
    ms.synthesizeWord(word, rng, LetterContext(letters));
    printf("%s\n", word.c_str());
  }
  for (int i = 0; i < 20; i++) {
    std::string word;
    ms.synthesizeWord(word, rng, NumberContext(numbers));
    printf("%s\n", word.c_str());
  }
#if 1
  for (int i = 0; i < 100; i++) {
    auto sent = ms.synthesizeParagraph(rng);
    printf("%s\n\n", sent.c_str());
    fflush(stdout);
  }
#endif
}


void encode(std::ostream& os, std::istream& is,
    WordMumble const& words, LetterMumble const& letters, NumberMumble const& numbers,
    std::vector<std::string>* log) {
  const bool debug = true;
  const bool verbose = false;
  DebugLog dbg(log);
  struct StringEncoder : public RangeCoder {
    std::ostream& os;
    StringEncoder(std::ostream& s) : os(s) {}
    virtual void putbit(int b) override { os.put(b ? '1' : '0'); }
    virtual int getbit() override { return 0; }
  } enc(os);
  Tokenizer tok(is);
  WordContext ctx(words);

  dbg.setVerbose(verbose && debug);

  for (;;) {
    auto t = tok.next();
    if (t.isNothing()) break;
    std::string capture;
    if (debug) enc.setLog(&capture);

    enc.encode(ctx.getStats(), t.toInt());
    ctx.updateState(t);
    dbg.push_back(enc.log() + " " + t.toString());
    if (t.needsLetters()) {
      auto& s = tok.spellThat();
      auto tt = encodeWord(enc, s, LetterContext(letters), dbg);
      assert(tt.hash() == t.hash());
    } else if (t.needsDigits()) {
      auto& s = tok.spellThat();
      auto tt = encodeWord(enc, s, NumberContext(numbers), dbg);
      assert(tt.hash() == t.hash());
    }
  }
}

void decode(std::ostream& os, std::istream& is,
    WordMumble const& words, LetterMumble const& letters, NumberMumble const& numbers,
    std::vector<std::string>* log) {
  const bool debug = true;
  const bool verbose = false;
  DebugLog dbg(debug ? log : NULL);
  MumbleStream ms(words, letters, numbers);
  struct StringDecoder : public RangeCoder {
    std::istream& is;
    size_t i = 0;
    uint8_t extend = 4;
    StringDecoder(std::istream& s) : is(s) { consume(64); }
    virtual void putbit(int) override {}
    virtual int getbit() override {
      int c = 0;
      while (is.good() && (c = is.get()) != EOF) {
        if (isxdigit(c)) return c & 1;
      }
      if (extend > 0) --extend;
      return 1;
    }
    bool eof() {
      return extend == 0;
    }
  } dec(is);
  WordContext ctx(words);

  dbg.setVerbose(verbose && debug);

  while (!dec.eof()) {
    std::string declog;
    dec.setLog(&declog);
    auto t = WordToken::fromInt(dec.decode(ctx.getStats()));
    dbg.push_back(dec.log() + " " + t.toString());

    std::string s;
    if (t.needsLetters()) {
      t = decodeWord(s, dec, LetterContext(letters), dbg);
    }
    else if (t.needsDigits()) {
      t = decodeWord(s, dec, NumberContext(numbers), dbg);
    } else {
      s = t.toString();
    }
    assert(!t.isNothing());
    ctx.updateState(t);
    s = ms.decorate(s, false);
    os.write(s.c_str(), s.length());
  }
}


void encode(std::string& os, std::istream& is,
    WordMumble const& words, LetterMumble const& letters, NumberMumble const& numbers,
    std::vector<std::string>* log) {
  std::ostringstream oss;
  encode(oss, is, words, letters, numbers, log);
  os = oss.str();
}

void decode(std::ostream& os, std::string const& is,
    WordMumble const& words, LetterMumble const& letters, NumberMumble const& numbers,
    std::vector<std::string>* log) {
  std::istringstream iss(is);
  decode(os, iss, words, letters, numbers, log);
}

void compare(std::vector<std::string> const& left, std::vector<std::string> const& right) {
  for (size_t i = 0; i < left.size() && i < right.size(); i++) {
    printf("l: %s\nr: %s\n", left[i].c_str(), right[i].c_str());
  }
}

void swizzle(std::ostream& out, std::istream& in,
    WordMumble const& words, LetterMumble const& letters, NumberMumble const& numbers,
    std::vector<std::string>* elog, std::vector<std::string>* dlog) {
  std::string bitstream;
  if (elog) elog->clear();
  printf("<enc>"); fflush(stdout);
  encode(bitstream, in, words, letters, numbers, elog);

  int tog = 0;
  for (auto& c : bitstream) {
    if (isdigit(c)) c = c ^ tog;
    tog = !tog;
  }

  if (dlog) dlog->clear();
  printf("<dec>"); fflush(stdout);
  decode(out, bitstream, words, letters, numbers, dlog);
}

void transcode(WordMumble const& words, LetterMumble const& letters, NumberMumble const& numbers, bool verbose) {
  std::vector<std::string> debug_encode;
  std::vector<std::string> debug_decode;
  std::ostringstream ciphertext;

  printf("ciphertext:");
  fflush(stdout);

  swizzle(ciphertext, std::cin, words, letters, numbers, &debug_encode, &debug_decode);

  printf("\n%s\n", ciphertext.str().c_str());
  fflush(stdout);

  if (verbose) compare(debug_encode, debug_decode);
  std::ostringstream cleartext;
  std::istringstream isct(ciphertext.str());

  printf("cleartext:");
  fflush(stdout);

  swizzle(cleartext, isct, words, letters, numbers, &debug_encode, NULL);

  printf("\n%s\n", cleartext.str().c_str());
  fflush(stdout);

  if (verbose) compare(debug_decode, debug_encode);
}


int main(int argc, char *argv[]) {
  {
    uint64_t s;
    int fd = open("/dev/urand", O_RDONLY);
    if (fd < 0 || read(fd, &s, sizeof(s)) != sizeof(s)) s = time(NULL);
    close(fd);
    srandom(s);
  }
  bool verbose = false;
  bool test = false;
  bool synthesis = false;
  int opt;

  while ((opt = getopt(argc, argv, "stv")) != -1) {
    switch (opt) {
    case 's':
      synthesis = true;
      break;
    case 't':
      test = true;
      break;
    case 'v':
      verbose = true;
    }
  }

  if (verbose) setvbuf(stdout, NULL, _IOLBF, 128);
  WordToken::init("dictionary.txt", verbose);
  WordMumble words(WordToken::range());
  LetterMumble letters(LetterToken::range());
  NumberMumble numbers(NumberToken::range());
  analyze(words, letters, numbers, "corpus.txt", verbose);
  if (synthesis) synthesize(words, letters, numbers);
  if (test) transcode(words, letters, numbers, verbose);
  if (!synthesis && !test) swizzle(std::cout, std::cin, words, letters, numbers, NULL, NULL);

  return 0;
}
