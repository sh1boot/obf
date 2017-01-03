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
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <functional>
#include <type_traits>

class Token {
 protected:
  uint16_t value;
  static const uint16_t nothingValue = UINT16_MAX;
  bool inRange(uint16_t l, uint16_t r) const { return l <= value && value < (l + r); }

 public:
  explicit constexpr Token(uint16_t i = nothingValue) : value(i) {}
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

static char constexpr letterAlphabet[] = "abcdefghijklmnopqrstuvwxyz'";
static char constexpr separatorAlphabet[] = ".!?,;:()[]-/\"'*";
static char constexpr numberAlphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789$.,";
typedef LetterlikeToken<letterAlphabet> LetterToken;
typedef LetterlikeToken<separatorAlphabet, false> SeparatorToken;
typedef LetterlikeToken<numberAlphabet> NumberToken;

class WordToken : public Token {
  static std::unordered_map<std::string, uint16_t> words;
  static std::unordered_map<std::string, uint16_t> suffixes;
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

  uint16_t hashValue;

 public:
  WordToken() : WordToken(nothingValue, nothingValue) {}
  WordToken(uint16_t i, uint16_t h) : Token(i), hashValue(h) {}
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
  /* TODO: eliminate these... */
  static ssize_t lookupWord(std::string const& s) {
    auto it = words.find(s);
    if (it != words.end()) {
      return it->second;
    }
    return -1;
  }
  static ssize_t lookupOtherWord(std::string const& s) {
    uint32_t h = 0xdeadbeef;
    for (auto c : s) {
      h = ((h ^ (h >> 15)) + c * 0x12345) * 31;
    }
    return (h ^ (h >> 17));
  }
};

class WordParser {
  template<typename T>
  class Parser {
   public:
    using TokenType = T;
    Parser() : illegalTokens(T::nothing()) {}
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
    //std::static_assert(std::is_base_of<Parser<T::TokenType>, T>::value, "bad T");

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

  class OtherWordParser_Middle : public Parser<LetterToken> {
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

  class NumberParser_Middle : public Parser<NumberToken> {
    bool seenDigit;

   protected:
    void reset() {
      seenDigit = false;
      /* can't have a . or , before we see a digit. */
      illegalTokens = NumberToken::fromChar('.');
    }
    void update(NumberToken t) {
      seenDigit = seenDigit || t.isDigit();
      if (!seenDigit) {
        illegalTokens = NumberToken::fromChar('.');
      } else if (!t.isAlnum()) {
        illegalTokens = NumberToken::end();
      }
    }
    WordToken get(uint32_t hash) const {
      if (seenDigit) return WordToken::numberWord(0);
      else return WordToken::nothing();
    }
  };

  class SeparatorParser_Middle : public Parser<SeparatorToken> {
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
std::unordered_map<std::string, uint16_t> WordToken::words;
std::unordered_map<std::string, uint16_t> WordToken::suffixes;
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
    if (isalpha(s[0]) && words.emplace(s, count).second) count++;
  }
  fb.close();

  if (verbose) printf("dictionary is %zu words.\n", words.size());
  /* TODO: suffixes */

  for (auto c : separatorAlphabet) {
    if (c != '\0') {
      separatorStrings.emplace_back(std::string(1, c));
    }
  }

  wordList.resize(words.size());
  for (auto& it : words) wordList[it.second] = &it.first;
  suffixList.resize(suffixes.size());
  for (auto& it : suffixes) suffixList[it.second] = &it.first;

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


template <unsigned D>
class HashState_T {
  static const uint64_t m = 0x98b5892bb6fb97d9;
  uint16_t history[D];
  unsigned histpos = 0;
  uint64_t hash = 0;

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
  HashState_T(void) { reset(); }
  void reset(void) {
    memset(history, 0, sizeof(history));
    histpos = 0;
    hash = 0;
  }
  void add(uint32_t t) {
    hash = hash - history[histpos] * lag();
    t *= 0xb16d5a03;
    t ^= t >> 15;
    history[histpos] = t;
    hash = (hash + history[histpos]) * m;
    histpos = (histpos + 1) % D;

    assert(hash == FullHash());
  }
  uint32_t get(unsigned bits) const {
    return (uint32_t)(hash ^ (hash >> 32)) >> (32 - bits);
  }
};


struct RangeCoderProb {
  uint64_t getTotal() const { return total; }
  virtual uint64_t getSpan(int symbol, uint64_t& base, uint64_t& size) const = 0;
  virtual int getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const = 0;

 protected:
  uint64_t total;
};

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
    uint64_t off = (bs - low) * (uint128_t)total / range;
    uint64_t base, size;
    int r = p.getSymbol(off, base, size);
    encode(base, size, total);
    return r;
  }

  void encode(RangeCoderProb const& p, int symbol) {
    uint64_t base, size, total;
    total = p.getSpan(symbol, base, size);
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

template<typename T>
struct TokenParser : public WordParser {
//  void start() { WordParser::start(); }
  bool nextChar(T t) { return WordParser::nextChar(t.toChar()); }
  T getIllegalTokens() const { return WordParser::getIllegalTokens<T>(); }
  T end() const { assert(WordParser::isOk<T>()); return T::end(); }
  WordToken getToken() const { return WordParser::getToken<T>(); }
};

template<>
struct TokenParser<WordToken> {
  WordToken v;
  static void start() {}
  void nextChar(WordToken t) { v = t; }
  static WordToken getIllegalTokens() { return WordToken::nothing(); }
  WordToken end() const { return WordToken::nothing(); }
  WordToken getToken() const { return v; }
};

template<typename T, int hashDist = 4>
class Context_T : public TokenParser<T> {
  using super = TokenParser<T>;
  void integrate(uint32_t t) {
    h.add(t);
    hb.add(t);
    hc.add(t);
  }
  void reset() {
    super::start();
    h.reset();
    hb.reset();
    hc.reset();
  }
  void reset(WordToken t) {
    reset();
    /* If we seed with a token, then use its canonical value, not its hash.
    * This is because the seed mechanism is intended to prepare a decoder to
    * produce a string we don't yet know, so its hash is unknown.
    */
    if (!t.isNothing()) integrate(t.toInt());
  }

 public:
  using TokenType = T;
  HashState_T<hashDist> h;
  HashState_T<2> hb;
  HashState_T<1> hc;

  Context_T() : super() { reset(); }
  Context_T(WordToken t) : super() { reset(t); }

  void integrate(TokenType t) {
    assert(!t.isNothing());
    integrate(t.hash());
    super::nextChar(t);
  }
};

template<typename Ctx, int hashBits = 18,
        uint64_t fscale = 16384, uint64_t fscale_b = 2, uint64_t fscale_c = 2>
class Mumble_T {
  uint16_t tableSize;

  uint32_t* freqMap;
  uint32_t summMap[(1 << hashBits) + 1];

 public:
  typedef Ctx Context;
  class RCProb : public RangeCoderProb {
    Mumble_T const& source;
    Context const& context;

    uint64_t getSpan_(int symbol, uint64_t& base, uint64_t& size) const {
      base = 0;
      for (int i = 0; i < symbol; i++) base += source.score(context, i);
      size = source.score(context, symbol);
      return total;
    }

    int getSymbol_(uint64_t off, uint64_t& base, uint64_t& size) const {
      assert(off < total);
      base = 0;
      int symbol = 0;
      for (base = 0; base < total; base += size, symbol++) {
        size = source.score(context, symbol);
        if (off < base + size) {
          return symbol;
        }
      }
      assert(!"findToken() failed");
      return -1;
    }

   public:
    RCProb(Mumble_T const& src, Context const& ctx)
      : source(src), context(ctx) {
      total = source.total(context);
    }

    virtual uint64_t getSpan(int symbol, uint64_t& base, uint64_t& size) const override;
    virtual int getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const override;
  };

 private:
  uint32_t const* freq() const {
    return &freqMap[tableSize << hashBits];
  }
  uint32_t const* freq(Context const& ctx) const {
    return &freqMap[ctx.h.get(hashBits) * tableSize];
  }
  uint32_t const* freq_b(Context const& ctx) const {
    return &freqMap[ctx.hb.get(hashBits) * tableSize];
  }
  uint32_t const* freq_c(Context const& ctx) const {
    return &freqMap[ctx.hc.get(hashBits) * tableSize];
  }
  uint32_t const& summ() const {
    return summMap[1 << hashBits];
  }
  uint32_t const& summ(Context const& ctx) const {
    return summMap[ctx.h.get(hashBits)];
  }
  uint32_t const& summ_b(Context const& ctx) const {
    return summMap[ctx.hb.get(hashBits)];
  }
  uint32_t const& summ_c(Context const& ctx) const {
    return summMap[ctx.hc.get(hashBits)];
  }

  Mumble_T const& cthis() { return *this; }
  uint32_t* freq() { return const_cast<uint32_t*>(cthis().freq()); }
  uint32_t* freq(Context const& ctx) { return const_cast<uint32_t*>(cthis().freq(ctx)); }
  uint32_t* freq_b(Context const& ctx) { return const_cast<uint32_t*>(cthis().freq_b(ctx)); }
  uint32_t* freq_c(Context const& ctx) { return const_cast<uint32_t*>(cthis().freq_c(ctx)); }
  uint32_t& summ() { return const_cast<uint32_t&>(cthis().summ()); }
  uint32_t& summ(Context const& ctx) { return const_cast<uint32_t&>(cthis().summ(ctx)); }
  uint32_t& summ_b(Context const& ctx) { return const_cast<uint32_t&>(cthis().summ_b(ctx)); }
  uint32_t & summ_c(Context const& ctx) { return const_cast<uint32_t&>(cthis().summ_c(ctx)); }


  uint64_t score(Context const& ctx, int i) const {
    return 1 + fscale   * freq(ctx)[i]
             + fscale_b * freq_b(ctx)[i]
             + fscale_c * freq_c(ctx)[i];
  }

  uint64_t total(Context const& ctx) const {
    uint64_t total = tableSize + fscale   * summ(ctx)
                               + fscale_b * summ_b(ctx)
                               + fscale_c * summ_c(ctx);
    if (!ctx.getIllegalTokens().isNothing()) {
      for (int i = ctx.getIllegalTokens().toInt(); i < tableSize; i++)
        total -= score(ctx, i);
    }
    return total;
  }

  void inc(Context const& ctx, int i) {
    ++freq()[i];
    ++summ();
    ++freq(ctx)[i];
    ++summ(ctx);
    ++freq_b(ctx)[i];
    ++summ_b(ctx);
    ++freq_c(ctx)[i];
    ++summ_c(ctx);
  }

 public:
  Mumble_T(int tmax) : tableSize(tmax) {
    size_t sz = (tableSize << hashBits) + tableSize;
    freqMap = reinterpret_cast<uint32_t*>(calloc(sz, sizeof(*freqMap)));
    assert(freqMap != NULL);
    memset(summMap, 0, sizeof(summMap));
  }

  RCProb getProb(Context const& ctx) const {
    return RCProb(*this, ctx);
  }

  void integrate(Context const& ctx, Token t) {
    if (t.isNothing()) return;
    uint16_t i = t.toInt();
    assert(i < tableSize);
    inc(ctx, i);
  }
};

typedef Mumble_T<Context_T<WordToken>, 15> WordMumble;
typedef Mumble_T<Context_T<LetterToken, 6>, 20, 16, 1, 1> LetterMumble;
typedef Mumble_T<Context_T<NumberToken, 6>, 8, 16, 1, 1> NumberMumble;

template<>
uint64_t WordMumble::RCProb::getSpan(int symbol, uint64_t& base, uint64_t& size) const {
  return getSpan_(symbol, base, size);
}
template<>
uint64_t LetterMumble::RCProb::getSpan(int symbol, uint64_t& base, uint64_t& size) const {
  return getSpan_(symbol, base, size);
}
template<>
uint64_t NumberMumble::RCProb::getSpan(int symbol, uint64_t& base, uint64_t& size) const {
  return getSpan_(symbol, base, size);
}
template<>
int WordMumble::RCProb::getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const {
  return getSymbol_(off, base, size);
}
template<>
int LetterMumble::RCProb::getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const {
  return getSymbol_(off, base, size);
}
template<>
int NumberMumble::RCProb::getSymbol(uint64_t off, uint64_t& base, uint64_t& size) const {
  return getSymbol_(off, base, size);
}


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

  template<class Mumble>
  static WordToken decodeWord(std::string& s, RangeCoder& dec,
      Mumble const& alphabet, DebugLog& dbg,
      WordToken seed = WordToken::nothing()) {
    s.clear();
    typename Mumble::Context ctx(seed);
    for (;;) {
      int i = dec.decode(alphabet.getProb(ctx));
      auto t = Mumble::Context::TokenType::fromInt(i);
      if (t.isEnd()) {
        dbg.push_back(dec.log() + " \"" + s + "\"");
        break;
      }
      s += t.toChar();
      ctx.integrate(t);
    }
    assert(s.length() > 0);

    /* For hashing we integrate the hash of the unknown word, rather than
     * just the constant 'other' token.  We can do this only after we've
     * decoded the word, so we need to update the caller now.
     */
    return ctx.getToken();
  }

  template <class Mumble>
  static void encodeWord(RangeCoder& enc, std::string const& s,
        Mumble const& alphabet, DebugLog& dbg,
        WordToken seed = WordToken::nothing()) {
    typename Mumble::Context ctx(seed);
    for (auto c : s) {
      auto t = Mumble::Context::TokenType::fromChar(c);
      assert(!t.isNothing() && t < ctx.getIllegalTokens());
      enc.encode(alphabet.getProb(ctx), t.toInt());
      ctx.integrate(t);
    }
    assert(!ctx.getToken().isNothing());
    enc.encode(alphabet.getProb(ctx), ctx.end().toInt());
    dbg.push_back(enc.log() + " \"" + s + "\"");
  }

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
    if (isalnum(word[0]) || word[0] == '\033') {
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

  template<class Mumble>
  WordToken synthesizeWord(std::string& string, RangeCoder& rng, Mumble const& alphabet, WordToken seed = WordToken::nothing()) {
    DebugLog dbg(NULL);
    return decodeWord<Mumble>(string, rng, alphabet, dbg, seed);
  }

  std::string synthesizeParagraph(RangeCoder& rng, WordToken seed = WordToken::nothing()) {
    std::string string;
    WordMumble::Context ctx(seed);
//    ctx.setCeiling(WordToken::period());  // TODO: this is cack!
    while (!endParagraph) {
      auto t = WordToken::fromInt(rng.decode(words.getProb(ctx)));
      std::string word;
      if (t.needsLetters()) t = synthesizeWord(word, rng, letters);
      else if (t.needsDigits()) t = synthesizeWord(word, rng, numbers);
      else word = t.toString();
      string += decorate(word, isParagraph(string));
      ctx.integrate(t);
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
  WordMumble::Context ctx;
  for (;;) {
    auto t = tok.next();
    if (t.isNothing()) break;
    if (t.isDictionaryWord() || t.isOtherWord()) {
      auto& s = tok.spellThat();
      LetterMumble::Context ctx;
      for (auto c : s) {
        auto t = LetterToken::fromChar(c);
        letters.integrate(ctx, t);
        ctx.integrate(t);
      }
      letters.integrate(ctx, LetterToken::end());
    }
    if (t.isNumber()) {
      auto& s = tok.spellThat();
      NumberMumble::Context ctx;
      for (auto c : s) {
        auto t = NumberToken::fromChar(c);
        numbers.integrate(ctx, t);
        ctx.integrate(t);
      }
      numbers.integrate(ctx, NumberToken::end());
    }
    words.integrate(ctx, t);
    ctx.integrate(t);
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
    ms.synthesizeWord(word, rng, letters);
    printf("%s\n", word.c_str());
  }
  for (int i = 0; i < 20; i++) {
    std::string word;
    ms.synthesizeWord(word, rng, numbers);
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
  WordMumble::Context ctx;

  dbg.setVerbose(verbose && debug);

  for (;;) {
    auto t = tok.next();
    if (t.isNothing()) break;
    std::string capture;
    if (debug) enc.setLog(&capture);

    enc.encode(words.getProb(ctx), t.toInt());
    ctx.integrate(t);
    dbg.push_back(enc.log() + " " + t.toString());
    if (t.needsLetters()) {
      auto& s = tok.spellThat();
      MumbleStream::encodeWord(enc, s, letters, dbg);
    } else if (t.needsDigits()) {
      auto& s = tok.spellThat();
      MumbleStream::encodeWord(enc, s, numbers, dbg);
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
  WordMumble::Context ctx;

  dbg.setVerbose(verbose && debug);

  while (!dec.eof()) {
    std::string declog;
    dec.setLog(&declog);
    auto t = WordToken::fromInt(dec.decode(words.getProb(ctx)));
    dbg.push_back(dec.log() + " " + t.toString());

    std::string s;
    if (t.needsLetters()) {
      t = ms.decodeWord(s, dec, letters, dbg);
    }
    else if (t.needsDigits()) {
      t = ms.decodeWord(s, dec, numbers, dbg);
    } else {
      s = t.toString();
    }
    assert(!t.isNothing());
    ctx.integrate(t);
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
  encode(bitstream, in, words, letters, numbers, elog);

  int tog = 0;
  for (auto& c : bitstream) {
    if (isdigit(c)) c = c ^ tog;
    tog = !tog;
  }

  if (dlog) dlog->clear();
  decode(out, bitstream, words, letters, numbers, dlog);
}

void transcode(WordMumble const& words, LetterMumble const& letters, NumberMumble const& numbers) {
  std::vector<std::string> debug_encode;
  std::vector<std::string> debug_decode;
  std::ostringstream ciphertext;
  swizzle(ciphertext, std::cin, words, letters, numbers, &debug_encode, &debug_decode);
  //compare(debug_encode, debug_decode);
  std::ostringstream cleartext;
  std::istringstream isct(ciphertext.str());
  swizzle(cleartext, isct, words, letters, numbers, &debug_encode, NULL);
  //compare(debug_decode, debug_encode);

  printf("ciphertext:\n%s\n", ciphertext.str().c_str());
  fflush(stdout);

  printf("cleartext:\n%s\n", cleartext.str().c_str());
  fflush(stdout);
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
  //analyze(words, letters, numbers, "monolith.txt");
  if (synthesis) synthesize(words, letters, numbers);
  if (test) transcode(words, letters, numbers);
  if (!synthesis && !test) swizzle(std::cout, std::cin, words, letters, numbers, NULL, NULL);

  return 0;
}
