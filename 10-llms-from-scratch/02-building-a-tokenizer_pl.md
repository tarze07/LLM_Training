# Budowa Tokenizatora od Podstaw

> Lekcja 01 dała ci zabawkę. Ta lekcja daje ci broń.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## Learning Objectives

- Zbuduj produkcyjny tokenizator BPE, który obsługuje Unicode, normalizację białych znaków i specjalne tokeny
- Zaimplementuj mechanizm awaryjny na poziomie bajtów, aby tokenizator mógł kodować dowolne wejście (w tym emoji, CJK i kod) bez nieznanych tokenów
- Dodaj wzorce regex pre-tokenizacji, które dzielą tekst na granicach słów przed zastosowaniem łączeń BPE
- Wytrenuj niestandardowy tokenizator na korpusie i oceń jego współczynnik kompresji w porównaniu z tiktoken na tekście wielojęzycznym

## Problem

Twój tokenizator BPE z Lekcji 01 działa na tekście angielskim. Teraz rzuć na niego japoński. Albo emoji. Albo kod Pythona z mieszanką tabulatorów i spacji.

Psuje się.

Nie dlatego, że BPE jest złe -- dlatego, że implementacja jest niekompletna. Produkcyjny tokenizator obsługuje surowe bajty w dowolnym kodowaniu, normalizuje Unicode przed dzieleniem, zarządza specjalnymi tokenami, które nigdy nie są łączone, łączy pre-tokenizację z dzieleniem podwyrazowym i robi to wszystko wystarczająco szybko, aby nie stanowić wąskiego gardła dla potoku treningowego przetwarzającego 15 bilionów tokenów.

Tokenizator GPT-2 ma 50,257 tokenów. Llama 3 ma 128,256. GPT-4 ma około 100,000. To nie są zabawkowe liczby. Tabele łączeń stojące za tymi słownikami były trenowane na setkach gigabajtów tekstu, a otaczająca je infrastruktura -- normalizacja, pre-tokenizacja, wstrzykiwanie specjalnych tokenów, formatowanie szablonów czatu -- to oddziela tokenizator, który radzi sobie z "hello world", od takiego, który radzi sobie z całym internetem.

Zbudujesz tę infrastrukturę.

## Koncepcja

### Pełny Potok

Produkcyjny tokenizator to nie jeden algorytm. To potok pięciu etapów, z których każdy rozwiązuje inny problem.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Każdy etap ma określone zadanie:

| Etap | Co robi | Dlaczego to ważne |
|-------|-------------|----------------|
| Normalizacja | NFKC Unicode, opcjonalnie małe litery, opcjonalnie usuwanie akcentów | Ligatura "fi" (U+FB01) staje się "fi" (dwa znaki). Bez tego to samo słowo dostaje różne tokeny. |
| Pre-tokenizacja | Dzieli tekst na fragmenty przed BPE | Zapobiega łączeniu przez BPE między granicami słów. "the cat" nigdy nie powinno wyprodukować tokena "e c". |
| Łączenie BPE | Stosuje nauczone reguły łączenia do sekwencji bajtów | Rdzeń kompresji. Zamienia surowe bajty w tokeny podwyrazowe. |
| Specjalne tokeny | Wstrzykuje [BOS], [EOS], [PAD], znaczniki szablonu czatu | Te tokeny mają stałe ID. Nigdy nie uczestniczą w łączeniach BPE. Model potrzebuje ich do struktury. |
| Mapowanie ID | Konwertuje ciągi tokenów na identyfikatory całkowite | Model widzi liczby całkowite, nie ciągi znaków. |

### Byte-Level BPE

Tokenizator z Lekcji 01 operował na bajtach UTF-8. To był słuszny wybór. Ale pominęliśmy coś ważnego: co się dzieje, gdy te bajty nie są prawidłowym UTF-8?

Byte-level BPE rozwiązuje to, traktując każdą możliwą wartość bajtu (0-255) jako prawidłowy token. Twój bazowy słownik to dokładnie 256 wpisów. Dowolny plik -- tekstowy, binarny, uszkodzony -- może być tokenizowany bez produkowania nieznanego tokena.

GPT-2 dodał sztuczkę: mapuje każdy bajt na drukowalny znak Unicode, aby słownik pozostał czytelny dla człowieka. Bajt 0x20 (spacja) staje się znakiem "G" w ich mapowaniu. To czysto kosmetyczne. Algorytm nie przejmuje się tym.

Prawdziwa moc: byte-level BPE obsługuje każdy język na ziemi. Chińskie znaki to po 3 bajty UTF-8 każdy. Japoński może mieć 3-4 bajty. Arabski, dewanagari, emoji -- wszystko to tylko sekwencje bajtów. Algorytm BPE znajduje wzorce w tych sekwencjach bajtów dokładnie tak samo, jak znajduje wzorce w angielskich bajtach ASCII.

### Pre-Tokenizacja

Zanim BPE dotknie twojego tekstu, musisz podzielić go na fragmenty. To zapobiega tworzeniu przez algorytm łączenia tokenów obejmujących granice słów.

GPT-2 używa wzorca regex do dzielenia tekstu:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Ten wzorzec dzieli na skróceniach ("don't" staje się "don" + "'t"), słowach z opcjonalną spacją wiodącą, liczbach, znakach interpunkcyjnych i białych znakach. Spacja wiodąca pozostaje przyczepiona do słowa -- więc "the cat" staje się [" the", " cat"], a nie ["the", " ", "cat"].

Llama używa SentencePiece, który pomija regex całkowicie. Traktuje surowy strumień bajtów jako jedną długą sekwencję i pozwala algorytmowi BPE samemu znaleźć granice. To prostsze, ale daje BPE więcej swobody do tworzenia tokenów międzywyrazowych.

Wybór ma znaczenie. Regex GPT-2 zapobiega nauczeniu się przez tokenizator, że "the" na końcu jednego słowa i "the" na początku następnego powinny się połączyć. SentencePiece na to pozwala, co czasami produkuje wydajniejszą kompresję, ale mniej interpretowalne tokeny.

### Specjalne Tokeny

Każdy produkcyjny tokenizator rezerwuje identyfikatory tokenów dla znaczników strukturalnych:

| Token | Cel | Używany przez |
|-------|---------|---------|
| `[BOS]` / `<s>` | Początek sekwencji | Llama 3, GPT |
| `[EOS]` / `</s>` | Koniec sekwencji | Wszystkie modele |
| `[PAD]` | Wypełnianie do wyrównania wsadu | BERT, T5 |
| `[UNK]` | Nieznany token (byte-level BPE eliminuje to) | BERT, WordPiece |
| `<\|im_start\|>` | Granica wiadomości czatu | ChatGPT, Qwen |
| `<\|im_end\|>` | Granica wiadomości czatu | ChatGPT, Qwen |
| `<\|user\|>` | Znacznik tury użytkownika | Llama 3 |
| `<\|assistant\|>` | Znacznik tury asystenta | Llama 3 |

Specjalne tokeny nigdy nie są dzielone przez BPE. Są dopasowywane dokładnie przed uruchomieniem algorytmu łączenia, zastępowane swoim stałym ID, a otaczający tekst jest tokenizowany normalnie.

### Szablony Czatu

To jest miejsce, gdzie większość ludzi się gubi i gdzie większość implementacji się psuje.

Gdy wysyłasz wiadomości do modelu czatu, API akceptuje listę wiadomości:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

Model nie widzi JSON-a. Widzi płaską sekwencję tokenów. Szablon czatu konwertuje wiadomości na tę płaską sekwencję za pomocą specjalnych tokenów. Każdy model robi to inaczej:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Jeśli pomylisz szablon, model produkuje śmieci. Został wytrenowany na jednym dokładnym formacie. Każde odstępstwo -- brakujący znak nowej linii, zamieniony token, dodatkowa spacja -- umieszcza wejście poza dystrybucją treningową.

### Szybkość

Python jest zbyt wolny do produkcyjnej tokenizacji.

tiktoken (OpenAI) jest napisany w Rust z wiązaniami Pythona. HuggingFace tokenizers to również Rust. SentencePiece to C++. Osiągają one 10-100x przyspieszenie w porównaniu do czystego Pythona.

Dla perspektywy: tokenizacja 15 bilionów tokenów do pre-treningu Llamy 3 przy 1 milionie tokenów na sekundę (szybki Python) zajęłaby 174 dni. Przy 100 milionach tokenów na sekundę (Rust) zajmuje 1.7 dnia.

Budujesz w Pythonie, aby zrozumieć algorytm. W produkcji użyłbyś skompilowanej implementacji i dotykał tylko opakowania Pythona.

```figure
weight-tying
```

## Zbuduj To

### Krok 1: Kodowanie na Poziomie Bajtów

Fundament. Konwertuj dowolny ciąg na sekwencję bajtów, mapuj każdy bajt na drukowalny znak do wyświetlenia i odwróć proces.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Przetestuj na tekście wielojęzycznym, aby zobaczyć liczbę bajtów:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" to 5 bajtów. "你好" to 6 bajtów (3 na znak). Emoji ognia to 4 bajty. Tokenizator na poziomie bajtów nie przejmuje się, jaki to język. Bajty to bajty.

### Krok 2: Pre-Tokenizator z Regex

Dziel tekst na fragmenty za pomocą wzorca regex GPT-2. Każdy fragment jest tokenizowany niezależnie przez BPE.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Moduł `regex` wspiera sekwencje ucieczki Unicode (`\p{L}` dla liter, `\p{N}` dla liczb). Standardowy moduł biblioteczny `re` nie wspiera, więc cofamy się do klas znaków ASCII. Dla produkcyjnych wielojęzycznych tokenizatorów zainstaluj `regex`.

Wypróbuj:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Spacja wiodąca pozostaje przyczepiona do słowa. Skrócenia dzielą się przy apostrofie. Znaki interpunkcyjne stają się własnym fragmentem. BPE nigdy nie połączy tokenów przez te granice.

### Krok 3: BPE na Sekwencjach Bajtów

Podstawowy algorytm z Lekcji 01, ale teraz działający na pre-tokenizowanych fragmentach niezależnie.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Krok 4: Obsługa Specjalnych Tokenów

Specjalne tokeny wymagają dokładnego dopasowania i stałych ID. Omijają BPE całkowicie.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Krok 5: Pełna Klasa Tokenizatora

Połącz wszystko razem: normalizuj, dziel na specjalnych tokenach, pre-tokenizuj, łącz BPE, mapuj na ID.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Krok 6: Test Wielojęzyczny

Prawdziwy test. Rzuć na to angielski, chiński, emoji i kod.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Chińskie znaki produkują po 3 bajty każdy. Emoji produkuje 4 bajty. Żadne z nich nie powoduje awarii tokenizatora. Żadne nie produkuje nieznanych tokenów. To jest moc byte-level BPE.

## Użyj Tego

### Porównanie Prawdziwych Tokenizatorów

Załaduj rzeczywiste tokenizatory z Llamy 3, GPT-4 i Mistrala. Zobacz, jak każdy radzi sobie z tym samym wielojęzycznym akapitem.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Zobaczysz różne liczby tokenów dla tego samego tekstu. Llama 3 ze słownikiem 128K jest bardziej agresywna w łączeniu popularnych wzorców. GPT-4 ze 100K jest pośrodku. Mistral z 32K produkuje więcej tokenów, ale ma mniejszą warstwę osadzeń.

Kompromis jest zawsze ten sam: większy słownik oznacza krótsze sekwencje, ale więcej parametrów.

## Dostarcz To

Ta lekcja produkuje prompt do budowania i debugowania produkcyjnych tokenizatorów. Zobacz `outputs/prompt-tokenizer-builder.md`.

## Ćwiczenia

1. **Łatwe:** Dodaj metodę `get_token_bytes(id)`, która pokazuje surowe bajty dla dowolnego ID tokena. Użyj jej, aby sprawdzić, co faktycznie reprezentują twoje najczęstsze połączone tokeny.
2. **Średnie:** Zaimplementuj pre-tokenizator w stylu Llamy, który dzieli na białych znakach i cyfrach, ale zachowuje spacje wiodące. Porównaj jego słownik z podejściem regex GPT-2 na tym samym korpusie.
3. **Trudne:** Dodaj metodę szablonu czatu, która przyjmuje listę wiadomości `{"role": ..., "content": ...}` i produkuje poprawną sekwencję tokenów dla formatu czatu Llamy 3. Przetestuj to względem implementacji HuggingFace.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|----------------------|
| Byte-level BPE | "Tokenizator działający na bajtach" | BPE z bazowym słownikiem 256 wartości bajtów -- obsługuje dowolne wejście bez nieznanych tokenów |
| Pre-tokenizacja | "Dzielenie przed BPE" | Dzielenie oparte na regex lub regułach, które zapobiega łączeniu przez BPE między granicami słów |
| Normalizacja NFKC | "Czyszczenie Unicode" | Dekompozycja kanoniczna, a następnie kompozycja kompatybilności -- ligatura "fi" staje się "fi", pełnowymiarowe "A" staje się "A" |
| Szablon czatu | "Jak wiadomości stają się tokenami" | Dokładny format konwersji listy wiadomości rola/treść na płaską sekwencję tokenów -- specyficzny dla modelu i musi pasować do formatu treningowego |
| Specjalne tokeny | "Tokeny kontrolne" | Zarezerwowane ID tokenów, które omijają BPE -- [BOS], [EOS], [PAD], znaczniki czatu -- dopasowywane dokładnie przed łączeniem |
| Płodność | "Tokeny na słowo" | Stosunek tokenów wyjściowych do słów wejściowych -- 1.3 dla angielskiego w GPT-4, 2-3 dla koreańskiego, wyższy oznacza zmarnowany kontekst |
| tiktoken | "Tokenizator OpenAI" | Implementacja BPE w Rust z wiązaniami Pythona -- 10-100x szybsza niż czysty Python |
| Tabela łączeń | "Słownik" | Uporządkowana lista łączeń par bajtów nauczonych podczas treningu -- TO JEST wiedza nauczona przez tokenizator |

## Dalsza Lektura

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- implementacja BPE w Rust używana przez GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- biblioteka tokenizatorów w Rust wspierająca BPE, WordPiece, Unigram
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- szczegóły dotyczące słownika 128K i trenowania tokenizatora
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- niezależna językowo tokenizacja
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- oryginalne mapowanie bajt-na-Unicode