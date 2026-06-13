# Przetwarzanie Tekstu — Tokenizacja, Stemming, Lematyzacja

> Język jest ciągły. Modele są dyskretne. Preprocessing jest mostem.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## Problem

Model nie może przeczytać "The cats were running." Czyta liczby całkowite.

Każdy system NLP zaczyna od tych samych trzech pytań. Gdzie zaczyna się słowo. Jaki jest rdzeń słowa. Jak traktować "run", "running", "ran" jako to samo, gdy to pomaga, a jako różne rzeczy, gdy nie.

Źle wykonana tokenizacja sprawia, że model uczy się na śmieciach. Jeśli tokenizer traktuje `don't` jako jeden token, ale `do n't` jako dwa, rozkład danych treningowych się rozszczepia. Jeśli stemmer redukuje `organization` i `organ` do tego samego rdzenia, modelowanie tematów umiera. Jeśli lematyzator potrzebuje kontekstu części mowy, ale go nie przekazujesz, czasowniki są traktowane jak rzeczowniki.

Ta lekcja buduje trzy kroki preprocessingowe od zera, a następnie pokazuje, jak NLTK i spaCy wykonują tę samą pracę, abyś mógł zobaczyć kompromisy.

## Koncepcja

Trzy operacje. Każda ma swoje zadanie i sposób na porażkę.

**Tokenizacja** dzieli ciąg znaków na tokeny. "Token" jest celowo nieostry, ponieważ odpowiednia granularność zależy od zadania. Poziom słowa dla klasycznego NLP. Podsłowo dla transformerów. Znak dla języków bez odstępów.

**Stemming** odcina przyrostki za pomocą reguł. Szybki, agresywny, głupi. `running -> run`. `organization -> organ`. Ten drugi to sposób na porażkę.

**Lematyzacja** redukuje słowo do jego formy słownikowej przy użyciu wiedzy gramatycznej. Wolniejsza, dokładna, potrzebuje tablicy odnośników lub analizatora morfologicznego. `ran -> run` (musi wiedzieć, że "ran" jest formą przeszłą "run"). `better -> good` (musi znać formy stopniowania).

Zasada kciuka. Używaj stemmingu, gdy liczy się szybkość i możesz tolerować szum (indeksowanie wyszukiwarki, przybliżona klasyfikacja). Używaj lematyzacji, gdy znaczenie ma znaczenie (odpowiadanie na pytania, wyszukiwanie semantyczne, wszystko, co użytkownik przeczyta).

```figure
edit-distance
```

## Zbuduj To

### Krok 1: regexowy tokenizer słów

Najprostszy użyteczny tokenizer dzieli na znakach niealfanumerycznych, zachowując interpunkcję jako osobne tokeny. Nieidealny, nieostateczny, ale działa w jednej linii.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Trzy wzorce w kolejności pierwszeństwa. Słowa z opcjonalnym wewnętrznym apostrofem (`don't`, `it's`). Czyste liczby. Dowolny pojedynczy znak niebędący białą spacją ani alfanumerycznym jako samodzielny token (interpunkcja).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Sposoby na porażkę warte odnotowania. `3pm` dzieli się na `['3', 'pm']`, ponieważ przełączaliśmy się między ciągami liter i cyfr. Wystarczająco dobre dla większości zadań. URL-e, e-maile, hashtagi wszystkie się psują. W produkcji dodaj wzorce przed ogólnymi.

### Krok 2: stemmer Portera (tylko krok 1a)

Pełny algorytm Portera ma pięć faz reguł. Sam krok 1a obejmuje najczęstsze angielskie przyrostki i uczy wzorca.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Czytaj reguły od góry do dołu. Reguła `ies -> i` jest powodem, dla którego `ponies -> poni`, a nie `pony`. Prawdziwy Porter ma krok 1b, który by to naprawił. Reguły konkurują. Wcześniejsze reguły wygrywają. Kolejność ma większe znaczenie niż jakakolwiek pojedyncza reguła.

### Krok 3: lematyzator oparty na tablicy odnośników

Prawdziwa lematyzacja potrzebuje morfologii. Wersja do nauczania używa małej tablicy lematów i mechanizmu awaryjnego.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

Ostatni przypadek to kluczowy moment edukacyjny. `watched` nie ma w naszej tablicy, a nasz mechanizm awaryjny obsługuje tylko `ing`. Prawdziwa lematyzacja obejmuje `ed`, czasowniki nieregularne, przymiotniki w stopniu wyższym, liczby mnogie ze zmianami brzmienia (`children -> child`). Dlatego systemy produkcyjne używają WordNet, morfologizatora spaCy lub pełnego analizatora morfologicznego.

### Krok 4: połączenie w potok

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

Brakującym elementem jest tagger części mowy. Faza 5 · 07 (Tagowanie Części Mowy) buduje go. Na razie ustaw wszystko na `NOUN` i przyznaj się do ograniczenia.

## Użyj Tego

NLTK i spaCy dostarczają wersje produkcyjne. Po kilka linii każda.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` obsługuje skróty, Unicode, przypadki brzegowe, których twój regex nie łapie. `PorterStemmer` wykonuje wszystkie pięć faz. `WordNetLemmatizer` potrzebuje tłumaczenia tagu POS ze schematu Penn Treebank NLTK na zestaw skrótów WordNet. Powyższe połączenie tłumaczące to fragment, który większość poradników pomija.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy ukrywa cały potok za `nlp(text)`. Tokenizacja, tagowanie POS i lematyzacja wszystkie działają. Szybszy niż NLTK na większych danych. Dokładniejszy od razu po wyjęciu z pudełka. Kompromis polega na tym, że nie można łatwo wymieniać poszczególnych komponentów.

### Kiedy który wybrać

| Sytuacja | Wybierz |
|-----------|--------|
| Nauczanie, badania, wymiana komponentów | NLTK |
| Produkcja, wielojęzyczność, liczy się szybkość | spaCy |
| Potok transformerowy (i tak będziesz tokenizować tokenizerem modelu) | Użyj `tokenizers` / `transformers` i pomiń klasyczny preprocessing |

### Dwa sposoby na porażkę, o których nikt nie ostrzega

Większość poradników uczy algorytmów i kończy. Dwie rzeczy ugryzą prawdziwy potok preprocessingowy i prawie nigdy nie są omawiane.

**Dryf odtwarzalności.** NLTK i spaCy zmieniają zachowanie tokenizacji i lematyzatora między wersjami. To, co produkowało `['do', "n't"]` w spaCy 2.x, może produkować `["don't"]` w 3.x. Twój model był trenowany na jednym rozkładzie. Inferencja działa teraz na innym. Dokładność cicho spada i nikt nie wie dlaczego. Przypnij wersje bibliotek w `requirements.txt`. Napisz test regresyjny preprocessingowy, który zamraża oczekiwaną tokenizację 20 przykładowych zdań. Uruchom go przy każdej aktualizacji.

**Niedopasowanie treningu i inferencji.** Trenuj z agresywnym preprocessingiem (małe litery, usuwanie stopwords, stemming), wdróż na surowych danych wejściowych użytkownika i patrz, jak wydajność spada. To najczęstsza produkcyjna porażka NLP. Jeśli preprocessujesz podczas treningu, musisz uruchomić identyczną funkcję podczas inferencji. Dostarczaj preprocessing jako funkcję wewnątrz pakietu modelu, a nie jako komórkę notebooka, którą zespół serwujący przepisuje.

## Dostarcz To

Wielokrotnego użytku prompt, który pomaga inżynierom wybrać strategię preprocessingową bez czytania trzech podręczników.

Zapisz jako `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Ćwiczenia

1. **Łatwe.** Rozszerz `tokenize`, aby zachować URL-e jako pojedyncze tokeny. Test: `tokenize("Visit https://example.com today.")` powinien wyprodukować jeden token URL.
2. **Średnie.** Zaimplementuj krok 1b Portera. Jeśli słowo zawiera samogłoskę i kończy się na `ed` lub `ing`, usuń je. Obsłuż zasadę podwójnej spółgłoski (`hopping -> hop`, nie `hopp`).
3. **Trudne.** Zbuduj lematyzator, który używa WordNet jako tablicy odnośników, ale wraca do stemmera Portera, gdy WordNet nie ma wpisu. Zmierz dokładność na oznakowanym korpusie w porównaniu z czystym WordNet i czystym Porterem.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|--------|-----------------|---------------------|
| Token | Słowo | Dowolna jednostka, którą model konsumuje. Może być słowem, podsłowem, znakiem lub bajtem. |
| Stem | Rdzeń słowa | Wynik regułowego usuwania przyrostków. Nie zawsze prawdziwe słowo. |
| Lemma | Forma słownikowa | Forma, której byś szukał. Wymaga kontekstu gramatycznego do poprawnego obliczenia. |
| Tag POS | Część mowy | Kategoria taka jak NOUN, VERB, ADJ. Potrzebna do dokładnej lematyzacji. |
| Morfologia | Reguły kształtu słowa | Jak słowo zmienia formę w zależności od czasu, liczby, przypadku. Lematyzacja na niej polega. |

## Dalsza Literatura

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — oryginalna praca, pięć stron, wciąż najjaśniejsze wyjaśnienie.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — jak okablowany jest prawdziwy potok.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — przypadki brzegowe tokenizacji, o których jeszcze nie pomyślałeś.