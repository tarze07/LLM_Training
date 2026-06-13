# Znakowanie części mowy i analiza składniowa

> Gramatyka była niemodna przez jakiś czas. Potem każdy potok LLM potrzebował walidować ustrukturyzowaną ekstrakcję i wróciła.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## Problem

Lekcja 01 obiecywała, że lematyzacja wymaga znaku części mowy. Bez wiedzy, że `running` jest czasownikiem, lematyzator nie może zredukować go do `run`. Bez wiedzy, że `better` jest przymiotnikiem, nie może zredukować do `good`.

To obietnica ukrywała całe podpole. Znakowanie części mowy przypisuje kategorie gramatyczne. Analiza składniowa odtwarza strukturę drzewiastą zdania: które słowo modyfikuje które, który czasownik rządzi którymi argumentami. Klasyczne NLP spędziło dwadzieścia lat na udoskonalaniu obu. Potem głębokie uczenie skompresowało je do zadania klasyfikacji tokenów na bazie wstępnie wytrenowanego transformera, a społeczność badawcza ruszyła dalej.

Nie społeczność praktyczna. Każdy potok ekstrakcji ustrukturyzowanej nadal wykorzystuje pod spodem znakowanie części mowy i drzewa zależności. JSON generowany przez LLM jest walidowany względem ograniczeń gramatycznych. Systemy odpowiadania na pytania rozkładają zapytania za pomocą analizy zależności. Ewaluatory jakości tłumaczenia maszynowego sprawdzają dopasowanie drzew składniowych.

Warto wiedzieć. Ta lekcja wprowadza zestawy znaków, modele bazowe i punkt, w którym przestajesz implementować od zera i wywołujesz spaCy.

## Koncepcja

**Znakowanie POS** etykietuje każdy token kategorią gramatyczną. Zestaw znaków **Penn Treebank (PTB)** jest domyślnym dla angielskiego. 36 znaków z rozróżnieniami, które przypadkowy czytelnik uzna za drobiazgowe: `NN` rzeczownik pojedynczy, `NNS` rzeczownik mnogi, `NNP` rzeczownik własny pojedynczy, `VBD` czasownik przeszły, `VBZ` czasownik 3. osoba liczby pojedynczej teraźniejszy i tak dalej. Zestaw znaków **Universal Dependencies (UD)** jest bardziej ogólny (17 znaków) i niezależny od języka; stał się domyślnym dla pracy wielojęzycznej.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Analiza składniowa** produkuje drzewo. Dwa główne style:

- **Analiza frazowa.** Frazy rzeczownikowe, frazy czasownikowe, frazy przyimkowe zagnieżdżają się wewnątrz siebie. Wynikiem jest drzewo kategorii nieterminalnych (NP, VP, PP) z wyrazami jako liśćmi.
- **Analiza zależnościowa.** Każde słowo ma jedno słowo nadrzędne, od którego zależy, oznaczone relacją gramatyczną. Wynikiem jest drzewo, w którym każda krawędź to trójka (nadrzędne, zależne, relacja).

Analiza zależnościowa zwyciężyła w latach 2010., ponieważ dobrze uogólnia się na różne języki, zwłaszcza te o swobodnym szyku wyrazów.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Zbuduj to

### Krok 1: model bazowy najczęstszego znaku

Najprostszy znakownik POS, który działa. Dla każdego słowa przewiduj znak, który występował najczęściej w treningu.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Na korpusie Brown ten model bazowy osiąga ~85% dokładności. Niewiele, ale to podłoga, poniżej której żaden poważny model nie powinien spaść.

### Krok 2: bigramowy HMM

Modeluj łączne prawdopodobieństwo sekwencji:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Dwie tablice: prawdopodobieństwa przejścia (znak po znaku), prawdopodobieństwa emisji (słowo po znaku). Oszacuj obie z liczebności z wygładzaniem Laplace'a. Dekoduj z Viterbim (programowanie dynamiczne na siatce znaków).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigramowy HMM na Brown osiąga ~93% dokładności. Skok z 85% do 93% to głównie zasługa prawdopodobieństw przejścia — model uczy się, że `DET NOUN` jest częste, a `NOUN DET` rzadkie.

### Krok 3: dlaczego nowoczesne znakowniki biją ten wynik

Prawdopodobieństwa przejścia i emisji są lokalne. Nie potrafią uchwycić, że `saw` jest rzeczownikiem w "I bought a saw", ale czasownikiem w "I saw the movie." CRF z dowolnymi cechami (przyrostek, kształt słowa, słowo przed i po, samo słowo) osiąga ~97%. BiLSTM-CRF lub transformer osiąga ~98%+.

Sufit w tym zadaniu wyznaczony jest przez niezgodność annotatorów. Ludzcy annotatorzy zgadzają się w około 97% przypadków na Penn Treebank. Modele powyżej 98% prawdopodobnie przetrenowują zbiór testowy.

### Krok 4: szkic parsera zależnościowego

Pełna analiza zależnościowa od zera wykracza poza zakres; kanoniczne opracowanie podręcznikowe znajduje się w Jurafsky i Martin. Dwie klasyczne rodziny, które warto znać:

- **Parsery przejściowe** (arc-eager, arc-standard) działają jak parser przesuń-redukuj: czytają tokeny, przesuwają je na stos i stosują akcje redukujące, które tworzą łuki. Zachłanne dekodowanie jest szybkie. Klasyczna implementacja to MaltParser. Nowoczesna wersja neuronowa: parser przejściowy Chena i Manninga.
- **Parsery grafowe** (algorytm Eisnera, biaffine Dozat-Manning) punktują każdą możliwą krawędź nadrzędny-zależny i wybierają maksymalne drzewo rozpinające. Wolniejsze, ale dokładniejsze.

Do większości prac aplikacyjnych wywołaj spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Przeczytaj kolumnę `dep` od dołu do góry, a struktura gramatyczna zdania sama się ujawni.

## Użyj tego

Każda produkcyjna biblioteka NLP dostarcza znakowniki POS i parsery zależnościowe jako część standardowego potoku.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Szybkie, dokładne, zintegrowane z tokenizacją + NER + lematyzacją. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (relacja zależności).
- **Stanford NLP (stanza)**. Następca CoreNLP od Stanforda. Najnowocześniejszy dla 60+ języków.
- **trankit**. Oparty na transformerach, dobra dokładność UD.
- **NLTK**. `pos_tag`. Używalne, wolne, starsze. Dobre do nauczania.

### Gdzie to wciąż ma znaczenie w 2026

- **Lematyzacja.** Lekcja 01 potrzebuje POS, aby lematyzować poprawnie. Zawsze.
- **Ustrukturyzowana ekstrakcja z wyników LLM.** Walidacja, że wygenerowane zdanie respektuje ograniczenia gramatyczne (np. zgodność podmiotu z orzeczeniem, wymagane modyfikatory).
- **Sentyment aspektowy.** Analiza zależnościowa mówi, który przymiotnik modyfikuje który rzeczownik.
- **Rozumienie zapytań.** "movies directed by Wes Anderson starring Bill Murray" rozkłada się na ustrukturyzowane ograniczenia poprzez analizę składniową.
- **Transfer wielojęzyczny.** Znaki UD i relacje zależności są niezależne od języka, umożliwiając zerową analizę strukturalną nowych języków.
- **Potoki o niskiej mocy obliczeniowej.** Jeśli nie możesz dostarczyć transformera, POS + analiza zależnościowa + gazeter zaprowadzą cię zaskakująco daleko.

## Dostarcz to

Zapisz jako `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Ćwiczenia

1. **Łatwe.** Używając modelu bazowego najczęstszego znaku na małym oznakowanym korpusie (np. podzbiorze Browna z NLTK), zmierz dokładność na wstrzymanych zdaniach. Zweryfikuj wynik ~85%.
2. **Średnie.** Wytrenuj bigramowy HMM powyżej i podaj precyzję/czułość dla każdego znaku. Które znaki HMM myli najbardziej?
3. **Trudne.** Użyj parsera zależnościowego spaCy, aby wydobyć trójki podmiot-czasownik-dopełnienie z próbki 1000 zdań. Oceń na 50 ręcznie oznakowanych trójkach. Udokumentuj, gdzie ekstrakcja zawodzi (często strona bierna, współrzędne i podmioty domyślne).

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Znak POS | Typ słowa | Kategoria gramatyczna. PTB ma 36; UD ma 17. |
| Penn Treebank | Standardowy zestaw znaków | Specyficzny dla angielskiego. Drobnoziarniste czasy czasowników i liczba rzeczowników. |
| Universal Dependencies | Wielojęzyczny zestaw znaków | Bardziej ogólny niż PTB; neutralny językowo; domyślny dla pracy wielojęzycznej. |
| Analiza zależnościowa | Drzewo zdania | Każde słowo ma jedno nadrzędne, każda krawędź ma relację gramatyczną. |
| Viterbi | Programowanie dynamiczne | Znajduje sekwencję znaków o najwyższym prawdopodobieństwie przy danych emisjach i przejściach. |

## Dalsze czytanie

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — kanoniczne podręcznikowe opracowanie POS i parsingu.
- [Universal Dependencies project](https://universaldependencies.org/) — wielojęzyczny zestaw znaków i kolekcja korpusów drzewiastych używana przez każdy wielojęzyczny parser.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — praktyczne odniesienie dla każdego atrybutu udostępnianego na `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — artykuł, który wprowadził parsery neuronowe do głównego nurtu.