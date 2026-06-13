# Rozpoznawanie Nazwanych Encji

> Wyciągnij nazwy. Brzmi łatwo, dopóki nie zmierzysz się z niejednoznacznymi granicami, zagnieżdżonymi encjami i żargonem dziedzinowym.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## Problem

"Apple sued Google over its iPhone search deal in the US." Pięć encji: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (może), US (GPE). Dobry system NER wyodrębnia wszystkie z poprawnymi typami. Zły pomija iPhone, myli Apple-owoc z Apple-firmą i oznacza "US" jako PERSON.

NER jest koniem roboczym pod każdym potokiem ekstrakcji strukturalnej. Parsowanie CV, skanowanie logów pod kątem zgodności, anonimizacja dokumentacji medycznej, zrozumienie zapytań wyszukiwania, ugruntowywanie odpowiedzi chatbotów, ekstrakcja kontraktów prawnych. Nigdy tego nie widzisz; zawsze na tym polegasz.

Ta lekcja przechodzi klasyczną ścieżkę (regułowa, HMM, CRF) do nowoczesnej (BiLSTM-CRF, a następnie transformery). Każdy krok rozwiązuje konkretne ograniczenie poprzedniego. Wzorzec jest lekcją.

## Koncepcja

**Tagowanie BIO** (lub BILOU) zamienia ekstrakcję encji w problem sekwencyjnego etykietowania. Oznakuj każdy token jako `B-TYP` (początek encji), `I-TYP` (wewnątrz encji) lub `O` (poza jakąkolwiek encją).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Wielotokenowe encje łańcuchują: `New B-GPE`, `York I-GPE`, `City I-GPE`. Model, który rozumie BIO, może wyodrębniać dowolne zakresy.

Progresja architektury:

- **Regułowa.** Regex + przeszukiwanie gazetera. Wysoka precyzja na znanych encjach, zerowe pokrycie na nowych.
- **HMM.** Ukryty model Markowa. Prawdopodobieństwo emisji tokenu przy danej etykiecie, prawdopodobieństwo przejścia etykieta-etykieta. Dekodowanie Viterbiego. Trenowany na oznakowanych danych.
- **CRF.** Warunkowe pola losowe. Jak HMM, ale dyskryminacyjny, więc możesz mieszać dowolne cechy (kształt słowa, wielkość liter, sąsiednie słowa). Wciąż klasyczny koń roboczy w produkcji w 2026 dla wdrożeń o niskich zasobach.
- **BiLSTM-CRF.** Neuralne cechy zamiast ręcznie robionych. LSTM czyta zdanie w obu kierunkach, warstwa CRF na górze wymusza spójne sekwencje etykiet.
- **Oparty na transformerach.** Dostrój BERT z głową do klasyfikacji tokenów. Najlepsza dokładność. Najwięcej mocy obliczeniowej.

```figure
ner-bio-tagging
```

## Zbuduj To

### Krok 1: pomocnicy tagowania BIO

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Krok 2: ręcznie robione cechy

Dla klasycznego (nieneuronalnego) NER, cechy są grą. Przydatne:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` zwraca `xXxxxx`. `word_shape("USA-2024")` zwraca `XXX-dddd`. Wzorce wielkich liter są sygnałem wysokiej jakości dla nazw własnych.

### Krok 3: prosty baseline regułowy + słownikowy

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Produkcyjne gazetery mają miliony wpisów zeskrobanych z Wikipedii i DBpedii. Pokrycie jest dobre. Dezambiguacja (`Apple` firma kontra owoc) jest okropna. Dlatego modele statystyczne wygrały.

### Krok 4: krok CRF (szkic, nie pełna implementacja)

Pełny CRF od zera w 50 liniach nie jest pouczający bez podstaw teorii prawdopodobieństwa. Użyj `sklearn-crfsuite` zamiast:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` i `c2` to regularyzacja L1 i L2. `all_possible_transitions=True` pozwala modelowi nauczyć się, że nielegalne sekwencje (np. `I-ORG` po `O`) są mało prawdopodobne, co jest sposobem, w jaki CRF wymusza spójność BIO bez pisania ograniczenia.

### Krok 5: co dodaje BiLSTM-CRF

Cechy stają się wyuczone. Wejścia: embeddingi tokenów (GloVe lub fastText). LSTM czyta od lewej do prawej i od prawej do lewej. Połączone stany ukryte przechodzą przez warstwę wyjściową CRF. CRF wciąż wymusza spójność sekwencji etykiet; LSTM zastępuje ręcznie robione cechy wyuczonymi.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Do warstwy CRF użyj `torchcrf.CRF` (pip install pytorch-crf). Zysk nad ręcznie robionym CRF jest mierzalny, ale mniejszy niż oczekujesz, chyba że masz dziesiątki tysięcy oznakowanych zdań.

## Użyj Tego

spaCy dostarcza produkcyjne NER od razu po wyjęciu z pudełka.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Zauważ, że `iPhone` oznaczone jako `ORG` zamiast `PRODUCT` — mały model spaCy ma słabe pokrycie encji produktów. Duży model (`en_core_web_lg`) radzi sobie lepiej. Model transformerowy (`en_core_web_trf`) radzi sobie jeszcze lepiej.

Hugging Face dla NER opartego na BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` scala sąsiednie tokeny B-X, I-X w zakres. Bez tego otrzymujesz etykiety na poziomie tokenów i musisz scalać samodzielnie.

### NER oparty na LLM (opcja 2026)

Zero-shot i few-shot NER z LLM są teraz konkurencyjne z dostrojonymi modelami w wielu dziedzinach i dramatycznie lepsze, gdy oznakowanych danych jest mało.

- **Promptowanie zero-shot.** Daj LLM listę typów encji i przykładowy schemat. Poproś o wynik JSON. Działa od razu; dokładność jest umiarkowana na nowych domenach.
- **Promptowanie w stylu ZeroTuneBio.** Rozbij zadanie na ekstrakcja kandydatów → wyjaśnienie znaczenia → osąd → ponowne sprawdzenie. Prompt wieloetapowy (nie jednorazowy) znacząco podnosi dokładność w biomedycznym NER. Ten sam wzorzec działa dla domen prawnych, finansowych i naukowych.
- **Dynamiczne promptowanie z RAG.** Pobierz najbardziej podobne oznakowane przykłady z małego adnotowanego zbioru początkowego dla każdego wywołania inferencyjnego; zbuduj prompt few-shot na bieżąco. W benchmarkach z 2026, to podnosi F1 biomedycznego NER GPT-4 o 11-12% w porównaniu do statycznego promptowania.
- **Dekompozycja na typ encji.** Dla długich dokumentów pojedyncze wywołanie, które wyodrębnia wszystkie typy encji naraz, traci recall wraz ze wzrostem długości. Uruchom jeden przebieg ekstrakcji na typ encji. Wyższy koszt inferencji, znacznie wyższa dokładność. To standardowy wzorzec dla notatek klinicznych i kontraktów prawnych.

Rekomendacja produkcyjna na 2026: zacznij od baseline'u zero-shot z LLM, zanim zbierzesz dane treningowe. Często F1 jest wystarczająco dobre, że nigdy nie będziesz potrzebować dostrajania.

### Gdzie klasyczne NER wciąż wygrywa

Nawet przy dostępnych LLM, klasyczne NER wygrywa, gdy:

- Budżet opóźnienia jest poniżej 50ms.
- Masz tysiące oznakowanych przykładów i potrzebujesz F1 98%+.
- Domeny mają stabilną ontologię, do której wstępnie wytrenowany CRF lub BiLSTM dobrze się przenosi.
- Wymogi regulacyjne wymagają modelu lokalnego, niegeneratywnego.

### Gdzie się rozpada

- **Przesunięcie domeny.** NER trenowany na CoNLL na kontraktach prawnych działa gorzej niż gazeter. Dostrój na swojej domenie.
- **Zagnieżdżone encje.** "Bank of America Tower" jest jednocześnie ORG i FACILITY. Standardowe BIO nie może reprezentować nakładających się zakresów. Potrzebujesz zagnieżdżonego NER (wieloprzebiegowego lub opartego na zakresach).
- **Długie encje.** "United States Federal Deposit Insurance Corporation." Modele na poziomie tokenów czasami to dzielą. Użyj `aggregation_strategy` lub post-processingu.
- **Rzadkie typy.** Medyczne etykiety NER, takie jak DRUG_BRAND, ADVERSE_EVENT, DOSE. Modele ogólnego przeznaczenia nie mają pojęcia. Scispacy i BioBERT są punktami wyjścia.

## Dostarcz To

Zapisz jako `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj `bio_to_spans` (odwrotność `spans_to_bio`) i sprawdź spójność round-trip na 10 zdaniach.
2. **Średnie.** Wytrenuj CRF z sklearn-crfsuite powyżej na angielskim zbiorze danych NER CoNLL-2003. Raportuj F1 na encję używając `seqeval`. Typowy wynik: ~84 F1.
3. **Trudne.** Dostrój `distilbert-base-cased` na dziedzinowym zbiorze danych NER (medycznym, prawnym lub finansowym). Porównaj z małym modelem spaCy. Udokumentuj kontrole wycieku danych i opisz, co cię zaskoczyło.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|--------|-----------------|---------------------|
| NER | Wyodrębnianie nazw | Oznakuj zakresy tokenów typami (PERSON, ORG, GPE, DATE, ...). |
| BIO | Schemat tagowania | `B-X` zaczyna, `I-X` kontynuuje, `O` poza. |
| BILOU | Lepsze BIO | Dodaje `L-X` (ostatni), `U-X` (jednostka) dla czystszych granic. |
| CRF | Klasyfikator strukturalny | Modeluje przejścia między etykietami, nie tylko emisje. Wymusza prawidłowe sekwencje. |
| Zagnieżdżone NER | Nakładające się encje | Jeden zakres jest inną encją niż podzakres w nim. BIO nie może tego wyrazić. |
| F1 na poziomie encji | Właściwa metryka NER | Przewidziany zakres musi dokładnie pasować do prawdziwego zakresu. F1 na poziomie tokenów zawyża dokładność. |

## Dalsza Literatura

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — praca BiLSTM-CRF. Kanoniczna.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — wprowadza wzorzec klasyfikacji tokenów, który stał się standardem.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — praktyczna referencja dla każdego atrybutu na `Doc.ents` i `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — poprawna biblioteka metryk. Używaj jej zawsze.