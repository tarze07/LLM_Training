# Naturalne Wnioskowanie Językowe — Wynikanie Tekstowe

> "t wynika z h" oznacza, że człowiek czytający t wywnioskowałby, że h jest prawdziwe. NLI to zadanie przewidywania wynikanie / sprzeczność / neutralny. Nudne na powierzchni, krytyczne w produkcji.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## Problem

Zbudowałeś sumaryzator. Wyprodukował podsumowanie. Skąd wiesz, że podsumowanie nie zawiera halucynacji?

Zbudowałeś chatbota. Odpowiedział "tak." Skąd wiesz, że odpowiedź jest poparta przez znaleziony fragment?

Musisz sklasyfikować 10 000 artykułów prasowych według tematu. Nie masz etykiet treningowych. Czy możesz ponownie użyć modelu?

Wszystkie trzy problemy sprowadzają się do Naturalnego Wnioskowania Językowego. NLI pyta: mając przesłankę `t` i hipotezę `h`, czy `h` wynika z `t`, jest sprzeczne, czy neutralne (niepowiązane)?

- **Sprawdzanie halucynacji:** `t` = dokument źródłowy, `h` = twierdzenie w podsumowaniu. Brak wynikanie = halucynacja.
- **Ugruntowane QA:** `t` = znaleziony fragment, `h` = wygenerowana odpowiedź. Brak wynikanie = fabrykacja.
- **Klasyfikacja zero-shot:** `t` = dokument, `h` = zwerbalizowana etykieta ("To jest o sporcie"). Wynikanie = przewidziana etykieta.

Jedno zadanie, trzy zastosowania produkcyjne. Dlatego każdy framework ewaluacji RAG ma pod maską model NLI.

## Koncepcja

![NLI: klasyfikacja trójstronna, przesłanka vs hipoteza](../assets/nli.svg)

**Trzy etykiety.**

- **Wynikanie (Entailment).** `t` → `h`. "Kot jest na macie" wynika z "Jest kot."
- **Sprzeczność (Contradiction).** `t` → ¬`h`. "Kot jest na macie" jest sprzeczne z "Nie ma kota."
- **Neutralny (Neutral).** Brak wnioskowania w żadną stronę. "Kot jest na macie" jest neutralne względem "Kot jest głodny."

**Nie logiczne wynikanie.** NLI to *naturalne* wnioskowanie językowe — to, co typowy czytelnik by wywnioskował, a nie ścisła logika. "Jan wyprowadził psa" wynika z "Jan ma psa" w NLI, ale ścisła logika pierwszego rzędu przyjęłaby to tylko wtedy, gdybyś zaksjomatyzował posiadanie.

**Zbiory danych.**

- **SNLI** (2015). 570k ręcznie adnotowanych par, podpisy do obrazów jako przesłanki. Wąska domena.
- **MultiNLI** (2017). 433k par w 10 gatunkach. Standardowy korpus treningowy w 2026.
- **ANLI** (2019). Adwersarialne NLI. Ludzie napisali przykłady specjalnie zaprojektowane do łamania istniejących modeli. Trudniejsze.
- **DocNLI, ConTRoL** (2020–21). Przesłanki długości dokumentu. Testuje wnioskowanie wieloetapowe i dalekiego zasięgu.

**Architektura.** Enkoder transformerowy (BERT, RoBERTa, DeBERTa) czyta `[CLS] przesłanka [SEP] hipoteza [SEP]`. Reprezentacja `[CLS]` zasila 3-drożny softmax. Trenuj na MNLI, oceniaj na wstrzymanych benchmarkach, osiągaj 90%+ dokładności na parach z tej samej dystrybucji.

**Zero-shot przez NLI.** Mając dokument i kandydackie etykiety, zamień każdą etykietę w hipotezę ("Ten tekst jest o sporcie"). Oblicz prawdopodobieństwo wynikanie dla każdej. Wybierz maksimum. To mechanizm stojący za pipeline'm `zero-shot-classification` Hugging Face.

## Zbuduj To

### Krok 1: uruchom wytrenowany model NLI

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Dla produkcyjnego NLI, `facebook/bart-large-mnli` i `microsoft/deberta-v3-large-mnli` są otwartymi domyślnymi wyborami. DeBERTa-v3 jest na szczycie tablic liderów.

### Krok 2: klasyfikacja zero-shot

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

Domyślny szablon to "This example is about {label}." Dostosuj za pomocą `hypothesis_template`. Nie wymaga danych treningowych. Żadnego fine-tuningu. Działa od razu.

### Krok 3: sprawdzanie wierności dla RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

To jest rdzeń wierności RAGAS. Podziel wygenerowaną odpowiedź na atomowe twierdzenia. Sprawdź każde twierdzenie względem znalezionego kontekstu. Raportuj frakcję, która wynika.

### Krok 4: ręcznie napisany klasyfikator NLI (koncepcyjny)

Zobacz `code/main.py` dla zabawki korzystającej tylko z biblioteki standardowej: przesłanka i hipoteza są porównywane przez nakładanie leksykalne + wykrywanie negacji. Nie konkuruje z modelami transformerowymi — ale pokazuje kształt zadania: dwa teksty na wejściu, etykieta 3-drożna na wyjściu, strata = cross-entropia nad `{entail, contradict, neutral}`.

## Pułapki

- **Skróty tylko z hipotezy.** Modele mogą przewidzieć etykietę z samej hipotezy na ~60% w SNLI, ponieważ "not", "nobody", "never" korelują ze sprzecznością. Silna linia bazowa do wykrywania wycieku etykiet.
- **Heurystyka nakładania leksykalnego.** Heurystyka podciągu ("każdy podciąg wynika") przechodzi SNLI, ale zawodzi na HANS/ANLI. Używaj benchmarków adwersarialnych.
- **Degradacja na długości dokumentu.** Modele NLI na pojedynczych zdaniach tracą 20+ F1 na przesłankach długości dokumentu. Używaj modeli trenowanych na DocNLI dla długiego kontekstu.
- **Wrażliwość szablonu zero-shot.** "This example is about {label}" vs "{label}" vs "The topic is {label}" może zmienić dokładność o 10+ punktów. Dostosuj szablon.
- **Niedopasowanie domeny.** MNLI trenuje na ogólnym angielskim. Teksty prawne, medyczne i naukowe potrzebują modeli NLI specyficznych dla domeny (np. SciNLI, MedNLI).

## Użyj Tego

Stos w 2026:

| Przypadek użycia | Model |
|---------|-------|
| Ogólne NLI | `microsoft/deberta-v3-large-mnli` |
| Szybkie / brzegowe | `cross-encoder/nli-deberta-v3-base` |
| Klasyfikacja zero-shot (lekka) | `facebook/bart-large-mnli` |
| NLI na poziomie dokumentu | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Wielojęzyczne | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Wykrywanie halucynacji w RAG | Warstwa NLI w RAGAS / DeepEval |

Meta-wzór w 2026: NLI to taśma klejąca rozumienia tekstu. Gdykolwiek potrzebujesz "czy A wspiera B?" lub "czy A zaprzecza B?" — sięgnij po NLI, zanim sięgniesz po kolejne wywołanie LLM.

## Wdróż To

Zapisz jako `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Ćwiczenia

1. **Łatwe.** Uruchom `facebook/bart-large-mnli` na 20 ręcznie przygotowanych trójkach (przesłanka, hipoteza, etykieta) pokrywających wszystkie trzy klasy. Zmierz dokładność. Dodaj adwersarialne pułapki heurystyki podciągu ("I did not eat the cake" vs "I ate the cake") i sprawdź, czy się psuje.
2. **Średnie.** Porównaj szablon zero-shot `"This text is about {label}"` z `"The topic is {label}"` i `"{label}"` na 100 nagłówkach AG News. Raportuj zmianę dokładności.
3. **Trudne.** Zbuduj sprawdzacz wierności RAG: dekompozycja na atomowe twierdzenia + NLI na każde twierdzenie. Oceń na 50 odpowiedziach generowanych przez RAG ze złotym kontekstem. Zmierz wskaźniki fałszywie pozytywnych i fałszywie negatywnych względem ręcznych etykiet.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | 3-drożna klasyfikacja relacji przesłanka-hipoteza. |
| RTE | Recognizing Textual Entailment | Starsza nazwa NLI; to samo zadanie. |
| Entailment | "t implikuje h" | Typowy czytelnik uznałby h za prawdziwe, mając dane t. |
| Contradiction | "t wyklucza h" | Typowy czytelnik uznałby h za fałszywe, mając dane t. |
| Neutral | "nierozstrzygnięte" | Brak wnioskowania z t do h w żadną stronę. |
| Zero-shot classification | NLI jako klasyfikator | Werbalizuj etykiety jako hipotezy, wybierz max wynikanie. |
| Faithfulness | Czy odpowiedź jest poparta? | NLI nad (znaleziony kontekst, wygenerowana odpowiedź). |

## Dalsza Lektura

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — benchmark ANLI.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-jako-klasyfikator.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — koń roboczy NLI w 2026.