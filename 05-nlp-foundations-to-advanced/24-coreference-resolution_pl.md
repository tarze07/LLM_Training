# Rozpoznawanie Koreferencji

> "Ona zadzwoniła do niego. On nie odpowiedział. Doktor był na lunchu." Trzy odniesienia do dwóch osób i nikt nie jest wymieniony z imienia. Rozpoznawanie koreferencji ustala, kto jest kim.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## Problem

Wyodrębnij każdą wzmiankę o Apple Inc. z 300-słownego artykułu. Łatwe, gdy artykuł mówi "Apple." Trudne, gdy mówi "firma," "oni," "technologiczny gigant z Cupertino," lub "firma Jobsa." Bez rozwiązania tych wzmianek do tego samego bytu, twój pipeline NER traci 60-80% wzmianek.

Rozpoznawanie koreferencji łączy każde wyrażenie odnoszące się do tego samego bytu w świecie rzeczywistym w jeden klaster. To klej między powierzchownym NLP (NER, parsowanie) a downstream semantyką (IE, QA, sumaryzacja, KG).

Dlaczego to ma znaczenie w 2026:

- **Sumaryzacja:** "CEO ogłosił..." vs "Tim Cook ogłosił..." — podsumowanie powinno wymieniać CEO.
- **Odpowiadanie na pytania:** "Do kogo ona zadzwoniła?" wymaga rozwiązania "ona."
- **Ekstrakcja informacji:** graf wiedzy z "PER1 założył Apple" i "Jobs założył Apple" jako osobnymi wpisami jest błędny.
- **Wielodokumentowa IE:** scalanie wzmianek z różnych artykułów o tym samym zdarzeniu to koreferencja międzydokumentowa.

## Koncepcja

![Klasteryzacja koreferencji: wzmianki → byty](../assets/coref.svg)

**Zadanie.** Wejście: dokument. Wyjście: klasteryzacja wzmianek (zakresów), gdzie każdy klaster odnosi się do jednego bytu.

**Typy wzmianek.**

- **Nazwany byt.** "Tim Cook"
- **Nominalny.** "CEO", "firma"
- **Zaimkowy.** "on", "ona", "oni", "to"
- **Apozycyjny.** "Tim Cook, CEO Apple,"

**Architektury.**

1. **Oparta na regułach (Hobbs, 1978).** Rozwiązywanie zaimków oparte na drzewie składniowym z użyciem reguł gramatycznych. Dobra linia bazowa. Zaskakująco trudna do pobicia na zaimkach.
2. **Klasyfikator par wzmianek.** Dla każdej pary wzmianek (m_i, m_j), przewiduj, czy są koreferencyjne. Klasteryzuj przez domknięcie przechodnie. Standard przed 2016.
3. **Ranking wzmianek.** Dla każdej wzmianki, rankinguj kandydujące antecedenty (w tym "brak antecedenta"). Wybierz najlepszy.
4. **Zakresowy end-to-end (Lee et al., 2017).** Enkoder transformerowy. Wylicz wszystkie kandydujące zakresy do limitu długości. Przewiduj wyniki wzmianek. Przewiduj prawdopodobieństwo antecedenta dla każdego zakresu. Klasteryzuj zachłannie. Nowoczesny domyślny wybór.
5. **Generatywny (2024+).** Podpowiedz LLM: "Wypisz każdy zaimek w tym tekście i jego antecedent." Działa dobrze na łatwych przypadkach, ma problemy z długimi dokumentami i rzadkimi referentami.

**Metryki ewaluacji.** Pięć standardowych metryk (MUC, B³, CEAF, BLANC, LEA), ponieważ żadna pojedyncza metryka nie oddaje jakości klasteryzacji. Raportuj średnią pierwszych trzech jako CoNLL F1. Stan sztuki w 2026 na CoNLL-2012: ~83 F1.

**Znane trudne przypadki.**

- Opisy określone odnoszące się do bytów wprowadzonych strony wcześniej.
- Anafora pomostowa ("koła" → wcześniej wspomniany samochód).
- Anafora zerowa w językach takich jak chiński i japoński.
- Katafora (zaimek przed referentem): "Gdy **ona** weszła, Mary się uśmiechnęła."

## Zbuduj To

### Krok 1: wytrenowana neuronowa koreferencja (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Na dłuższym dokumencie otrzymasz coś takiego:
- Klaster 1: [Apple, The company, they]
- Klaster 2: [new products]

### Krok 2: rozwiązywacz zaimków oparty na regułach (dydaktyczny)

Zobacz `code/main.py` dla implementacji tylko z biblioteką standardową:

1. Wyodrębnij wzmianki: nazwane byty (zakresy z wielkiej litery), zaimki (wyszukiwanie w słowniku), opisy określone ("the X").
2. Dla każdego zaimka spójrz na poprzednie K wzmianek i oceń je według:
   - zgodności rodzaju/liczby (heurystyczna)
   - świeżości (bliższe wygrywa)
   - roli składniowej (podmioty preferowane)
3. Połącz z najwyżej ocenionym antecedentem.

Nie konkuruje z modelami neuronowymi. Ale pokazuje przestrzeń poszukiwań i decyzje, które musi podjąć model end-to-end.

### Krok 3: używanie LLM do koreferencji

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Dwa tryby awarii do obserwacji. Po pierwsze, LLM nadmiernie łączą ("him" i "her" odnoszące się do dwóch różnych osób). Po drugie, LLM po cichu pomijają wzmianki w długich dokumentach. Zawsze weryfikuj za pomocą kontroli przesunięcia zakresów.

### Krok 4: ewaluacja

Standardowy skrypt conll-2012 oblicza MUC, B³, CEAF-φ4 i raportuje średnią. Dla wewnętrznej ewaluacji zacznij od precyzji i czułości na poziomie zakresów na swoim adnotowanym zbiorze testowym, a następnie dodaj F1 łączenia wzmianek.

## Pułapki

- **Eksplozja singletonów.** Niektóre systemy raportują każdą wzmiankę jako własny klaster. B³ jest pobłażliwe. MUC to karze. Zawsze sprawdzaj wszystkie trzy metryki.
- **Zaimki w długim kontekście.** Wydajność spada ~15 F1 na dokumentach powyżej 2000 tokenów. Dziel ostrożnie.
- **Założenia dotyczące rodzaju.** Sztywne reguły rodzaju nie działają na referentach niebinarnych, organizacjach, zwierzętach. Używaj uczonych modeli lub neutralnego punktowania.
- **Dryf LLM na długich dokumentach.** Pojedyncze wywołanie API nie może niezawodnie klasteryzować wzmianek w 50+ akapitach. Używaj przesuwanego okna + scalania.

## Użyj Tego

Stos w 2026:

| Sytuacja | Wybór |
|-----------|------|
| Angielski, jeden dokument | `en_coreference_web_trf` (spaCy-experimental) lub AllenNLP neural coref |
| Wielojęzyczny | SpanBERT / XLM-R trenowany na OntoNotes lub Multilingual CoNLL |
| Międzydokumentowa koref. zdarzeń | Wyspecjalizowane modele end-to-end (SOTA 2025–26) |
| Szybka linia bazowa LLM | GPT-4o / Claude z podpowiedzią koreferencji ustrukturyzowanych wyników |
| Produkcyjne systemy dialogowe | Regułowe awaryjne + neuronowe główne + recenzja ręczna dla krytycznych slotów |

Wzór integracji dostarczany w 2026: uruchom NER najpierw, uruchom koreferencję, scal klastry koreferencji w byty NER. Zadania downstream widzą jeden byt na klaster, a nie jeden byt na wzmiankę.

## Wdróż To

Zapisz jako `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Ćwiczenia

1. **Łatwe.** Uruchom resolver oparty na regułach z `code/main.py` na 5 ręcznie przygotowanych akapitach. Zmierz dokładność łączenia wzmianek względem ground truth.
2. **Średnie.** Użyj wytrenowanego neuronowego modelu koreferencji na artykule prasowym. Porównaj klastry z własną ręczną adnotacją. Gdzie zawiódł?
3. **Trudne.** Zbuduj pipeline NER wzbogacony o koreferencję: najpierw NER, potem scal przez klastry koreferencji. Zmierz poprawę pokrycia bytów vs samo NER na 100 artykułach.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Mention | Odniesienie | Zakres tekstu odnoszący się do bytu (nazwa, zaimek, fraza rzeczownikowa). |
| Antecedent | Do czego "to" się odnosi | Wcześniejsza wzmianka, z którą późniejsza jest koreferencyjna. |
| Cluster | Wzmianki bytu | Zbiór wzmianek odnoszących się do tego samego bytu w świecie rzeczywistym. |
| Anaphora | Odniesienie wsteczne | Późniejsza wzmianka odnosi się do wcześniejszej ("on" → "Jan"). |
| Cataphora | Odniesienie naprzód | Wcześniejsza wzmianka odnosi się do późniejszej ("Gdy on przybył, Jan..."). |
| Bridging | Odniesienie domyślne | "Kupiłem samochód. Koła były złe." (koła TEGO samochodu.) |
| CoNLL F1 | Liczba na tablicach liderów | Średnia wyników F1 z MUC, B³, CEAF-φ4. |

## Dalsza Lektura

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — kanoniczny rozdział podręcznika.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — zakresowy end-to-end.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — pretrenowanie poprawiające koreferencję.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — benchmark.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — klasyk oparty na regułach.