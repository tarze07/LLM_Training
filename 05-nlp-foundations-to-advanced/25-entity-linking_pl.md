# Entity Linking & Disambiguation (Łączenie i dezambiguacja encji)

> NER znalazło "Paris." Łączenie encji decyduje: Paryż, Francja? Paris Hilton? Paris, Teksas? Paris (trojański książę)? Bez łączenia twój graf wiedzy pozostaje niejednoznaczny.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## Problem

Zdanie brzmi: "Jordan beat the press." Twój NER oznacza "Jordan" jako PERSON. Dobrze. Ale *który* Jordan?

- Michael Jordan (koszykówka)?
- Michael B. Jordan (aktor)?
- Michael I. Jordan (profesor ML z Berkeley — tak, to zamieszanie jest realne w pracach ML)?
- Jordania (państwo)?
- Jordan (hebrajskie imię)?

Łączenie encji (Entity Linking, EL) rozwiązuje każde wystąpienie do unikalnego wpisu w bazie wiedzy: Wikidata, Wikipedia, DBpedia lub domenowej KB. Dwa podzadania:

1. **Generowanie kandydatów.** Dla "Jordan", które wpisy w KB są możliwe?
2. **Dezambiguacja.** Biorąc pod uwagę kontekst, który kandydat jest właściwy?

Oba etapy są uczalne. Oba są benchmarkowane. Połączony pipeline jest stabilny od dekady — zmienia się jakość dezambiguatora.

## Koncepcja

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Generowanie kandydatów.** Dla danej formy powierzchniowej ("Jordan"), przeszukaj indeks aliasów w poszukiwaniu kandydatów. Słowniki aliasów Wikipedii obejmują większość nazwanych encji: "JFK" → John F. Kennedy, Jacqueline Kennedy, lotnisko JFK, film JFK. Typowy indeks zwraca 10-30 kandydatów na wystąpienie.

**Dezambiguacja: trzy podejścia.**

1. **Priorytet + kontekst (Milne & Witten, 2008).** `P(encja | wystąpienie) × podobieństwo-kontekstu(encja, tekst)`. Działa dobrze, szybko, bez uczenia.
2. **Oparte na embeddingach (ESS / REL / BLINK).** Zakoduj wystąpienie + kontekst. Zakoduj opis każdego kandydata. Wybierz max cosine. Standard 2020-2024.
3. **Generatywne (GENRE, 2021; oparte na LLM, 2023+).** Dekoduj kanoniczną nazwę encji token po tokenie. Ograniczone do trie poprawnych nazw encji, więc wynik jest gwarantowanym poprawnym ID w KB.

**End-to-end vs pipeline.** Nowoczesne modele (ELQ, BLINK, ExtEnD, GENRE) wykonują NER + generowanie kandydatów + dezambiguację w jednym przejściu. Systemy pipeline wciąż dominują w produkcji, ponieważ można wymieniać komponenty.

### Dwa pomiary

- **Recall wystąpień (generowanie kandydatów).** Frakcja złotych wystąpień, dla których poprawny wpis w KB znajduje się na liście kandydatów. Podstawa dla całego pipeline.
- **Dokładność dezambiguacji / F1.** Dla poprawnych kandydatów, jak często top-1 jest trafny.

Zawsze raportuj oba. System z 99% dezambiguacją przy 80% recallu kandydatów to pipeline o skuteczności 80%.

## Zbuduj To

### Krok 1: zbuduj indeks aliasów z przekierowań Wikipedii

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Dane aliasów Wikipedii: ~18M par (alias, encja). Pobierz z zrzutów Wikidata. Przechowuj jako odwrócony indeks.

### Krok 2: dezambiguacja oparta na kontekście

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard overlap to zabawka. Zastąp cosinusowym podobieństwem na embeddingach (patrz `code/main.py` krok-2 dla wersji transformerowej).

### Krok 3: oparte na embeddingach (w stylu BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

W czasie indeksowania, osadź każdą encję w KB raz. W czasie zapytania, osadź wystąpienie + kontekst raz, wykonaj iloczyn skalarny z pulą kandydatów, wybierz max.

### Krok 4: generatywne łączenie encji (koncepcja)

GENRE dekoduje tytuł Wikipedii encji znak po znaku. Ograniczone dekodowanie (patrz lekcja 20) zapewnia, że tylko poprawne tytuły mogą być wyjściem. Ścisła integracja z trie opartym na KB. Współczesnym następcą jest REL-G i EL promptowane przez LLM z ustrukturyzowanym wyjściem.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

W połączeniu z białą listą (Outlines `choice`), jest to najprostszy pipeline EL do wdrożenia w 2026.

### Krok 5: ewaluacja na AIDA-CoNLL

AIDA-CoNLL to standardowy benchmark EL: 1393 artykułów Reutersa, 34k wystąpień, encje z Wikipedii. Raportuj dokładność w-KB (`P@1`) i wskaźnik detekcji NIL (poza-KB).

## Pułapki

- **Obsługa NIL.** Niektóre wystąpienia nie są w KB (nowe encje, mało znane osoby). Systemy muszą przewidywać NIL zamiast zgadywać błędną encję. Mierzone osobno.
- **Błędy granic wystąpień.** Nadrzędny NER pomija fragmenty zakresów ("Bank of America" oznaczone tylko jako "Bank"). Recall EL spada.
- **Bias popularności.** Uczone systemy nadmiernie przewidują częste encje. Wzmianka o "Michael I. Jordan" w pracy o ML często łączy się z Jordanem koszykarzem.
- **Wielojęzyczne EL.** Mapowanie wystąpień w tekście chińskim na angielskie encje Wikipedii. Wymaga wielojęzycznego enkodera lub kroku tłumaczenia.
- **Nieaktualność KB.** Nowe firmy, wydarzenia, osoby nie znajdują się w zeszłorocznym zrzucie Wikipedii. Pipeline produkcyjne potrzebują pętli odświeżania.

## Użyj Tego

Stos w 2026:

| Sytuacja | Wybór |
|-----------|------|
| Ogólny angielski + Wikipedia | BLINK lub REL |
| Wielojęzyczne, KB = Wikipedia | mGENRE |
| Przyjazne LLM, kilka wystąpień/dzień | Promptuj Claude/GPT-4 z listą kandydatów + ograniczone JSON |
| Domenowa KB (medyczna, prawna) | Niestandardowy BERT z wyszukiwaniem w KB + fine-tune na domenowym zbiorze w stylu AIDA |
| Ekstremalnie niskie opóźnienie | Tylko priorytet dokładnego dopasowania (baseline Milne-Witten) |
| Badawczy SOTA | GENRE / ExtEnD / generatywne LLM-EL |

Wzorzec produkcyjny w 2026: NER → coref → EL na każdym wystąpieniu → scalenie klastrów do jednej kanonicznej encji na klaster. Wynik: jedno ID KB na encję w dokumencie, nie jedno na wystąpienie.

## Dostarcz To

Zapisz jako `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj dezambiguator priorytet+kontekst w `code/main.py` na 10 niejednoznacznych wystąpieniach (Paris, Jordan, Apple). Ręcznie oznacz poprawną encję. Zmierz dokładność.
2. **Średnie.** Zakoduj 50 niejednoznacznych wystąpień za pomocą sentence transformer. Osadź opis każdego kandydata. Porównaj dezambiguację opartą na embeddingach z Jaccard overlap.
3. **Trudne.** Zbuduj domenową KB z 1k encji (np. pracownicy + produkty w twojej firmie). Zaimplementuj NER + EL end-to-end. Zmierz precyzję i recall na 100 wstrzymanych zdaniach.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|-----------------|-----------------------|
| Łączenie encji (EL) | Link do Wikipedii | Mapowanie wystąpienia do unikalnego wpisu w KB. |
| Generowanie kandydatów | Kto to może być? | Zwraca krótką listę prawdopodobnych wpisów w KB dla wystąpienia. |
| Dezambiguacja | Wybierz właściwy | Oceń kandydatów używając kontekstu, wybierz zwycięzcę. |
| Indeks aliasów | Tabela przeglądowa | Mapowanie z formy powierzchniowej → encje kandydujące. |
| NIL | Nie w KB | Wyraźna predykcja, że żaden wpis w KB nie pasuje. |
| KB | Baza wiedzy | Wikidata, Wikipedia, DBpedia lub domenowa KB. |
| AIDA-CoNLL | Benchmark | 1393 artykułów Reutersa ze złotymi łączami encji. |

## Dalsza Literatura

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — foundational prior+context approach.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — the embedding-based workhorse.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — generative EL with constrained decoding.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — the benchmark paper.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — the open production stack.