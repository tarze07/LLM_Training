# Relation Extraction & Knowledge Graph Construction (Ekstrakcja relacji i budowa grafu wiedzy)

> NER znalazł encje. Łączenie encji je zakotwiczyło. Ekstrakcja relacji znajduje krawędzie między nimi. Graf wiedzy to suma węzłów, krawędzi i ich proweniencji.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## Problem

Analityk czyta: "Tim Cook został CEO Apple w 2011." Cztery fakty:

- `(Tim Cook, rola, CEO)`
- `(Tim Cook, pracodawca, Apple)`
- `(Tim Cook, data_rozpoczęcia, 2011)`
- `(Apple, typ, Organizacja)`

Ekstrakcja relacji (Relation Extraction, RE) zamienia wolny tekst na ustrukturyzowane trójki `(podmiot, relacja, obiekt)`. Agregacja w całym korpusie daje graf wiedzy. Agregacja i zapytanie dają substrat rozumowania dla RAG, analityki lub audytów zgodności.

Problem w 2026: LLM ekstrahują relacje entuzjastycznie. Zbyt entuzjastycznie. Halucynują trójki, których tekst źródłowy nie wspiera. Bez proweniencji nie odróżnisz prawdziwych trójek od prawdopodobnej fikcji. Odpowiedzią w 2026 są pipeline'y w stylu AEVS z kotwiczeniem i weryfikacją.

## Koncepcja

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Forma trójki.** `(encja_podmiotu, typ_relacji, encja_obiektu)`. Relacje pochodzą z zamkniętej ontologii (właściwości Wikidata, FIBO, UMLS) lub otwartego zbioru (OpenIE-style, wszystko dozwolone).

**Trzy podejścia do ekstrakcji.**

1. **Oparte na regułach / wzorcach.** Wzorce Hearsta: "X such as Y" → `(Y, isA, X)`. Plus ręcznie tworzone regex. Kruche, precyzyjne, wyjaśnialne.
2. **Nadzorowany klasyfikator.** Dla dwóch wystąpień encji w zdaniu, przewidź relację z ustalonego zbioru. Trenowany na TACRED, ACE, KBP. Standard 2015–2022.
3. **Generatywne LLM.** Promptuj model do emitowania trójek. Działa od razu. Potrzebuje proweniencji, w przeciwnym razie halucynuje pozornie wiarygodne śmieci.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Obecny framework do ograniczania halucynacji:

- **Anchor (Kotwica).** Zidentyfikuj każdy zakres encji i zakres frazy relacyjnej z dokładnymi pozycjami.
- **Extract (Ekstrakcja).** Generuj trójki powiązane z zakresami kotwic.
- **Verify (Weryfikacja).** Dopasuj każdy element trójki z powrotem do tekstu źródłowego; odrzuć wszystko, co nie jest wspierane.
- **Supplement (Uzupełnienie).** Przejście pokrycia zapewnia, że żaden zakotwiczony zakres nie został pominięty.

Halucynacje gwałtownie spadają. Wymaga więcej mocy obliczeniowej, ale jest audytowalne.

**Kompromis otwarta vs zamknięta ontologia.**

- **Zamknięta ontologia.** Stała lista właściwości (np. 11 000+ właściwości Wikidata). Przewidywalna. Zapytywalna. Trudna do wymyślenia.
- **Otwarte IE.** Każda fraza werbalna staje się relacją. Wysoki recall. Niska precyzja. Nieporządna w zapytaniach.

Produkcyjne KG zwykle mieszają: otwarte IE do odkrywania, a następnie kanonizacja relacji do zamkniętej ontologii przed scaleniem z głównym grafem.

## Zbuduj To

### Krok 1: ekstrakcja oparta na wzorcach

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Zobacz `code/main.py` dla pełnego zabawkowego ekstraktora. Wzorce Hearsta wciąż są używane w pipeline'ach domenowych, ponieważ są debugowalne.

### Krok 2: nadzorowana klasyfikacja relacji

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL to sekwencyjny ekstraktor relacji: tekst na wejściu, trójki na wyjściu, już w identyfikatorach właściwości Wikidata. Dostrajany na danych z distant-supervision. Standardowy baseline z otwartymi wagami.

### Krok 3: ekstrakcja promptowana przez LLM z kotwiczeniem

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Zweryfikuj każdy zwrócony zakres względem źródła. Odrzuć wszystko, gdzie `text[start:end] != triple_entity`. To jest krok "verify" AEVS w minimalnej formie.

### Krok 4: kanonizacja na zamkniętą ontologię

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Kanonizacja to często 60-80% pracy inżynieryjnej. Uwzględnij to w budżecie.

### Krok 5: zbuduj mały graf i zapytaj

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

To jest atom każdego systemu RAG-over-KG. Skaluj go za pomocą RDF triple stores (Blazegraph, Virtuoso), grafów właściwości (Neo4j) lub magazynów grafów wzbogaconych wektorowo.

## Pułapki

- **Coreferencja przed RE.** "He founded Apple" — RE musi wiedzieć, kim jest "he". Najpierw uruchom coref (lekcja 24).
- **Kanonizacja encji.** "Apple Inc" i "Apple" muszą rozwiązać się do tego samego węzła. Najpierw łączenie encji (lekcja 25).
- **Halucynowane trójki.** LLM emitują trójki, których tekst nie wspiera. Wymuś weryfikację zakresu.
- **Dryf kanonizacji relacji.** Relacje Open IE są niespójne ("was born in", "came from", "is a native of"). Sprowadź do kanonicznych identyfikatorów, w przeciwnym razie graf jest niezapytywalny.
- **Błędy czasowe.** "Tim Cook is CEO of Apple" — prawda teraz, fałsz w 2005. Wiele relacji jest ograniczonych czasowo. Użyj kwalifikatorów (`P580` czas rozpoczęcia, `P582` czas zakończenia w Wikidata).
- **Niedopasowanie domeny.** REBEL trenowany na Wikipedii. Teksty prawnicze, medyczne i naukowe często potrzebują modeli RE dostrojonych do domeny.

## Użyj Tego

Stos w 2026:

| Sytuacja | Wybór |
|-----------|------|
| Szybka produkcja, ogólna domena | REBEL lub LlamaPred z kanonizacją Wikidata |
| Domenowa (biomed, prawo) | SciREX-style fine-tune domeny + niestandardowa ontologia |
| Promptowane LLM, audytowane wyjście | Pipeline AEVS: anchor → extract → verify → supplement |
| IE z dużej liczby wiadomości | Hybryda wzorców + nadzorowana |
| Budowa KG od zera | Otwarte IE + ręczna kanonizacja |
| Temporalny KG | Ekstrakcja z kwalifikatorami (czas rozpoczęcia/zakończenia, punkt w czasie) |

Wzorzec integracji: NER → coref → łączenie encji → ekstrakcja relacji → mapowanie ontologii → załadowanie grafu. Każdy etap to potencjalna bramka jakości.

## Dostarcz To

Zapisz jako `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Ćwiczenia

1. **Łatwe.** Uruchom ekstraktor wzorców w `code/main.py` na 5 zdaniach z artykułów prasowych. Ręcznie sprawdź precyzję.
2. **Średnie.** Użyj REBEL (lub małego LLM) na tych samych zdaniach. Porównaj trójki. Który ekstraktor ma wyższą precyzję? Wyższy recall?
3. **Trudne.** Zbuduj pipeline AEVS: ekstrakcja z LLM + weryfikacja zakresów względem źródła. Zmierz wskaźnik halucynacji przed i po kroku weryfikacji na 50 zdaniach w stylu Wikipedii.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|-----------------|-----------------------|
| Trójka | Podmiot-relacja-obiekt | Krotka `(s, r, o)` będąca jednostką atomową KG. |
| Otwarte IE | Ekstrahuj wszystko | Frazy relacyjne o otwartym słownictwie; wysoki recall, niska precyzja. |
| Zamknięta ontologia | Stały schemat | Ograniczony zbiór typów relacji (Wikidata, UMLS, FIBO). |
| Kanonizacja | Normalizuj wszystko | Mapowanie nazw powierzchniowych / relacji do kanonicznych identyfikatorów. |
| AEVS | Ugrundowana ekstrakcja | Pipeline Anchor-Extraction-Verification-Supplement (2026). |
| Proweniencja | Link do źródła prawdy | Każda trójka niesie ID dokumentu + zakres znakowy do swojego źródła. |
| Distant supervision | Tanie etykiety | Dopasowanie tekstu do istniejącego KG w celu tworzenia danych treningowych. |

## Dalsza Literatura

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — the distant-supervision paper.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE workhorse.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — joint IE.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination-mitigation design.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — canonical graph queries.