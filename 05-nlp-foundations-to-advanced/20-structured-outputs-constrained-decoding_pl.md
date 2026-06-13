# Ustrukturyzowane Wyniki & Ograniczone Dekodowanie

> Poproś LLM o JSON. Dostajesz JSON przez większość czasu. W produkcji "większość" jest problemem. Ograniczone dekodowanie zamienia "większość" w "zawsze" poprzez modyfikację logitów przed próbkowaniem.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## Problem

Klasyfikator podpowiada LLM-owi: "Zwróć jedną z {positive, negative, neutral}." Model zwraca "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...". Twój parser się wysypuje. F1 twojego klasyfikatora wynosi 0.0.

Swobodna generacja to nie umowa. To sugestia. System produkcyjny potrzebuje umowy.

W 2026 roku istnieją trzy warstwy.

1. **Podpowiadanie.** Grzecznie poproś. "Zwróć tylko obiekt JSON." Działa ~80% na czołowych modelach, mniej na mniejszych.
2. **Natywne API ustrukturyzowanych wyników.** OpenAI `response_format`, Anthropic tool use, Gemini JSON mode. Niezawodne na obsługiwanych schematach. Zależne od dostawcy.
3. **Ograniczone dekodowanie.** Modyfikuj logity na każdym kroku generacji, tak aby model *nie mógł* wyemitować nieprawidłowych tokenów. 100% poprawne z konstrukcji. Działa na każdym lokalnym modelu.

Ta lekcja buduje intuicję dla wszystkich trzech i wskazuje, kiedy sięgnąć po które.

## Koncepcja

![Ograniczone dekodowanie maskujące nieprawidłowe tokeny na każdym kroku](../assets/constrained-decoding.svg)

**Jak działa ograniczone dekodowanie.** Na każdym kroku generacji LLM produkuje wektor logitów nad pełnym słownikiem (~100k tokenów). *Procesor logitów* siedzi między modelem a próbnikiem. Oblicza, które tokeny są prawidłowe, biorąc pod uwagę aktualną pozycję w docelowej gramatyce — JSON Schema, regex, gramatyka bezkontekstowa — i ustawia logity wszystkich nieprawidłowych tokenów na minus nieskończoność. Softmax nad pozostałymi logitami umieszcza masę prawdopodobieństwa tylko na prawidłowych kontynuacjach.

Implementacje w 2026:

- **Outlines.** Kompiluje JSON Schema lub regex do automatu skończonego. Każdy token ma O(1) wyszukiwanie prawidłowego następnego tokenu. Oparty na FSM, więc rekurencyjne schematy wymagają spłaszczenia.
- **XGrammar / llguidance.** Silniki gramatyk bezkontekstowych. Obsługują rekurencyjne JSON Schema. Prawie zerowy narzut dekodowania. OpenAI przypisało zasługę llguidance w swojej implementacji ustrukturyzowanych wyników z 2025.
- **vLLM guided decoding.** Wbudowane `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` poprzez backendy Outlines, XGrammar lub lm-format-enforcer.
- **Instructor.** Nakładka oparta na Pydantic na dowolny LLM. Ponawia próby w przypadku błędu walidacji. Międzydostawcowy, ale nie modyfikuje logitów — polega na ponownych próbach + podpowiedziach świadomych ustrukturyzowanych wyników.

### Kontrintuicyjny wynik

Ograniczone dekodowanie jest często *szybsze* niż nieograniczona generacja. Dwa powody. Po pierwsze, zmniejsza przestrzeń poszukiwań następnego tokenu. Po drugie, sprytne implementacje pomijają generację tokenów całkowicie dla wymuszonych tokenów (rusztowanie takie jak `{"name": "` — każdy bajt jest zdeterminowany).

### Pułapka, która cię kosztuje

Kolejność pól ma znaczenie. Umieść `answer` przed `reasoning`, a model zobowiązuje się do odpowiedzi, zanim pomyśli. JSON jest poprawny. Odpowiedź jest zła. Żadna walidacja tego nie wychwyci.

```json
// ŹLE
{"answer": "yes", "reasoning": "because ..."}

// DOBRZE
{"reasoning": "... therefore ...", "answer": "yes"}
```

Kolejność pól w schemacie to logika, a nie formatowanie.

## Zbuduj To

### Krok 1: generacja ograniczona regexem od zera

Zobacz `code/main.py` dla samodzielnej implementacji FSM. Główna idea w 30 liniach:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM śledzi, które części gramatyki zostały już spełnione. `valid_tokens(state, tokenizer)` oblicza, które tokeny słownika mogą przesunąć FSM bez opuszczania ścieżki akceptującej.

### Krok 2: Outlines dla JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Zero błędów walidacji. Nigdy. FSM sprawia, że nieprawidłowe wyjście jest nieosiągalne.

### Krok 3: Instructor dla niezależnego od dostawcy Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Inny mechanizm. Instructor nie dotyka logitów. Formatuje schemat w podpowiedzi, parsuje wyjście i ponawia próbę w przypadku błędu walidacji (domyślnie 3 razy). Działa z każdym dostawcą. Ponowne próby dodają opóźnienia i koszty. Przenośność między dostawcami jest atutem.

### Krok 4: natywne API dostawców

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                   "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Ograniczone dekodowanie po stronie serwera. Niezawodność porównywalna z Outlines dla obsługiwanych schematów. Brak zarządzania lokalnym modelem. Przywiązuje cię do dostawcy.

## Pułapki

- **Rekurencyjne schematy.** Outlines spłaszcza rekurencję do ustalonej głębokości. Wyjścia struktury drzewiastej (zagnieżdżone komentarze, AST) potrzebują XGrammar lub llguidance (CFG).
- **Ogromne enumy.** Enum z 10 000 opcji kompiluje się wolno lub przekracza limit czasu. Przełącz się na wyszukiwarkę: najpierw przewidź top-k kandydatów, ogranicz do nich.
- **Zbyt ścisła gramatyka.** Wymuś regex `date: "YYYY-MM-DD"` i model nie może wypisać `"unknown"` dla brakujących dat. Model kompensuje, wymyślając datę. Zezwól na `null` lub wartownik.
- **Przedwczesne zobowiązanie.** Patrz wyżej na pułapkę z kolejnością pól. Zawsze umieszczaj reasoning jako pierwsze.
- **Tryb JSON dostawcy bez schematu.** Czysty tryb JSON gwarantuje tylko poprawny JSON, a nie poprawny *dla twojego przypadku użycia*. Zawsze podawaj pełny schemat.

## Użyj Tego

Stos w 2026:

| Sytuacja | Wybór |
|-----------|------|
| Model OpenAI/Anthropic/Google, prosty schemat | Natywne ustrukturyzowane wyniki dostawcy |
| Dowolny dostawca, workflow Pydantic, toleruje ponowne próby | Instructor |
| Lokalny model, potrzeba 100% poprawności, płaski schemat | Outlines (FSM) |
| Lokalny model, rekurencyjny schemat | XGrammar lub llguidance |
| Samodzielnie hostowany serwer inferencyjny | vLLM guided decoding |
| Przetwarzanie wsadowe z tolerancją ponownych prób | Instructor + najtańszy model |

## Wdróż To

Zapisz jako `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Ćwiczenia

1. **Łatwe.** Podpowiedz małemu modelowi z otwartymi wagami (np. Llama-3.2-3B) bez ograniczonego dekodowania dla `Review(sentiment, confidence, evidence_span)`. Zmierz frakcję, która parsuje się jako poprawny JSON na 100 recenzjach.
2. **Średnie.** Ten sam korpus z trybem JSON Outlines. Porównaj wskaźnik zgodności, opóźnienie i dokładność semantyczną.
3. **Trudne.** Zaimplementuj dekoder ograniczony regexem od zera dla numerów telefonów (`\d{3}-\d{3}-\d{4}`). Zweryfikuj 0 nieprawidłowych wyjść na 1000 próbek.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Constrained decoding | Wymuś poprawne wyjście | Maskuj logity nieprawidłowych tokenów na każdym kroku generacji. |
| Logit processor | To, co ogranicza | Funkcja: `(logits, state) -> masked_logits`. |
| FSM | Automat skończony | Skompilowana reprezentacja gramatyki; O(1) wyszukiwanie prawidłowego następnego tokenu. |
| CFG | Gramatyka bezkontekstowa | Gramatyka obsługująca rekurencję; wolniejsza, ale bardziej ekspresyjna niż FSM. |
| Schema field order | Czy to ma znaczenie? | Tak — pierwsze pole zobowiązuje; zawsze umieszczaj reasoning przed odpowiedzią. |
| Guided decoding | Nazwa vLLM dla tego | Ta sama koncepcja, zintegrowana z serwerem inferencyjnym. |
| JSON mode | Wczesna wersja OpenAI | Gwarantuje składnię JSON; NIE gwarantuje zgodności ze schematem. |

## Dalsza Lektura

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — artykuł o Outlines.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — szybkie ograniczone dekodowanie oparte na CFG.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — integracja z serwerem inferencyjnym.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API reference + pułapki.
- [Instructor library](https://python.useinstructor.com/) — Pydantic + ponowne próby u różnych dostawców.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — benchmarkowanie 6 frameworków ograniczonego dekodowania.