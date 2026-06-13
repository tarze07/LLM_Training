# Dialogue State Tracking (Śledzenie stanu dialogu)

> "Chcę tanią restaurację na północy... właściwie zrób to średnią... i dodaj włoską." Trzy tury, trzy aktualizacje stanu. DST utrzymuje słownik slot-wartość w synchronizacji, aby rezerwacja działała.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## Problem

W systemie dialogowym zorientowanym zadaniowo cel użytkownika jest kodowany jako zbiór par slot-wartość: `{kuchnia: włoska, obszar: północ, cena: średnia}`. Każda tura użytkownika może dodać, zmienić lub usunąć slot. System musi przeczytać całą rozmowę i poprawnie zwrócić bieżący stan.

Pomyłka w jednym slocie i system rezerwuje złą restaurację, planuje zły lot lub obciąża złą kartę. DST jest zawiasem między tym, co użytkownik powiedział, a tym, co wykonuje backend.

Dlaczego wciąż ma znaczenie w 2026 pomimo LLM:

- Domeny wrażliwe na zgodność (bankowość, opieka zdrowotna, rezerwacja lotów) wymagają deterministycznych wartości slotów, a nie generowania dowolnego tekstu.
- Agenci używający narzędzi wciąż potrzebują rozstrzygnięcia slotów przed wywołaniem API.
- Wieloturowa korekta jest trudniejsza niż wygląda: "właściwie nie, zrób to na czwartek."

Nowoczesny pipeline: klasyczne koncepcje DST + ekstraktory LLM + zabezpieczenia ustrukturyzowanego wyjścia.

## Koncepcja

![DST: dialog history → slot-value state](../assets/dst.svg)

**Struktura zadania.** Schemat definiuje domeny (restauracja, hotel, taksówka) i ich sloty (kuchnia, obszar, cena, osoby). Każdy slot może być pusty, wypełniony wartością z zamkniętego zbioru (cena: {tania, średnia, droga}) lub wartością dowolną (nazwa: "The Copper Kettle").

**Dwie formulacje DST.**

- **Klasyfikacja.** Dla każdej pary (slot, wartość_kandydacka) przewidź tak/nie. Działa dla slotów z zamkniętym słownictwem. Standard przed 2020.
- **Generacja.** Mając dialog, wygeneruj wartości slotów jako dowolny tekst. Działa dla slotów z otwartym słownictwem. Nowoczesny standard.

**Metryka.** Joint Goal Accuracy (JGA) — frakcja tur, w których *każdy* slot jest poprawny. Wszystko albo nic. Liderbord MultiWOZ 2.4 osiąga około 83% w 2026.

**Architektury.**

1. **Oparte na regułach (regex slotów + słowa kluczowe).** Silny baseline dla wąskich domen. Debugowalne.
2. **TripPy / BERT-DST.** Generacja oparta na kopiowaniu z kodowaniem BERT. Standard przed LLM.
3. **LDST (LLaMA + LoRA).** LLM dostrojony instrukcjami z promptowaniem domena-slot. Osiąga jakość na poziomie ChatGPT na MultiWOZ 2.4.
4. **Bez ontologii (2024–26).** Pomiń schemat; generuj nazwy slotów i wartości bezpośrednio. Obsługuje otwarte domeny.
5. **Prompt + ustrukturyzowane wyjście (2024–26).** LLM ze schematem Pydantic + ograniczone dekodowanie. 5 linii kodu, gotowe do produkcji.

### Klasyczne tryby awarii

- **Koreferencja między turami.** "Zostańmy przy pierwszej opcji." Wymaga rozstrzygnięcia, która opcja.
- **Nadpisanie vs dodanie.** Użytkownik mówi "dodaj włoską." Czy zastępujesz kuchnię, czy dodajesz?
- **Niejawne potwierdzenia.** "OK, spoko" — czy to zaakceptowało oferowaną rezerwację?
- **Korekta.** "Właściwie zrób to na 19:00." Musi zaktualizować czas bez czyszczenia innych slotów.
- **Koreferencja do poprzedniej wypowiedzi systemu.** "Tak, to." Które "to"?

## Zbuduj To

### Krok 1: ekstraktor slotów oparty na regułach

Zobacz `code/main.py`. Regex + słowniki synonimów pokrywają 70% kanonicznych wypowiedzi w wąskich domenach:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Kruche poza kanonicznym słownictwem. Działa do deterministycznych potwierdzeń slotów.

### Krok 2: pętla aktualizacji stanu

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Trzy niezmienniki:

- Nigdy nie resetuj slotu, którego użytkownik nie dotknął.
- Wyraźna negacja ("nieważne z kuchnią") musi czyścić.
- Korekta użytkownika ("właściwie...") musi nadpisywać, nie dodawać.

### Krok 3: DST napędzane LLM z ustrukturyzowanym wyjściem

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic gwarantuje poprawny obiekt stanu. Bez regex, bez niezgodności schematów, bez halucynowanych slotów.

### Krok 4: ewaluacja JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Kalibracja: jaka frakcja tur ma WSZYSTKIE sloty poprawne? Dla MultiWOZ 2.4, najlepsze systemy 2026: 80-83%. Twój system wewnątrzdomenowy powinien to przekraczać na swoim wąskim słownictwie, w przeciwnym razie baseline LLM cię pokonuje.

### Krok 5: obsługa korekty

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Po wykrytej korekcie nadpisz ostatnio zaktualizowany slot, zamiast dodawać. Trudne do zrobienia poprawnie bez pomocy LLM. Nowoczesny wzorzec: zawsze pozwól LLM wygenerować cały stan z historii, zamiast aktualizować inkrementalnie — to naturalnie obsługuje korekty.

## Pułapki

- **Koszt regeneracji pełnej historii.** Pozwalanie LLM na regenerację stanu w każdej turze kosztuje O(n²) całkowitych tokenów. Ogranicz historię lub podsumuj starsze tury.
- **Dryf schematu.** Dodawanie nowych slotów po fakcie psuje stare dane treningowe. Wersjonuj swój schemat.
- **Wrażliwość na wielkość liter.** "Italian" vs "italian" vs "ITALIAN" — normalizuj wszędzie.
- **Niejednoznaczne dziedziczenie.** Jeśli użytkownik wcześniej określił "dla 4 osób," nowe żądanie innego czasu nie powinno czyścić osób. Zawsze przekazuj pełną historię.
- **Dowolne vs zamknięte zbiory.** Nazwy, czasy i adresy potrzebują slotów dowolnych; kuchnie i obszary są zamknięte. Mieszaj oba w schemacie.

## Użyj Tego

Stos w 2026:

| Sytuacja | Podejście |
|-----------|----------|
| Wąska domena (jeden lub dwa intenty) | Oparte na regułach + regex |
| Szeroka domena, dostępne oznaczone dane | LDST (LLaMA + LoRA na danych w stylu MultiWOZ) |
| Szeroka domena, brak etykiet, gotowe do produkcji | LLM + Instructor + schemat Pydantic |
| Mowa / głos | ASR + normalizator + LLM-DST |
| Wielodomenowy przepływ rezerwacji | LLM kierowany schematem z modelami Pydantic na domenę |
| Wrażliwe na zgodność | Głównie regułowe, LLM jako zapasowy z przepływem potwierdzeń |

## Dostarcz To

Zapisz jako `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Ćwiczenia

1. **Łatwe.** Zbuduj stanowy tracker oparty na regułach w `code/main.py` dla 3 slotów (kuchnia, obszar, cena). Przetestuj na 10 ręcznie stworzonych dialogach. Zmierz JGA.
2. **Średnie.** Ten sam zestaw danych z Instructor + Pydantic + małym LLM. Porównaj JGA. Przejrzyj najtrudniejsze tury.
3. **Trudne.** Zaimplementuj obie ścieżki i kieruj ruchem: głównie regułowe, LLM jako zapasowy, gdy regułowe wyemituje <2 sloty z pewnością. Zmierz łączny JGA i koszt wnioskowania na turę.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|-----------------|-----------------------|
| DST | Śledzenie stanu dialogu | Utrzymanie słownika slot-wartość w trakcie tur dialogu. |
| Slot | Jednostka intencji użytkownika | Parametr nazwany, którego potrzebuje backend (kuchnia, data). |
| Domen | Obszar zadania | Restauracja, hotel, taksówka — zbiory slotów. |
| JGA | Joint Goal Accuracy | Frakcja tur, gdzie każdy slot jest poprawny. Wszystko albo nic. |
| MultiWOZ | Benchmark | Wielodomenowy zbiór danych WOZ; standardowa ewaluacja DST. |
| DST bez ontologii | Brak schematu | Generuj nazwy slotów i wartości bezpośrednio, bez ustalonej listy. |
| Korekta | "Właściwie..." | Tura, która nadpisuje wcześniej wypełniony slot. |

## Dalsza Literatura

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — the canonical benchmark.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning for DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — the copy-based DST workhorse.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM-based unsupervised TOD.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — canonical DST results.