# Bloki pamięci i Sleep-Time Compute (Letta)

> MemGPT stał się Lettą w 2024 roku. Ewolucja 2026 dodaje dwa pomysły: dyskretne funkcjonalne bloki pamięci, które model może bezpośrednio edytować, oraz agenta czasu bezczynności, który asynchronicznie konsoliduje pamięć, gdy główny agent jest bezczynny. W ten sposób skalujesz pamięć poza jedną rozmowę.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 minutes

## Learning Objectives

- Wymień trzy poziomy pamięci, których używa Letta (core, recall, archival) i rolę każdego z nich.
- Wyjaśnij wzorzec bloku pamięci: blok Human, blok Persona i bloki zdefiniowane przez użytkownika jako pierwszoklasowe typowane obiekty.
- Opisz, czym jest sleep-time compute, dlaczego leży poza ścieżką krytyczną i dlaczego może uruchamiać silniejszy model niż główny agent.
- Zaimplementuj skryptowaną pętlę dwuagencką, w której główny agent obsługuje odpowiedzi, a agent czasu bezczynności konsoliduje bloki między turami.

## The Problem

MemGPT (Lekcja 07) rozwiązał przepływ sterowania wirtualnej pamięci. Pojawiły się trzy problemy produkcyjne:

1. **Opóźnienie.** Każda operacja pamięciowa leży na ścieżce krytycznej. Jeśli agent musi przycinać, podsumowywać lub uzgadniać, podczas gdy użytkownik czeka, opóźnienie ogonowe eksploduje.
2. **Gnicie pamięci.** Zapisów przybywa. Sprzeczne fakty pozostają. Wyszukiwanie tonie w nieaktualnej treści.
3. **Utrata struktury.** Płaski magazyn archiwalny nie może wyrazić "blok Human jest zawsze w prompcie; blok Persona jest zawsze w prompcie; blok Task zmienia się per sesja."

Letta (letta.com) to przepisanie z 2026 roku. Bloki pamięci czynią strukturę jawną; sleep-time compute przenosi konsolidację poza ścieżkę krytyczną.

## The Concept

### Trzy poziomy

| Poziom | Zakres | Gdzie żyje | Pisany przez |
|------|-------|----------------|------------|
| Core | Zawsze widoczny | Wewnątrz głównego promptu | Wywołanie narzędzia agenta + przepisania sleep-time |
| Recall | Historia rozmów | Do pobrania | Automatyczne logowanie tur |
| Archival | Dowolne fakty | Wektor + KV + graf | Wywołanie narzędzia agenta + ingest sleep-time |

Core to core MemGPT. Recall to bufor rozmów z odciętym ogonem. Archival to zewnętrzny magazyn. Podział porządkuje przeciążenie dwupoziomowe MemGPT.

### Bloki pamięci

Blok to typowana, trwała, edytowalna sekcja poziomu core. Oryginalna praca MemGPT zdefiniowała dwa:

- **Blok Human** — fakty o użytkowniku (imię, rola, preferencje, cele).
- **Blok Persona** — samo-koncepcja agenta (tożsamość, ton, ograniczenia).

Letta uogólnia do dowolnych bloków zdefiniowanych przez użytkownika: blok `Task` dla bieżącego celu, blok `Project` dla faktów o kodzie, blok `Safety` dla twardych ograniczeń. Każdy blok ma `id`, `label`, `value`, `limit` (limit znaków), `description` (aby model wiedział, kiedy go edytować).

Bloki są edytowalne przez powierzchnię narzędzi:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — skondensuj blok, który jest blisko swojego limitu.

### Sleep-time compute

Dodatek Letty z 2025: uruchom drugiego agenta w tle, poza ścieżką krytyczną. Agenci czasu bezczynności przetwarzają transkrypty rozmów i kontekst bazy kodu, zapisują `learned_context` do współdzielonych bloków oraz konsolidują lub unieważniają rekordy archiwalne.

Właściwości, które z tego wynikają:

- **Brak kosztu opóźnienia.** Podstawowe odpowiedzi nie czekają na operacje pamięciowe.
- **Dozwolony silniejszy model.** Agent czasu bezczynności może być droższym, wolniejszym modelem, ponieważ nie jest ograniczony opóźnieniem.
- **Naturalne okno konsolidacji.** Deduplikuj, podsumowuj, unieważniaj sprzeczne fakty, gdy użytkownik nie czeka.

Kształt pasuje do tego, jak pracują ludzie: wykonujesz zadanie, przesypiasz je, pamięć długoterminowa osadza się przez noc.

### Letta V1 i natywne rozumowanie

Letta V1 (`letta_v1_agent`, 2026) wycofuje `send_message`/heartbeat i wbudowane tokeny `Thought:` na rzecz natywnego rozumowania. Responses API (OpenAI) i Messages API z rozszerzonym myśleniem (Anthropic) emitują rozumowanie na osobnym kanale, przekazywanym przez tury (szyfrowanym u dostawców w produkcji). Pętla sterowania to wciąż ReAct. Ślad myśli jest strukturalny, a nie w kształcie promptu.

### Gdzie ten wzorzec się psuje

- **Rozdęcie bloków.** Nieskończone `block_append` szybko uderza w limit. Podłącz podsumowywacz bloku przed zapisem, który przekracza limit.
- **Cichy dryf.** Agent czasu bezczynności przepisuje blok, a główny agent nigdy nie zauważa. Wersjonuj bloki i pokazuj różnice w śladzie.
- **Zatruta konsolidacja.** Agent czasu bezczynności przetwarza treść dostępną dla atakującego do core. Lekcja 27 dotyczy również powierzchni czasu bezczynności.

## Build It

`code/main.py` implementuje:

- `Block` — id, label, value, limit, description.
- `BlockStore` — CRUD + pomocnik `near_limit(label)`.
- Dwóch skryptowanych agentów — `PrimaryAgent` obsługuje turę, `SleepTimeAgent` konsoliduje między turami.
- Ślad, który pokazuje trzyturą rozmowę z zapisami bloków, plus przejście czasu bezczynności, które podsumowuje blok i unieważnia nieaktualny fakt.

Uruchom:

```
python3 code/main.py
```

Transkrypt pokazuje podział: tury podstawowe są szybkie i produkują surowe zapisy; przejście bezczynności kompaktuje i czyści.

## Use It

- **Letta** (letta.com) dla referencyjnej implementacji. Samodzielny hosting lub zarządzana chmura.
- **Umiejętności Claude Agent SDK** jako wiedza w kształcie bloku — umiejętność to nazwany, wersjonowany, pobieralny blok instrukcji, który agent ładuje na żądanie.
- **Niestandardowe konstrukcje** dla zespołów, które chcą kontroli nad backendem przechowywania. Użyj kontraktu API Letta, aby móc migrować później.

## Ship It

`outputs/skill-memory-blocks.md` generuje system bloków w kształcie Letta z hakami czasu bezczynności dla dowolnego środowiska uruchomieniowego, w tym reguły bezpieczeństwa i okablowanie cytowań.

## Exercises

1. Dodaj narzędzie `block_summarize`, które zastępuje wartość bloku podsumowaniem wygenerowanym przez model, gdy `near_limit` zwraca true. Który próg wyzwalania minimalizuje zarówno wywołania podsumowywania, jak i przepełnienie bloku?
2. Zaimplementuj deduplikację czasu bezczynności w archiwum: dwa rekordy, których tekst ma >90% nakładania tokenów, zwijają się w jeden. Rób to tylko w przejściu bezczynności, nigdy na ścieżce krytycznej.
3. Wersjonuj bloki. Przy każdym zapisie rejestruj starą wartość i różnicę. Udostępnij `block_history(label)`, aby operatorzy mogli debugować "dlaczego agent zapomniał X."
4. Traktuj agentów czasu bezczynności jako niezaufanych pisarzy. Gdy dotykają bloku Persona lub Safety, wymagaj przeglądu drugiego agenta przed zatwierdzeniem.
5. Przenieś przykład do użycia API Letta (`letta_v1_agent`). Co zmienia się w schemacie bloku i jak natywne rozumowanie zmienia kształt śladu?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|------|----------------|------------------------|
| Blok pamięci | "Edytowalna sekcja promptu" | Typowany, trwały, edytowalny przez LLM segment pamięci core |
| Blok Human | "Pamięć użytkownika" | Fakty o użytkowniku, przypięte w core |
| Blok Persona | "Tożsamość agenta" | Samo-koncepcja, ton, ograniczenia, przypięte w core |
| Sleep-time compute | "Asynchroniczna praca pamięci" | Drugi agent wykonujący konsolidację poza ścieżką krytyczną |
| Core / Recall / Archival | "Poziomy" | Trójwarstwowy podział pamięci: zawsze-widoczne / rozmowa / zewnętrzne |
| Limit bloku | "Limit" | Limit znaków na blok; wymusza podsumowywanie |
| Natywne rozumowanie | "Kanał myślenia" | Wyjście rozumowania na poziomie dostawcy, a nie `Thought:` na poziomie promptu |
| Nauczony kontekst | "Wyjście snu" | Fakty, które agent czasu bezczynności zapisuje do współdzielonych bloków |

## Further Reading

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — wzorzec bloku
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — asynchroniczna konsolidacja
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — przepisanie z natywnym rozumowaniem
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — geneza

(End of file - total 130 lines)