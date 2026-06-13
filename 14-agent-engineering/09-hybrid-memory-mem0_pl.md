# Pamięć hybrydowa: wektorowa + grafowa + KV (Mem0)

> Mem0 (Chhikara i in., 2025) traktuje pamięć jako trzy magazyny równoległe — wektorowy dla podobieństwa semantycznego, KV dla szybkiego wyszukiwania faktów, grafowy dla wnioskowania o relacjach między encjami. Warstwa scoringowa łączy wyniki z trzech magazynów podczas wyszukiwania. Jest to standard produkcyjny dla pamięci zewnętrznej w 2026 roku.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnij, dlaczego pojedynczy magazyn (tylko wektorowy, tylko grafowy, tylko KV) jest niewystarczający dla pamięci agenta.
- Wymień trzy równoległe magazyny Mem0 i co każdy z nich optymalizuje.
- Opisz scoring fuzyjny Mem0 — trafność, ważność, świeżość — i dlaczego jest to suma ważona, a nie hierarchia.
- Zaimplementuj zabawkową pamięć trzech magazynów w stdlib z metodą `add()` zapisującą do wszystkich trzech i `search()` łączącą wyniki.

## The Problem

Jeden magazyn jest niewłaściwy dla jednej z trzech klas zapytań:

- **Podobieństwo semantyczne** — "co omawialiśmy na temat dryfu agenta w zeszłym tygodniu?" Wektor wygrywa; KV i graf nie trafiają.
- **Wyszukiwanie faktów** — "jaki jest numer telefonu użytkownika?" KV wygrywa; wektor jest rozrzutny, graf to przerost formy.
- **Wnioskowanie o relacjach** — "którzy klienci mają ten sam podmiot rozliczeniowy?" Graf wygrywa; wektor i KV nie potrafią odpowiedzieć.

Agenci produkcyjni zadają wszystkie trzy typy w jednej sesji. Pamięć jednomagazynowa jest zawsze niewłaściwa dla dwóch z nich. Wkładem Mem0 jest połączenie wszystkich trzech za jednym interfejsem `add`/`search` z funkcją scoringową, która je łączy.

## The Concept

### Trzy magazyny równoległe

Mem0 (arXiv:2504.19413, kwiecień 2025) przy `add(text, user_id, metadata)`:

1. Wyodrębnia kandydackie fakty z tekstu (krok sterowany przez LLM).
2. Zapisuje każdy fakt do magazynu wektorowego (embedding) dla wyszukiwania semantycznego.
3. Zapisuje każdy fakt do magazynu KV z kluczem (user_id, fact_type, entity) dla wyszukiwania O(1).
4. Zapisuje każdy fakt do magazynu grafowego (Mem0g) jako typowane krawędzie dla zapytań o relacje.

Przy `search(query, user_id)`:

1. Magazyn wektorowy zwraca top-k według cosinusa embeddingu.
2. Magazyn KV zwraca bezpośrednie trafienia według klucza (user_id, type, entity) pochodzącego z zapytania.
3. Magazyn grafowy zwraca podgraf osiągalny z encji zapytania.
4. Warstwa scoringowa łączy wyniki z trzech magazynów.

### Scoring fuzyjny

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Trafność** — cosinus wektorowy, dokładne dopasowanie KV, waga ścieżki w grafie.
- **Ważność** — oznaczona w momencie zapisu lub wyuczona (niektóre fakty są ważniejsze: nazwy, ID, polityki).
- **Świeżość** — wykładniczy zanik w czasie od ostatniego zapisu lub odczytu.

Wagi są dostrajane dla każdego produktu. Wyższe `w_recency` dla agentów czatowych; wyższe `w_importance` dla agentów compliance; wyższe `w_relevance` dla agentów wyszukiwania.

### Mem0g i wnioskowanie temporalne

Mem0g dodaje detektor konfliktów. Gdy nowy fakt jest sprzeczny z istniejącą krawędzią, istniejąca krawędź jest oznaczana jako nieprawidłowa, ale nie usuwana. Zapytania temporalne ("jakie było miasto użytkownika w marcu?") przeszukują podgraf ważny w danym czasie.

Jest to zachowanie klasy compliance, które uogólnia wzór unieważniania z Letta.

### Liczby benchmarkowe

Artykuł Mem0 podaje (2025):

- **LoCoMo** (pamięć długich rozmów): 91.6
- **LongMemEval** (pamięć epizodyczna długiego horyzontu): 93.4
- **BEAM 1M** (benchmark pamięci 1M tokenów): 64.1

Linie bazowe (pełny kontekst 128k LLM, płaski magazyn wektorowy, płaski KV) wszystkie tracą o 10+ punktów. Same benchmarki nie uzasadniają wyboru — robi to charakter operacyjny — ale liczby pokazują, że konstrukcja fuzyjna nie jest błędem zaokrągleń.

### Taksonomia zakresów

Mem0 dzieli pamięć według zakresu:

- **Pamięć użytkownika** — utrzymuje się między sesjami, kluczowana po `user_id`.
- **Pamięć sesji** — utrzymuje się w obrębie jednego wątku.
- **Pamięć agenta** — stan instancji agenta.

Każdy zapis wybiera jeden zakres. Wyszukiwanie może pytać między zakresami z wagami per-zakres. Mieszanie zakresów bez zastanowienia prowadzi do incydentów typu "asystent powiedział Alice o projekcie Boba".

### Gdzie ten wzór zawodzi

- **Dryf embeddingu.** Wyniki wektorowe, które wyglądają dobrze przy pierwszych stu zapytaniach, degradują się w miarę wzrostu korpusu. Dodaj okresowe ponowne embeddingowanie najczęściej używanych rekordów.
- **Pełzanie schematu KV.** `(user_id, type, entity)` wygląda prosto, dopóki każdy zespół nie doda własnego `type`. Audytuj zestaw typów kwartalnie.
- **Eksplozja grafu.** Jeden głośny ekstraktor dodaje 50 krawędzi na wiadomość. Ogranicz zapisy grafu na wywołanie `add`; odrzucaj krawędzie o niskim poziomie ufności.

## Build It

`code/main.py` implementuje wzór trzech magazynów w stdlib:

- `VectorStore` — naiwne podobieństwo nakładania tokenów jako substytut embeddingu.
- `KVStore` — słownik kluczowany po `(user_id, fact_type, entity)`.
- `GraphStore` — typowane krawędzie (subject, relation, object, valid).
- `Mem0` — fasada najwyższego poziomu z `add()`, `search()`, scoringiem fuzyjnym i wyszukiwaniem z uwzględnieniem zakresu.
- Prześledzony przykład na konwersacji z wieloma użytkownikami i sesjami.

Uruchom:

```
python3 code/main.py
```

Wynik pokazuje trzy oddzielne ścieżki odtwarzania plus połączone top-k. Zmień wagi scoringowe na górze `main()` i obserwuj zmianę rankingu.

## Use It

- **Mem0 (Apache 2.0)** — gotowe do produkcji. Hostuj samodzielnie z Postgres + Qdrant + Neo4j lub użyj zarządzanej chmury.
- **Letta** — trójwarstwowe core/recall/archival; przynieś własne backendy wektorowe i grafowe.
- **Zep** — komercyjna alternatywa z temporalnym KG i ekstrakcją faktów.
- **Własne implementacje** — gdy potrzebujesz pełnej kontroli nad ekstraktorem (compliance) lub wagami fuzyjnymi (agenci głosowi, gdzie dominuje świeżość).

## Ship It

`outputs/skill-hybrid-memory.md` generuje szkielet pamięci trzech magazynów z scoringiem fuzyjnym, taksonomią zakresów i unieważnianiem temporalnym.

## Exercises

1. Zastąp zabawkowe podobieństwo wektorowe prawdziwym modelem embeddingu (sentence-transformers, Ollama, OpenAI embeddings). Zmierz recall@10 na syntetycznej długiej rozmowie. Czy ranking dryfuje po 1000 zapisach?
2. Dodaj zapytanie temporalne: `search(query, as_of=timestamp)`. Zwracaj tylko rekordy ważne w tym czasie lub wcześniej. Który magazyn wymaga najwięcej pracy?
3. Zaimplementuj detektor konfliktów: jeśli przychodzący fakt jest sprzeczny z krawędzią grafu, oznacz starą krawędź jako nieprawidłową i zaloguj obie. Przetestuj na "użytkownik mieszka w Berlinie" -> "użytkownik mieszka w Lizbonie."
4. Rozszerz scorer fuzyjny o wymiar `user_feedback` (kciuk w górę dla odzyskanych rekordów). Jak zapobiec naginaniu (agent zwraca tylko rekordy, które już polubił)?
5. Przeczytaj dokumentację Mem0 (`docs.mem0.ai`). Przenieś zabawkę na wywołania klienta `mem0`. Porównaj jakość wyszukiwania na tych samych 20 zapytaniach testowych.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pamięć hybrydowa | "Wektor plus graf plus KV" | Trzy magazyny zapisywane równolegle, łączone przy wyszukiwaniu |
| Ekstrakcja faktów | "Ingestia pamięci" | Krok LLM dzielący tekst na krotki (encja, relacja, fakt) |
| Scoring fuzyjny | "Ranking trafności" | Suma ważona trafności, ważności i świeżości |
| Zakres | "Przestrzeń nazw pamięci" | użytkownik / sesja / agent — określa, kto co widzi |
| Mem0g | "Graf pamięci" | Typowane krawędzie z ważnością temporalną dla zapytań o relacje |
| Unieważnianie temporalne | "Miękkie usunięcie" | Oznacz sprzeczne krawędzie jako nieprawidłowe; nigdy nie usuwaj |
| Dryf embeddingu | "Gnicie wyszukiwania" | Jakość wektorów spada w miarę wzrostu korpusu; wykonuj ponowny embedding okresowo |

## Further Reading

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — oryginalny artykuł
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — API produkcyjne, SDK, zarządzana chmura
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — poprzednik z wirtualnym kontekstem
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — siostrzany projekt trójwarstwowy