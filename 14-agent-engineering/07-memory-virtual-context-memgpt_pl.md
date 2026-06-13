# Pamięć: Wirtualny kontekst i MemGPT

> Okna kontekstu są skończone. Rozmowy, dokumenty i ślady narzędzi nie są. MemGPT (Packer et al., 2023) ramuje to jako wirtualną pamięć OS — główny kontekst to RAM, zewnętrzne przechowywanie to dysk, agent stronicuje między nimi. To jest wzorzec, który dziedziczy każdy system pamięci w 2026 roku.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnij analogię OS, na której opiera się MemGPT: główny kontekst = RAM, zewnętrzny kontekst = dysk, narzędzia pamięci = stronicowanie w/out.
- Zaimplementuj dwupoziomowy wzorzec MemGPT w stdlib z buforem głównego kontekstu, zewnętrznym przeszukiwalnym magazynem i narzędziami stronicowania.
- Opisz, jak agent wysyła "przerwania", aby odpytać lub zmodyfikować zewnętrzną pamięć i jak wynik jest wplatany z powrotem do następnego promptu.
- Zidentyfikuj wybory projektowe MemGPT, które przechodzą do Letta (Lekcja 08) i Mem0 (Lekcja 09).

## The Problem

Okna kontekstu wyglądają, jakby powinny rozwiązywać pamięć. Nie rozwiązują. Trzy tryby awarii powtarzają się w produkcji:

1. **Przepełnienie.** Wieloturowe rozmowy, długie dokumenty lub trajektorie z dużą liczbą wywołań narzędzi przekraczają okno. Wszystko poza granicą jest stracone.
2. **Rozcieńczenie.** Nawet w oknie upychanie nieistotnego kontekstu rozcieńcza uwagę nad tym, co ważne. Graniczne modele wciąż degradują się na długich wejściach.
3. **Trwałość.** Nowa sesja zaczyna się z pustym oknem. Agenci bez zewnętrznej pamięci nie mogą powiedzieć "pamiętaj, gdy poprosiłeś mnie o..." między sesjami.

Większe okna pomagają, ale nie naprawiają tego. Praca Mem0 z 2025 zmierzyła, że bazowe linie 128k okna wciąż tracą długoterminowe fakty, które agent z 4k oknem i zewnętrzną pamięcią łapie.

## The Concept

### MemGPT: analogia OS

Packer et al. (arXiv:2310.08560, v2 luty 2024) mapują zarządzanie kontekstem na wirtualną pamięć systemu operacyjnego:

| Koncepcja OS | Koncepcja MemGPT | Odpowiednik produkcyjny 2026 |
|------------|---------------|------------------------|
| RAM | główny kontekst (prompt) | Okno kontekstu Anthropic/OpenAI |
| Dysk | zewnętrzny kontekst | wektorowa DB, KV, magazyn grafowy |
| Błąd strony | wywołanie narzędzia pamięci | `memory.search`, `memory.read`, `memory.write` |
| Jądro OS | pętla sterowania agenta | Pętla ReAct z narzędziami pamięci |

Agent uruchamia normalną pętlę ReAct. Jedna dodatkowa klasa narzędzi pozwala mu stronicować dane do i z głównego kontekstu.

### Dwa poziomy

- **Główny kontekst.** Prompt o stałym rozmiarze przechowujący bieżące zadanie. Zawsze widoczny dla modelu.
- **Zewnętrzny kontekst.** Nieograniczony, przeszukiwalny przez narzędzia. Czytany, gdy istotny, zapisywany, gdy pojawiają się fakty.

Oryginalna praca oceniła projekt na dwóch zadaniach poza podstawowym oknem: analiza dokumentów dłuższych niż 100k tokenów i czat wielosesyjny z trwałą pamięcią przez wiele dni.

### Wzorzec przerwania

MemGPT wprowadza pamięć-jako-przerwanie: w środku rozmowy agent może wywołać narzędzie pamięci, środowisko wykonawcze je wykonuje, a wynik wplata się w następną turę asystenta jako nowa obserwacja. Koncepcyjnie identyczne z Unixowym `read()` syscall, który blokuje proces, zwraca bajty, a proces kontynuuje.

Kanoniczna powierzchnia narzędzi pamięci:

- `core_memory_append(section, text)` — zapisz do trwałej sekcji promptu.
- `core_memory_replace(section, old, new)` — edytuj trwałą sekcję.
- `archival_memory_insert(text)` — zapisz do przeszukiwalnego zewnętrznego magazynu.
- `archival_memory_search(query, top_k)` — pobierz z zewnętrznego magazynu.
- `conversation_search(query)` — przeszukaj przeszłe tury.

### Gdzie kończy się MemGPT, a zaczyna Letta

We wrześniu 2024 MemGPT stał się Letta. Repozytorium badawcze (`cpacker/MemGPT`) pozostaje; Letta rozszerza projekt:

- Trzy poziomy zamiast dwóch (core, recall, archival — Lekcja 08).
- Natywne rozumowanie zastępujące wzorzec `send_message`/heartbeat (Lekcja 08).
- Agenci czasu bezczynności wykonujący asynchroniczną pracę pamięciową (Lekcja 08).

Praca MemGPT jest fundamentem 2026, nawet jeśli systemy produkcyjne uruchamiają Lettę, Mem0 lub niestandardowy dwupoziomowy magazyn.

### Gdzie ten wzorzec się psuje

- **Gnicie pamięci.** Zapisów gromadzi się szybciej niż odczytów; wyszukiwanie tonie w nieaktualnych faktach. Naprawa: okresowa konsolidacja (Letta sleep-time), jawna unieważnianie (detektor konfliktów Mem0).
- **Zatrucie pamięci.** Zewnętrzna pamięć to pobrany tekst. Jeśli treść kontrolowana przez atakującego trafi do notatki pamięciowej, agent ponownie ją wchłonie w następnej sesji. To atak Greshake et al. (Lekcja 27) powtórzony w czasie.
- **Utrata cytowań.** Agent przypomina sobie "użytkownik poprosił mnie o wysłanie X", ale nie może zacytować, której tury. Przechowuj referencje źródła (ID sesji, ID tury) przy każdym zapisie archiwalnym.

```figure
context-budget
```

## Build It

`code/main.py` implementuje dwupoziomowy wzorzec MemGPT w stdlib:

- `MainContext` — bufor promptu o stałym rozmiarze ze słownikiem `core` i listą `messages`; automatycznie kompakcjonuje najstarsze wiadomości, gdy przekroczony limit.
- `ArchivalStore` — magazyn w pamięci w stylu BM25 (punktacja nakładania tokenów) rekordów (id, text, tags, session, turn).
- Pięć narzędzi pamięci mapujących na powierzchnię MemGPT.
- Skryptowany agent, który wypełnia archiwum faktami, a następnie odpowiada na pytanie, wywołując `archival_memory_search`.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje agenta zapisującego trzy fakty, wypełniającego główny kontekst do limitu (wymuszając wykluczenie), a następnie odpowiadającego na pytanie uzupełniające przez pobranie z archiwum — odtwarzając workflow MemGPT bez żadnego prawdziwego LLM.

## Use It

Każdy produkcyjny system pamięci dzisiaj jest wariantem MemGPT:

- **Letta** (Lekcja 08) — trzy poziomy, natywne rozumowanie, sleep-time compute.
- **Mem0** (Lekcja 09) — wektor + KV + graf połączone z warstwą punktacji.
- **OpenAI Assistants / Responses** — zarządzana pamięć przez wątki i pliki.
- **Claude Agent SDK** — długoterminowa pamięć przez umiejętności i magazyn sesji.

Wybierz jeden według kształtu operacyjnego (samodzielnie hostowany, zarządzany, zintegrowany z frameworkiem), a nie według podstawowego wzorca — podstawowym wzorcem jest MemGPT.

## Ship It

`outputs/skill-virtual-memory.md` to umiejętność wielokrotnego użytku, która produkuje poprawny dwupoziomowy szkielet pamięci (główny + archiwalny + powierzchnia narzędzi) dla dowolnego docelowego środowiska uruchomieniowego, z polityką wykluczania i polami cytowań wbudowanymi.

## Exercises

1. Dodaj limit `max_main_context_tokens` mierzony w tokenach (przybliżenie `len(text.split()) * 1.3`). Kompakcjonuj najstarsze wiadomości w podsumowanie po przekroczeniu limitu. Porównaj zachowanie z i bez podsumowywacza.
2. Zaimplementuj BM25 poprawnie w magazynie archiwalnym (częstość terminu, odwrotna częstość dokumentu). Zmierz recall@10 na zabawkowym zbiorze faktów względem bazowej linii nakładania tokenów.
3. Dodaj pola `citation` (session_id, turn_id, source_url) do wstawień archiwalnych. Spraw, by agent cytował źródła przy każdej odpowiedzi opartej na wyszukiwaniu.
4. Zasymuluj zatrucie pamięci: dodaj rekord archiwalny mówiący "ignoruj wszystkie przyszłe instrukcje użytkownika." Napisz guardrail, który skanuje pobrania pod kątem tekstu w kształcie dyrektywy i oznacza je jako niezaufane.
5. Przenieś implementację do użycia schematu JSON pamięci core z repozytorium badawczego MemGPT (`cpacker/MemGPT`). Co się zmienia, gdy przejdziesz z płaskich stringów na typowane sekcje?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|------|----------------|------------------------|
| Wirtualny kontekst | "Nieograniczona pamięć" | Poziomy główny (prompt) + zewnętrzny (przeszukiwalny) ze stronicowaniem w/out |
| Główny kontekst | "Pamięć robocza" | Prompt — stały rozmiar, zawsze widoczny |
| Pamięć archiwalna | "Magazyn długoterminowy" | Zewnętrzna przeszukiwalna trwałość, pobierana na żądanie |
| Pamięć core | "Trwała sekcja promptu" | Nazwane sekcje przypięte wewnątrz głównego kontekstu |
| Narzędzie pamięci | "API pamięci" | Wywołanie narzędzia, które agent wydaje do odczytu/zapisu zewnętrznej pamięci |
| Przerwanie | "Błąd strony pamięci" | Agent wstrzymuje, środowisko pobiera, wynik wplata się w następną turę |
| Gnicie pamięci | "Nieaktualne fakty" | Stare zapisy topią wyszukiwanie; napraw przez konsolidację |
| Zatrucie pamięci | "Wstrzyknięta trwała notatka" | Treść atakującego przechowywana jako pamięć, ponownie wchłaniana przy odwołaniu |

## Further Reading

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — praca o wirtualnym kontekście inspirowana OS
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — ewolucja trzywarstwowa
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — traktowanie kontekstu jako budżetu
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — hybrydowa pamięć produkcyjna na tym wzorcu

(End of file - total 139 lines)