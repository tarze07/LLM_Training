# Agenci Generatywni i Symulacja Emergencjalna

> Park i in. 2023 (UIST '23, arXiv:2304.03442) zaludnili **Smallville**, piaskownicę 25 agentów, trójczęściową architekturą: **strumień pamięci** (log w języku naturalnym), **refleksja** (syntezy wyższego rzędu, które agent generuje o własnym strumieniu) i **plan** (zachowanie na poziomie dnia, potem podplany). Przełomowym wynikiem była emergencja przyjęcia walentynkowego: jeden agent zainicjowany celem „chce zorganizować przyjęcie walentynkowe", bez dalszego skryptowania, wyprodukował zaproszenia, które rozprzestrzeniły się w populacji, skoordynowane daty i przyjęcie się odbyło — z 24 agentów, którzy zaczęli bez wiedzy o nim. Ablacje pokazują, że wszystkie trzy komponenty są wymagane dla wiarygodności. Udokumentowane błędy to błędy norm przestrzennych (wchodzenie do zamkniętych sklepów, dzielenie jednoosobowych łazienek). Jest to referencyjna architektura dla symulacji agentowych i wieloagentowej oceny społecznej w 2026 roku.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Problem

Większość systemów wieloagentowych to ściśle skryptowane zespoły: planista planuje, programista koduje, recenzent recenzuje. To działa dla dobrze zdefiniowanych zadań. Nie oddaje to jednak emergencjalnego, nieskryptowanego zachowania, które powstaje, gdy agenci mają pamięć, priorytety i otwarty świat. Badania, symulacje społeczne i coraz częściej AI w grach potrzebują tego drugiego rodzaju.

Architektura Smallville jest dla tego benchmarkiem. Przed Park 2023, najlepsze symulacje agentowe były płytkimi naśladowcami skryptów; po nim, ten wzorzec jest domyślnym dla agentów generatywnych w otwartych światach. Jeśli budujesz symulację agentową w 2026 roku, albo używasz trzech komponentów Smallville, albo wyraźnie uzasadniasz, dlaczego nie.

## Koncepcja

### Trzy komponenty

**Strumień pamięci.** Log tylko do dopisywania obserwacji, działań, refleksji i planów. Każdy wpis ma znacznik czasu, typ, opis (język naturalny) i metadane pochodne: **świeżość**, **ważność** (samoocena 1-10 przez agenta) oraz **istotność** (podobieństwo cosinusowe do bieżącego zapytania).

```
[2026-02-14 09:12:03] obserwacja: Isabella Rodriguez zapytała mnie, czy lubię jazz
[2026-02-14 09:14:22] refleksja:   Lubię długie rozmowy o muzyce
[2026-02-14 10:05:00] plan:         Uczestniczyć w przyjęciu walentynkowym Isabelli dziś wieczorem
```

Wyszukiwanie w pamięci łączy trzy wyniki: `score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`. Wpisy z top-k wchodzą do bieżącego promptu.

**Refleksja.** Okresowo (co N pamięci lub przy ważnych zdarzeniach), agent generuje syntezy wyższego rzędu z ostatnich wspomnień. Wpisy refleksji wracają do strumienia i są dostępne do wyszukania jak każda inna pamięć. W ten sposób agenci budują „zrozumienia" — odpowiednik długoterminowych przekonań w architekturze.

**Plan.** Dekompozycja od góry do dołu. Najpierw plan na poziomie dnia w ogólnych zarysach („idź do pracy, zjedz obiad z Klausem"). Potem plany na poziomie godzin. Potem plany na poziomie działań. Plany są rewidowalne: gdy obserwacja zaprzecza planowi, agent planuje ponownie dotknięty segment.

### Dlaczego wszystkie trzy mają znaczenie (ablacja)

Park i in. przeprowadzili ablacje pomijając każdą z obserwacji, refleksji i planu. Każda ablacja szkodzi wiarygodności:

- Bez **obserwacji** agent traci kontekst i działa na nieaktualnych przekonaniach.
- Bez **refleksji** agent nie może tworzyć przekonań wyższego rzędu; interakcje pozostają płytkie.
- Bez **planu** zachowanie staje się reaktywnym szumem; cele zanikają.

Wyniki wiarygodności od ludzkich oceniających są najwyższe ze wszystkimi trzema; pominięcie któregokolwiek powoduje mierzalną regresję.

### Emergencja walentynkowa

Jeden agent, Isabella Rodriguez, otrzymuje cel „chce zorganizować przyjęcie walentynkowe w Hobbs Cafe 14 lutego o 17:00." Pozostałych 24 agentów nie otrzymuje takiego celu. W ciągu symulowanych dni:

1. Plan Isabelli obejmuje zapraszanie ludzi.
2. Każde zaproszenie staje się obserwacją w strumieniu pamięci sąsiada.
3. Refleksja tego sąsiada generuje przekonania: „Isabella organizuje przyjęcie."
4. Plan sąsiada uwzględnia „uczestnictwo w przyjęciu 14 lutego."
5. Sąsiedzi mówią innym sąsiadom. Zaproszenie rozprzestrzenia się bez centralnej koordynacji.
6. O 17:00 14 lutego kilku agentów zbiera się w Hobbs Cafe.

Jest to emergencja w sensie technicznym: zachowanie na poziomie systemu (przyjęcie) powstało z lokalnych interakcji (dwustronne zaproszenia + indywidualne planowanie) bez centralnego orkiestratora.

### Udokumentowane tryby awarii

Park i in. wyraźnie dokumentują:

- **Błędy norm przestrzennych.** Agenci wchodzą do zamkniętych sklepów. Agenci próbują korzystać z tej samej jednoosobowej łazienki. Agenci jedzą w pomieszczeniach nieprzeznaczonych do jedzenia. Model nie wnioskuje norm społeczno-fizycznych z samego otoczenia.
- **Przepełnienie pamięci.** Głębokie przebiegi symulacji powodują wzrost kosztu wyszukiwania w pamięci. Praktyczne rozwiązanie: okresowa kompakcja pamięci (podsumuj i przytnij) oraz wygaszanie wpisów o niskiej ważności.
- **Halucynacja refleksji.** Refleksje mogą wymyślać relacje, które nie istnieją w strumieniu pamięci. Łagodzenie: dołącz identyfikatory źródłowych pamięci w promptach refleksji i weryfikuj przy wyszukiwaniu.

Są to tryby awarii istotne dla produkcji: każda symulacja agentowa w 2026 roku je dziedziczy.

### Reguły implementacji trzech komponentów

1. **Pamięć jest tylko do dopisywania.** Nigdy nie modyfikuj wpisu pamięci. Korekty to nowe wpisy.
2. **Oceny ważności są tanie.** Wywołaj LLM, aby ocenił ważność 1-10 w momencie zapisu. Zbuforuj wynik.
3. **Wyszukiwanie jest rankingowane, nie filtrowane.** Top-k według łącznego wyniku; nie używaj twardych filtrów (które tracą kontekst).
4. **Refleksja uruchamia się okresowo.** Uruchom, gdy suma ważności nieprzetworzonych wspomnień przekroczy próg (np. 150).
5. **Plany są rewidowalne.** Gdy nowa obserwacja zaprzecza planowi, regeneruj tylko dotknięty segment, a nie cały plan.

### Agenci generatywni poza Smallville

Literatura uzupełniająca z lat 2024-2026 rozszerza architekturę:

- **Symulacja społeczna dla badań polityki / rynku.** Populacje w stylu Smallville symulują zachowanie użytkowników w odpowiedzi na funkcje. Szybciej niż testy A/B; dokładność jest kwestionowana.
- **AI postaci niezależnych (NPC) w grach.** RPG z agentami Smallville produkują emergentne historie zamiast skryptowanych zadań.
- **Benchmarki oceny agentów generatywnych.** Zamiast dokładności zadań, metryką staje się wiarygodność + spójność zachowania w długich przebiegach.

Architektura jest referencją. Rozszerzenia wymieniają komponenty (magazyn wektorowy dla pamięci, refleksja wspomagana wyszukiwaniem, plan neurosymboliczny), ale zachowują trójczęściową strukturę.

### Dlaczego to ma znaczenie dla inżynierii wieloagentowej

Smallville jest dowodem koncepcji, że emergencja wieloagentowa jest tania, gdy komponenty są odpowiednie. Architektura została teraz odtworzona na modelach open source (mniejsze LLMy tracą wiarygodność łagodnie, a nie gwałtownie). Każdy system produkcyjny potrzebujący **emergencjalnego zachowania społecznego** używa tego kształtu. Każdy system potrzebujący **ścisłego wykonywania zadań** używa wzorców nadzorca / role / primitives z wcześniejszych części tej fazy.

## Build It

`code/main.py` implementuje trzy komponenty w stdlib Pythonie ze skryptowanymi politykami agentów (bez prawdziwego LLM). Demo odtwarza emergencję przyjęcia walentynkowego w miniaturze:

- `MemoryStream` — log tylko do dopisywania z wyszukiwaniem według świeżości/ważności/istotności.
- `reflect(stream)` — skryptowana refleksja nad ostatnimi wysokoważnymi wspomnieniami.
- `plan(agent_state)` — plany na poziomie dnia i godzin oparte na bieżących przekonaniach.
- Scenariusz: 5 agentów. Agent 1 zaczyna z „zorganizuj przyjęcie o 17:00." W ciągu symulowanych tyknięć zaproszenie rozprzestrzenia się i agenci zbiegają się.

Uruchomienie:

```
python3 code/main.py
```

Oczekiwane wyjście: ślad tyknięcie po tyknięciu. Do ostatniego tyknięcia co najmniej 3 z 5 agentów pokazuje przyjęcie w swoim planie i zbiegają się w lokalizacji przyjęcia. Pojedynczy inicjator wyprodukował skoordynowane przybycie bez żadnego orkiestratora.

## Use It

`outputs/skill-simulation-designer.md` projektuje symulację agentów generatywnych: liczbę agentów, schemat pamięci, kadencję refleksji, horyzont planowania i metrykę oceny.

## Ship It

Reguły dla produkcyjnych symulacji:

- **Pamięć to baza danych.** Wybierz prawdziwe przechowalnie (wektorowa baza danych, Postgres) w skali. Pamięć w stdlib w pamięci RAM jest dla prototypów.
- **Loguj ślad wyszukiwania.** Dla każdego działania zaloguj top-k wspomnień, które je napędzały. To twoja zdolność debugowania.
- **Budżetuj tokeny na agenta.** Każde pobranie + refleksja + plan na tyknięcie to O(k) wywołań LLM. N agentów × T tyknięć × wywołań-na-tyknięcie może zdominować twój budżet.
- **Kompresuj pamięć okresowo.** Podsumowuj i przytnij wpisy o niskiej ważności. Polityka retencji to decyzja projektowa, a nie szczegół.
- **Wykrywaj naruszenia norm przestrzennych/społecznych** w sposób jawny. Architektura nie uczy się ich.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że 3+ agentów zbiega się na przyjęciu. Zwiększ agentów do 10 — czy emergencja nadal występuje?
2. Usuń krok refleksji. Jak wygląda zachowanie? Odnieś do wyniku ablacji w Park 2023.
3. Wprowadź konkurencyjny cel inicjowany („Klaus chce dać wykład badawczy o 17:00"). Czy agenci się dzielą, czy jeden cel dominuje? Co o tym decyduje?
4. Dodaj ograniczenia przestrzenne: Hobbs Cafe mieści maksymalnie 4 agentów. Czy symulacja radzi sobie z przepełnieniem w sposób elegancki, czy uderza we wzorzec awarii „jednoosobowa łazienka"?
5. Przeczytaj Park i in. (arXiv:2304.03442) Sekcja 6 (eksperymenty z zachowaniem emergencjalnym). Zidentyfikuj jedno zachowanie, którego nie można odtworzyć w twojej miniaturze. Jaki komponent architektury musiałbyś ulepszyć?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Strumień pamięci | „Pamiętnik agenta" | Log tylko do dopisywania obserwacji, działań, refleksji, planów. |
| Świeżość | „Jak nowa jest pamięć" | Wynik zaniku wykładniczego według wieku. |
| Ważność | „Jak bardzo agentowi zależy" | Samoocena 1-10 w momencie zapisu. Buforowana. |
| Istotność | „Jak bardzo związana z bieżącym zapytaniem" | Podobieństwo cosinusowe (oparte na embeddingach). |
| Refleksja | „Przekonanie wyższego rzędu" | Synteza generowana z ostatnich wspomnień, ponownie wchłaniana jako nowa pamięć. |
| Plan | „Dekompozycja dzień/godzina/działanie" | Drzewo planu od góry do dołu. Rewidowalne, gdy obserwacje zaprzeczają. |
| Smallville | „Piaskownica Parka 2023" | Symulacja 25 agentów, która wyprodukowała emergencję walentynkową. |
| Wiarygodność | „Metryka jakości" | Wynik od ludzkiego oceniającego, czy zachowanie wydaje się wiarygodnym agentem. |

## Dalsza Literatura

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — referencyjna architektura
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — miejsce publikacji
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — referencyjna implementacja w Pythonie
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — wcześniejsza sztuka dla agentów z pamięcią strukturalną