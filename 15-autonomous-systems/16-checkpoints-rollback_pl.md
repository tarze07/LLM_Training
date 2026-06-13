# Punkty Kontrolne i Wycofywanie

> Każde przejście stanu grafu jest utrwalane. Gdy pracownik ulegnie awarii, jego dzierżawa wygasa, a inny pracownik przejmuje pracę od najnowszego punktu kontrolnego. Cloudflare Durable Objects przechowują stan przez godziny lub tygodnie. Proponuj, a potem zatwierdź (Lekcja 15) definiuje plan wycofania dla każdej akcji. Weryfikacja po akcji zamyka pętlę. EU AI Act Artykuł 14 nakazuje skuteczny nadzór człowieka dla systemów wysokiego ryzyka — w praktyce oznacza to, że punkty kontrolne muszą być możliwe do odpytywania, wycofania muszą być przećwiczone, a ślad audytowy musi przetrwać wdrożenie. Ostry tryb awarii: bez kluczy idempotentności i kontroli warunków wstępnych, ponowienie po przejściowym błędzie może podwójnie wykonać już zatwierdzoną akcję. Weryfikacja po akcji jest tym, co to łapie.

**Type:** Learn
**Languages:** Python (stdlib, checkpoint and rollback state machine)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**Time:** ~60 minutes

## Problem

Trwałe wykonanie (Lekcja 12) sprawia, że agent, który uległ awarii, może być wznowiony. Proponuj, a potem zatwierdź (Lekcja 15) sprawia, że zatwierdzona akcja podlega audytowi. Ta lekcja łączy je: co się dzieje, gdy zatwierdzona akcja wykonuje się częściowo, ulega awarii i wznawia się? Kiedy uruchamia się wycofanie i względem jakiego stanu?

Rzeczywiste systemy podłączają to różnie:

- **LangGraph** robi punkty kontrolne każdego przejścia stanu grafu do PostgreSQL. Przy awarii pracownika dzierżawa zwalnia się i inny pracownik wznawia od najnowszego punktu kontrolnego. Przepływy pracy wstrzymują się na `interrupt()`, co samo w sobie jest utrwalane.
- **Cloudflare Durable Objects** przechowują stan na klucz przez godziny lub tygodnie. Współlokują obliczenia z magazynem dla zatwierdzonej akcji.
- **Microsoft Agent Framework** udostępnia prymitywy `Checkpoint` w API przepływu pracy; odtwarzanie plus idempotentność pokrywa ponowienia.

W każdym przypadku kombinacja, która faktycznie działa, to: klucz idempotentności (zapobiega podwójnemu wykonaniu) + kontrola warunku wstępnego (stan jest wciąż tym, co zatwierdziliśmy) + weryfikacja po akcji (efekt uboczny rzeczywiście wystąpił) + wycofanie przy niepowodzeniu weryfikacji.

## Koncepcja

### Każde przejście jest utrwalane

Przejście stanu grafu to dowolny krok, który przenosi przepływ pracy z jednego nazwanego stanu do drugiego. Naiwne implementacje utrwalają tylko w konkretnych punktach zatwierdzenia; produkcyjne implementacje utrwalają każde przejście. Koszt (kilka dodatkowych zapisów) jest mały w porównaniu do zysku z niezawodności (odtworzenie ląduje w dowolnym miejscu, odzyskiwanie dzierżawy jest precyzyjne).

### Odzyskiwanie dzierżawy

Gdy pracownik ulegnie awarii, przepływ pracy nie jest tracony; dzierżawa (krótkotrwałe roszczenie, że ten pracownik wykonuje to uruchomienie) po prostu wygasa. Inny pracownik przejmuje najnowszy punkt kontrolny i wznawia. Mechanizm dzierżawy jest tym, co pozwala systemom produkcyjnym przetrwać toczące się wdrożenia bez utraty pracy w toku.

### Idempotentność plus warunki wstępne

Sama idempotentność nie wystarczy. Rozważmy: przepływ pracy jest zatwierdzony, aby "przenieść 100 USD z A do B, gdy saldo > 1000 USD." Przepływ pracy jest zatwierdzony, ulega awarii w trakcie wykonywania i wznawia się. Jeśli sprawdzany jest tylko klucz idempotentności, a wykonanie wznawia się, przelew uruchamia się raz (poprawnie). Ale rozważmy, że między awarią a wznowieniem saldo A spada do 500 USD przez inny przepływ pracy. Sprawdzanie idempotentności wciąż przechodzi; warunek wstępny nie. Bez kontroli warunku wstępnego dostarczamy debet.

Każda znacząca akcja potrzebuje obu:

- **Klucz idempotentności**: zapobiega podwójnemu wykonaniu.
- **Kontrola warunku wstępnego**: potwierdza, że stan jest wciąż zgodny z tym, co zostało zatwierdzone.

### Weryfikacja po akcji

"Narzędzie zwróciło 200" to nie jest weryfikacja. Prawdziwa weryfikacja ponownie odczytuje stan docelowy i potwierdza, że efekt uboczny rzeczywiście wystąpił. Wzorce:

- Aktualizacja bazy danych: `UPDATE ... RETURNING *` następnie asercja, że zwrócony wiersz jest zgodny z zamierzonym stanem.
- Wysłanie e-maila: sprawdź folder wysłanych dla ID wiadomości po submisji.
- Zapis pliku: odczytaj plik z powrotem i zahaszuj go.
- Wywołanie API: następcze `GET` na docelowym zasobie.

Jeśli weryfikacja się nie powiedzie, przepływ pracy jest w znanym złym stanie. Wycofanie zostaje uruchomione.

### Plany wycofania

Każda znacząca akcja w proponuj, a potem zatwierdź (Lekcja 15) niesie plan wycofania. Typy:

- **Wycofanie w paśmie**: odwróć efekt uboczny bezpośrednio (`DELETE` po `INSERT`, `Send-correction-email` po wysłaniu).
- **Transakcja kompensująca**: nowa akcja, która neutralizuje oryginalną (standardowy wzór SAGA).
- **Wycofanie poza pasmem**: zaalarmuj człowieka, wstrzymaj przepływ pracy, pozostaw zły stan do zbadania.

Brak operacji wycofania ("nie możemy tego cofnąć") musi być nazwany w propozycji. Akcje bez wycofania wymagają silniejszego HITL w momencie zatwierdzenia (Lekcja 15 wyzwanie-i-odpowiedź).

### Operacyjne odczytanie EU AI Act Artykułu 14

Artykuł 14 wymaga "skutecznego nadzoru człowieka" dla systemów wysokiego ryzyka. W ujęciu operacyjnym implementujący odczytują to jako:

- Punkty kontrolne są możliwe do odpytywania przez audytora.
- Wycofania są przećwiczone (przetestowane od końca do końca co najmniej raz).
- Ślad audytowy przetrwa wdrożenie (backend punktów kontrolnych nie jest efemeryczny).
- Nieudane weryfikacje są alarmowane, a nie cicho logowane.

Przepływ pracy, który ulega awarii w trakcie zatwierdzenia, wznawia się i kończy efekt uboczny bez ścieżki weryfikacji + wycofania, nie przechodzi testu Artykułu 14.

### Ostry tryb awarii: podwójne wykonanie

Najczęstszy incydent produkcyjny w tej przestrzeni:

1. Akcja zatwierdzona, klucz idempotentności k.
2. Zatwierdzenie zaczyna się, wykonuje, zwraca 200.
3. Przepływ pracy ulega awarii przed utrwaleniem statusu "zatwierdzone".
4. Przepływ pracy wznawia się; widzi "zatwierdzone, ale nie wykonane"; wykonuje ponownie.
5. Efekt uboczny zadziała dwa razy.

Środek zaradczy: utrwal intencję "w trakcie" przed wykonaniem, wykonaj z kluczem idempotentności, a następnie oznacz jako "zatwierdzone" dopiero po pomyślnej weryfikacji po akcji. Jeśli akcja zadziałała, a zapis statusu się nie powiódł, wiesz, że musisz zweryfikować i (w razie potrzeby) ponownie uruchomić. Jeśli zapis statusu się powiódł, a akcja się nie powiodła, weryfikujesz i uruchamiasz dokładnie raz przez ścieżkę odzyskiwania.

## Użyj Tego

`code/main.py` implementuje przepływ pracy z punktami kontrolnymi z idempotentnością, warunkami wstępnymi, weryfikacją i wycofaniem. Sterownik symuluje cztery scenariusze: czyste uruchomienie, ponowienie po awarii (łapane przez idempotentność), niepowodzenie warunku wstępnego (przepływ pracy przerywany bez uruchamiania), niepowodzenie weryfikacji (wycofanie zadziała).

## Dostarcz To

`outputs/skill-rollback-rehearsal.md` projektuje test przećwiczonego wycofania dla proponowanego przepływu pracy i audytuje backend punktów kontrolnych pod kątem trwałości śladu audytowego.

## Ćwiczenia

1. Uruchom `code/main.py`. Zweryfikuj cztery scenariusze. Dla przypadku awarii w trakcie zatwierdzenia potwierdź, że akcja zadziała dokładnie raz podczas ponowień.

2. Zmodyfikuj wzorzec "najpierw oznacz jako zrobione, potem zrób" tak, aby zapis statusu następował po akcji. Uruchom ponownie scenariusz awarii. Zmierz, ile duplikatów akcji zadziała.

3. Zaprojektuj plan wycofania dla konkretnej akcji produkcyjnej (np. "opublikuj na kanale Slack"). Sklasyfikuj jako w paśmie, kompensującą lub poza pasmem. Uzasadnij wybór.

4. Weź jeden przepływ pracy, który znasz. Zidentyfikuj każde przejście stanu. Oznacz każde wymaganiem trwałości (utrwal / nie utrwalaj). Policz te, których obecnie nie utrwalasz.

5. Test przećwiczonego wycofania: zaprojektuj test od końca do końca, który uruchamia prawdziwy przepływ pracy, powoduje jego awarię i potwierdza, że ścieżka wycofania zadziała. Czego test oczekuje?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|---|---|---|
| Punkt kontrolny | "Punkt zapisu" | Każde przejście stanu grafu utrwalane w trwałym magazynie |
| Dzierżawa | "Roszczenie pracownika" | Krótkotrwałe roszczenie, że pracownik wykonuje uruchomienie; wygasa przy awarii |
| Warunek wstępny | "Bramka stanu" | Asercja, że stan jest wciąż zgodny z zatwierdzoną akcją |
| Weryfikacja po akcji | "Sprawdzenie ponownego odczytu" | Potwierdź, że efekt uboczny rzeczywiście wystąpił w systemie docelowym |
| Wycofanie w paśmie | "Bezpośrednie cofnięcie" | Odwróć efekt uboczny za pomocą odwrotnej operacji |
| Transakcja kompensująca | "Cofnięcie SAGA" | Nowa akcja, która neutralizuje oryginalną |
| Najpierw oznacz jako zrobione | "Kolejność zapisu statusu" | Utrwal status zatwierdzenia przed powrotem z zatwierdzenia |
| Artykuł 14 | "EU AI Act nadzór człowieka" | Operacyjnie: możliwe do odpytywania punkty kontrolne, przećwiczone wycofania, audytowalny ślad |

## Dalsza Lektura

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — prymitywy punktów kontrolnych i odzyskiwanie dzierżawy.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects jako substrat stanu.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulacyjna linia bazowa.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — ramy niezawodności dla długoterminowych przepływów pracy.
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — kształt przepływu pracy dla Claude Code Routines.