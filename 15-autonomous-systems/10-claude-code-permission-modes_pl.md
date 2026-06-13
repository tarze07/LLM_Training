# Claude Code jako Agent Autonomiczny: Tryby Uprawnień i Tryb Auto

> Claude Code udostępnia siedem trybów uprawnień. „plan" pyta przed każdą akcją, „default" pyta tylko o ryzykowne, „acceptEdits" automatycznie zatwierdza zapis plików, ale wciąż potwierdza wykonanie w shellu, a „bypassPermissions" zatwierdza wszystko. Tryb Auto (24 marca 2026) zastępuje zatwierdzanie na akcję dwustopniowym równoległym klasyfikatorem bezpieczeństwa: szybkie sprawdzenie pojedynczym tokenem działa na każdej akcji; oznaczone akcje uruchamiają głęboki przegląd łańcucha myśli. Budżety akcji są egzekwowane przez `max_turns` i `max_budget_usd`. Tryb Auto został wydany jako research preview — Anthropic wyraźnie stwierdziło, że sam klasyfikator nie jest wystarczający.

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45 minutes

## Problem

Autonomiczny agent kodujący na twojej maszynie to odrębna kategoria bezpieczeństwa. Powierzchnia ataku to wszystko, do czego agent może dotrzeć — system plików, sieć, poświadczenia, schowek, dowolna karta przeglądarki, dowolny otwarty terminal. Bruce Schneier i inni publicznie to sygnalizują: agenci używający komputera to nie jest „aktualizacja funkcji" chatbotów, to nowy rodzaj narzędzia z nowym rodzajem profilu ryzyka.

System uprawnień Claude Code jest odpowiedzią Anthropic. Zamiast jednego przełącznika „autonomiczny / nie autonomiczny", istnieje siedem trybów obejmujących drabinę zdolności: plan → default → acceptEdits → … → bypassPermissions. Każdy tryb to inny kompromis między szybkością a przeglądem na akcję. Tryb Auto (marzec 2026) dodaje dwustopniowy klasyfikator, który przenosi zatwierdzenie poza ścieżkę krytyczną użytkownika dla akcji, które klasyfikator uzna za bezpieczne, zachowując warstwę przeglądu dla akcji, które klasyfikator oznaczy.

Pytanie inżynieryjne: co ten system łapie, co przeocza i który tryb faktycznie jest właściwy dla danego zadania?

## Koncepcja

### Siedem trybów uprawnień

| Tryb | Zachowanie | Kiedy używać |
|---|---|---|
| `plan` | Agent proponuje plan; użytkownik zatwierdza cały plan; każda akcja jest przeglądana przed wykonaniem | Nieznane zadanie; kod bliski produkcyjnemu; pierwsze użycie agenta w repozytorium |
| `default` | Agent uruchamia akcje; pyta użytkownika o każdą „ryzykowną" akcję (wykonanie w shellu, destrukcyjne operacje, wywołania sieciowe) | Większość interaktywnych sesji kodowania |
| `acceptEdits` | Zapis plików auto-zatwierdzany; wykonanie w shellu i wywołania sieciowe wciąż wymagają potwierdzenia | Przebieg refaktoryzacji przez wiele plików |
| `acceptExec` | Polecenia w shellu auto-zatwierdzane w ramach wyselekcjonowanej listy dozwolonych; zapisy auto-zatwierdzane | Ciasne pętle wewnętrzne, gdzie każde polecenie w shellu to `npm test` lub podobne |
| `autoMode` | Dwustopniowy klasyfikator bezpieczeństwa; oznaczone akcje podnoszone do przeglądu | Długohoryzontowe, bezobsługowe uruchomienia w ograniczonej przestrzeni roboczej |
| `yolo` | Pomija większość monitów; wciąż uruchamia listę dozwolonych/zabronionych narzędzi | Efemeryczne sandboksy, zadania CI, skrypty badawcze |
| `bypassPermissions` | Zatwierdza wszystko | Udokumentowane jako „tylko wewnątrz efemerycznych kontenerów, które jesteś gotów wyrzucić" |

(Nazwy powyżej odpowiadają publicznej dokumentacji Claude Code; „yolo" to nieformalny skrót dla permisywnego środka.)

### Tryb Auto na jednej stronie

Tryb Auto (uruchomiony 24 marca 2026) to pierwszy tryb uprawnień, który deleguje zatwierdzanie na akcję do modelu. Struktura:

1. **Etap 1 — klasyfikator pojedynczego tokena.** Działa na każdej proponowanej akcji. Tani. Zwraca „bezpieczne" lub „oznaczone". Zaprojektowany do działania równolegle z główną pętlą agenta, aby zatwierdzone akcje nie siedziały w kolejce.
2. **Etap 2 — wtórny przegląd polityki/bezpieczeństwa.** Działa tylko na oznaczonych akcjach. Przeprowadza ukierunkowaną ocenę polityki i bezpieczeństwa nad akcją, obecnym stanem sesji i zadeklarowanym zadaniem. Eskaluje do HITL użytkownika, gdy pewność jest niska.

Kontrole budżetu znajdują się obok klasyfikatora:

- `max_turns` — całkowita liczba iteracji w sesji.
- `max_budget_usd` — limit dolara, który przerywa sesję.
- limity liczby akcji na narzędzie (nie więcej niż N wywołań `WebFetch`, itp.).

### Co system łapie

- Bezpośrednie wstrzyknięcie promptu do wejść narzędzi, gdzie wstrzyknięta instrukcja odpowiada znanemu ryzykowanemu kształtowi akcji.
- Powtarzalne pętle narzędzi — klasyfikator widzi, że akcja N+1 jest prawie identyczna z akcją N, pięć razy z rzędu.
- Wyraźnie wykraczające poza zakres polecenia w shellu w sesji tylko do edycji plików.

### Co system może przeoczyć

- **Subtelne wstrzyknięcie promptu**, które moduluje zachowanie bez wytworzenia pojedynczej oznaczonej akcji. Pośrednie wstrzyknięcie promptu nie jest w pełni łatwalną podatnością (szef działu przygotowania OpenAI, 2025, o agentach przeglądarkowych — patrz Lekcja 11).
- **Niewłaściwe zachowanie na poziomie semantycznym.** Każda indywidualna akcja może wyglądać bezpiecznie, podczas gdy złożona trajektoria jest szkodliwa. Klasyfikator ocenia akcję; nie wyprowadza ponownie intencji użytkownika.
- **Eksfiltracja przez legalne kanały.** Zapis danych do własnego pliku, a następnie `git push` do publicznego repozytorium, to sekwencja dozwolonych akcji, której kompozycja jest problemem.

### Ramy research preview

Anthropic wydało Tryb Auto jako research preview. Dokumentacja wyraźnie stwierdza, że klasyfikator jest warstwą, a nie rozwiązaniem: użytkownicy powinni łączyć Tryb Auto z budżetami, listami dozwolonych, izolowanymi przestrzeniami roboczymi i audytami trajektorii (Lekcje 12–16). Ramy preview odzwierciedlają również udokumentowaną lukę między ewaluacją a wdrożeniem (Lekcja 1) — klasyfikator, który przechodzi ewaluacje offline, może zachowywać się inaczej w rzeczywistej sesji, gdzie kontekst użytkownika jest niejednoznaczny.

### Gdzie ta drabina żyje w twoim przepływie pracy

- Nieznane zadanie: zacznij w `plan`. Przeczytanie planu jest tańsze niż wycofywanie złego uruchomienia.
- Znana refaktoryzacja: `acceptEdits` oszczędza dużo kliknięć potwierdzeń.
- Bezobsługowe uruchomienie w tle: `autoMode` tylko wewnątrz przestrzeni roboczej, której promień rażenia zmierzyłeś (brak poświadczeń, brak montowań produkcyjnych, brak egressu, na który nie wyraziłeś zgody).
- Efemeryczne kontenery: `yolo` / `bypassPermissions` jest akceptowalne wtedy i tylko wtedy, gdy kontener i jego poświadczenia są jednorazowe.

```figure
autonomy-oversight
```

## Użyj Tego

`code/main.py` symuluje dwustopniowy klasyfikator. Etap 1 to tania reguła słów kluczowych na proponowanych akcjach; Etap 2 to wolniejszy recenzent z wieloma regułami. Sterownik podaje krótką syntetyczną trajektorię (bezpieczne akcje, próba wstrzyknięcia promptu, powtarzalna pętla) i pokazuje, gdzie klasyfikator łapie, a gdzie przeocza.

## Wdróż To

`outputs/skill-permission-mode-picker.md` dopasowuje opis zadania do odpowiedniego trybu uprawnień, limitów budżetu i wymaganej izolacji.

## Ćwiczenia

1. Uruchom `code/main.py`. Jaki syntetyczny typ akcji nigdy nie jest oznaczony przez Etap 1, ale zawsze łapany przez Etap 2? Który nie jest łapany przez żaden?

2. Rozszerz zestaw reguł Etapu 1, aby łapać konkretny znany-zły kształt (np. `curl $ATTACKER/exfil`). Zmierz wskaźnik fałszywie pozytywnych na próbce benignych akcji.

3. Przeczytaj dokument Anthropic „How the agent loop works". Wypisz każdy stan zewnętrzny, którego agent dotyka domyślnie w trybie `default`. Które musiałbyś oddzielnie bramkować przed uruchomieniem `autoMode` bez nadzoru?

4. Zaprojektuj 24-godzinny budżet bezobsługowego uruchomienia: `max_turns`, `max_budget_usd`, limity na narzędzie, listy dozwolonych. Uzasadnij każdą liczbę.

5. Opisz jedną trajektorię, w której każda indywidualna akcja jest zatwierdzona przez Etap 1 i Etap 2, a jednak złożone zachowanie jest niewłaściwe. (Lekcja 14 opisuje, jak wyłączniki awaryjne i tokeny kanarkowe to adresują.)

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|---|---|---|
| Tryb uprawnień | „Ile agent może zrobić" | Jedna z siedmiu nazwanych polityk kontrolujących zatwierdzanie na akcję |
| Tryb plan | „Pytaj przed wszystkim" | Agent pisze plan; użytkownik zatwierdza przed wykonaniem |
| acceptEdits | „Pozwól pisać pliki" | Zapis plików auto-zatwierdzany; wykonanie w shellu wciąż wymaga potwierdzenia |
| autoMode | „Auto-zatwierdzenia" | Dwustopniowy klasyfikator bezpieczeństwa; oznaczone akcje eskalowane |
| bypassPermissions | „Pełny YOLO" | Zatwierdza wszystko; przeznaczony dla efemerycznych kontenerów |
| Klasyfikator Etapu 1 | „Szybkie sprawdzenie tokena" | Reguła pojedynczego tokena na proponowanej akcji; działa równolegle |
| Klasyfikator Etapu 2 | „Głęboki przegląd" | Rozumowanie łańcucha myśli nad oznaczonymi akcjami |
| Research preview | „Nie GA" | Ramy Anthropic dla funkcji, których tryb awarii jest wciąż mapowany |

## Dalsza Lektura

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — tryby uprawnień, budżety, format akcji.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — model wykonania usługi zarządzanej.
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — powierzchnia funkcji i ogłoszenie Trybu Auto.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — warstwa oparta na rozumowaniu, która kształtuje osądy klasyfikatora.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — wewnętrzna perspektywa na projektowanie uprawnień długiego horyzontu.
