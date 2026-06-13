# Budżety Akcji, Limity Iteracji oraz Regulatory Kosztów

> Miesięczny koszt LLM agenta e-commerce średniej wielkości wzrósł z 1 200 USD do 4 800 USD po tym, jak jego zespół włączył umiejętność "śledzenia zamówień". To nie jest błąd cenowy. To agent, który znalazł nową pętlę i kontynuował wydawanie w jej ramach. Microsoft Agent Governance Toolkit (2 kwietnia 2026) kodyfikuje obronę przed tą klasą problemów: `max_tokens` na żądanie, budżety tokenowe i dolatowe na zadanie, limity na dzień/miesiąc, limity iteracji, warstwowe kierowanie modeli, buforowanie promptów, okienkowanie kontekstu, punkty kontrolne HITL na kosztownych akcjach, wyłączniki awaryjne przy przekroczeniu budżetu. Claude Code Agent SDK od Anthropic dostarcza te same elementy pod różnymi nazwami. Limity prędkości finansowej — np. odcięcie dostępu przy >50 USD w ciągu 10 minut — łapią pętle szybciej niż limity miesięczne.

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## Problem

Agenci autonomiczni wydają prawdziwe pieniądze na każdym kroku. Zły wynik czata to zła odpowiedź; zła pętla agenta to rachunek. Udokumentowany w branży termin na ten tryb awarii to "Denial of Wallet" — agent kontynuuje wnioskowanie, kontynuuje wywoływanie narzędzi, kontynuuje generowanie kosztów, i nic go nie zatrzymuje, ponieważ nic nie zostało zaprojektowane, aby go zatrzymać.

Rozwiązaniem nie jest jedna liczba. To stos limitów o różnych skalach czasowych i granularnościach: na żądanie, na zadanie, na godzinę, na dzień, na miesiąc. Dobrze zaprojektowany stos łapie niekontrolowaną pętlę w ciągu minut, powolny wyciek w ciągu godzin, a złe wydanie w ciągu dnia. Ten sam stos utrzymuje budżet w ogóle, gdy agent jest długoterminowy i autonomiczny.

To lekcja inżynieryjna: matematyka jest trywialna, dyscyplina jest tym, gdzie zespoły zawodzą. Lista limitów poniżej pochodzi z Microsoft Agent Governance Toolkit lub dokumentacji Anthropic Claude Code Agent SDK.

## Koncepcja

### Stos regulatorów kosztów

1. **`max_tokens` na żądanie.** Proste. Zapobiega emisji nieograniczonego uzupełnienia przez pojedyncze wywołanie.
2. **Budżet tokenowy na zadanie.** W trakcie całego uruchomienia nie przekraczaj N tokenów. Twarde zatrzymanie przy limicie.
3. **Budżet dolatowy na zadanie.** To samo co tokeny, ale w walucie. `max_budget_usd` w Claude Code.
4. **Limit wywołań narzędzia.** Nie więcej niż N wywołań `WebFetch`, N wywołań `shell_exec`, itp.
5. **Limit iteracji (`max_turns`).** Całkowita liczba iteracji pętli agenta; zapobiega nieskończonym pętlom wnioskowania.
6. **Limit na minutę / godzinę / dzień / miesiąc.** Przesuwne okna. Łapie wycieki w różnych skalach czasowych.
7. **Limit prędkości finansowej.** Np. "jeśli wydatek przekroczy 50 USD w ciągu 10 minut, odetnij dostęp." Łapie wypalanie w pętli, zanim zadziałają limity miesięczne.
8. **Warstwowe kierowanie modeli.** Domyślnie używaj mniejszego modelu; eskaluj do większego tylko wtedy, gdy klasyfikator uzna, że zadanie tego wymaga.
9. **Buforowanie promptów.** System prompt i stabilny kontekst przechowywane w cache dostawcy; koszt tokenowy ponownego wysłania jest bliski zeru.
10. **Okienkowanie kontekstu.** Kompakcja / streszczanie, aby utrzymać aktywny kontekst poniżej progu; bezpośrednia redukcja kosztów tokenowych.
11. **Punkty kontrolne HITL przy kosztownych akcjach.** Przed akcją uznaną za kosztowną (długie wywołanie narzędzia, duże pobieranie, kosztowne uaktualnienie modelu), wymagaj zatwierdzenia przez człowieka.
12. **Wyłącznik awaryjny przy przekroczeniu budżetu.** Sesja przerywana, gdy zadziała jakikolwiek limit. Limit jest rejestrowany; wymaga osobnej ścieżki ponownego włączenia.

### Dlaczego stos, a nie jeden limit

Pojedynczy limit miesięczny łapie niekontrolowanego agenta dopiero po utracie budżetu. Pojedynczy limit na żądanie nie łapie niczego na poziomie sesji. Różne tryby awarii wymagają różnych skal czasowych:

- **Niekontrolowana pętla** (agent utknięty w 5-sekundowej pętli ponawiania): łapana przez limit prędkości.
- **Powolny wyciek** (agent wykonujący ~2x oczekiwaną pracę na zadanie): łapany przez limit dzienny.
- **Złe wydanie** (nowa wersja używa 5x tokenów): łapane przez limit tygodniowy / miesięczny.
- **Legalny wzrost** (rzeczywiste zapotrzebowanie, nie błąd): łapany przez limit godzinny / dzienny z wyraźnym logiem.

### Powierzchnia budżetowa Claude Code

Claude Code Agent SDK udostępnia (z publicznej dokumentacji):

- `max_turns` — limit iteracji.
- `max_budget_usd` — limit dolatowy; sesja przerywana przy przekroczeniu.
- `allowed_tools` / `disallowed_tools` — lista dozwolonych i zabronionych narzędzi.
- Punkty zaczepienia przed użyciem narzędzia dla niestandardowego rozliczania kosztów.

Połącz z drabinką trybów uprawnień (Lekcja 10). Sesja `autoMode` bez `max_budget_usd` to niekontrolowana autonomia. Anthropic wyraźnie stwierdza, że Auto Mode wymaga kontroli budżetowych; klasyfikator jest ortogonalny względem kosztów.

### EU AI Act, OWASP Agentic Top 10

Microsoft Agent Governance Toolkit obejmuje wymagania OWASP Agentic Top 10 oraz EU AI Act Artykuł 14 (nadzór człowieka). W produkcji w UE logowanie i egzekwowanie limitów nie są opcjonalne.

### Zaobserwowany przypadek 1 200 USD → 4 800 USD

Rzeczywisty przypadek z dokumentacji Microsoft: agent e-commerce, którego miesięczny koszt potroił się po dodaniu nowego narzędzia. Narzędzie pozwalało agentowi na sprawdzanie statusu zamówienia podczas każdej sesji. Brak wykrywania pętli. Brak limitu na narzędzie. Brak alertu o wzroście tydzień do tygodnia. Rozwiązaniem był limit na narzędzie plus dzienny alert o wzroście. To szablon: każda nowa powierzchnia narzędzia to nowa potencjalna pętla; każde nowe narzędzie potrzebuje własnego limitu i własnego alertu.

## Użyj Tego

`code/main.py` symuluje działanie agenta z warstwowym stosem regulatorów kosztów i bez niego. Symulowany agent wchodzi w pętlę odpytywania po kilku krokach; warstwowy stos łapie go w oknie prędkości, podczas gdy pojedynczy limit miesięczny zadziałałby dopiero kilka dni później.

## Dostarcz To

`outputs/skill-agent-budget-audit.md` audytuje stos regulatorów kosztów proponowanego wdrożenia agenta i wskazuje brakujące warstwy.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że limit prędkości zadziała przed limitem iteracji na trajektorii pętli odpytywania. Teraz wyłącz limit prędkości i zmierz, ile agent "wyda", zanim limit iteracji go złapie.

2. Zaprojektuj zestaw limitów na narzędzie dla agenta przeglądarkowego (Lekcja 11). Które narzędzie potrzebuje najciaśniejszego limitu? Które narzędzie może działać bez ograniczeń bez ryzyka?

3. Przeczytaj dokumentację Microsoft Agent Governance Toolkit. Wymień każdy typ limitu, który toolkit nazywa. Przyporządkuj każdy do jednego z trybów awarii (niekontrolowana pętla, powolny wyciek, złe wydanie, wzrost).

4. Wyceń nocne, bezobsługowe uruchomienie dla realistycznego zadania (np. "triage 50 zgłoszeń w repozytorium"). Ustaw `max_budget_usd` na 2x swojego oszacowania. Uzasadnij 2x.

5. `max_budget_usd` w Claude Code zadziała na zagregowanym koszcie sesji. Zaprojektuj komplementarny limit prędkości, który egzekwowałbyś zewnętrznie. Co wyzwala odcięcie i jak wygląda ponowne włączenie?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|---|---|---|
| Denial of Wallet | "Nie kontrolowany rachunek" | Pętla agenta generująca wydatki bez limitu, który by ją zatrzymał |
| max_tokens | "Limit na żądanie" | Górna granica rozmiaru pojedynczego uzupełnienia |
| max_turns | "Limit iteracji" | Górna granica iteracji pętli agenta w sesji |
| max_budget_usd | "Dolatowy wyłącznik" | Limit kosztu sesji; przerywa przy przekroczeniu |
| Limit prędkości | "Limit stawki" | Limit wydatków w krótkim oknie (np. 50 USD / 10 min) |
| Warstwowe kierowanie | "Najpierw mały model" | Tani model domyślnie; eskaluj tylko gdy klasyfikator uzna za stosowne |
| Buforowanie promptów | "Buforowany system prompt" | Cache dostawcy redukuje koszt tokenowy ponownego wysłania do bliskiego zeru |
| Punkt kontrolny HITL | "Bramka zatwierdzenia przez człowieka" | Wymagane zatwierdzenie przez człowieka przed kosztowną akcją |

## Dalsza Lektura

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`, `max_budget_usd`, listy dozwolonych narzędzi.
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — punkty kontrolne regulatorów kosztów.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — kontrola kosztów po stronie dostawcy.
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — mechanika buforowania.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — profil kosztów dla agentów długoterminowych.