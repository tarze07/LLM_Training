# Specjalizacja Ról — Planista, Krytyk, Wykonawca, Weryfikator

> Najpopularniejsza dekompozycja wieloagentowa w 2026 roku: jeden agent planuje, jeden wykonuje, jeden krytykuje lub weryfikuje. MetaGPT (arXiv:2308.00352) formalizuje to jako SOP zakodowane w promptach ról — Menedżer Produktu, Architekt, Kierownik Projektu, Inżynier, Inżynier QA — zgodnie z `Code = SOP(Team)`. ChatDev (arXiv:2307.07924) łączy projektanta, programistę, recenzenta i testera w "łańcuchu czatu" z "komunikatywną dehalucynacją" (agenci jawnie proszą o brakujące szczegóły). Weryfikator jest kluczowy: Cemri i in. (MAST, arXiv:2503.13657) pokazują, że każdą awarię systemu wieloagentowego można prześledzić do brakującej lub uszkodzonej weryfikacji. PwC odnotowało 7-krotny wzrost dokładności (10% → 70%) dzięki strukturalnym pętlom walidacyjnym w CrewAI.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 minutes

## Problem

Ogólne systemy wieloagentowe produkują ogólne wyniki. Trzech programistów w grupowym czacie pisze trzy warianty tego samego przeciętnego kodu. Możesz dodać więcej agentów, dodać więcej rund i wciąż nie przekroczyć progu jakości.

Rozwiązaniem nie jest więcej agentów — to *inni* agenci. Przydziel odrębne role. Daj krytykowi narzędzia, których nie ma planista. Daj weryfikatorowi obiektywny zestaw testów. Teraz system ma wewnętrzną niezgodność z ugruntowaną korektą, a nie tylko równoległe zgadywanie.

## Koncepcja

### Cztery kanoniczne role

**Planista.** Czyta cel, tworzy listę kroków lub specyfikację. Narzędzia: wyszukiwanie wiedzy, dokumentacja. Wynik: strukturalny plan.

**Wykonawca.** Czyta jeden krok planu naraz, tworzy artefakt. Narzędzia: rzeczywiste narzędzia pracy (kompilator, shell, klient API). Wynik: artefakt.

**Krytyk.** Czyta wynik wykonawcy w kontekście intencji planisty. Narzędzia: dostęp tylko do odczytu artefaktu, analiza statyczna. Wynik: akceptacja/odrzucenie z uzasadnieniem.

**Weryfikator.** Czyta artefakt i wykonuje deterministyczne sprawdzenie. Narzędzia: runner testów, kontroler typów, walidator schematów. Wynik: zaliczenie/niezaliczenie z dowodami.

Krytyk jest subiektywny, opiniotwórczy, często oparty na LLM. Weryfikator jest obiektywny, deterministyczny, często oparty na kodzie. To nie ta sama rola.

### Wzorzec SOP w MetaGPT

MetaGPT (arXiv:2308.00352) koduje procedury SOP inżynierii oprogramowania jako prompty ról:

- **Menedżer Produktu** pisze PRD.
- **Architekt** tworzy projekt systemu.
- **Kierownik Projektu** dzieli zadania.
- **Inżynier** implementuje.
- **Inżynier QA** uruchamia testy.

Każda rola ma ścisły schemat wejścia/wyjścia. Prompt roli określa, czym rola *jest* i co *musi wyprodukować*. Sformułowanie `Code = SOP(Team)` — deterministyczne SOP zamieniają zespół LLM-ów w przewidywalny pipeline.

### Komunikatywna dehalucynacja w ChatDev

ChatDev dodaje kluczowy ruch: gdy wykonawca potrzebuje konkretnego szczegółu, który nie został uwzględniony w planie, jawnie pyta projektanta przed kontynuacją. Zapobiega to klasycznej awarii LLM polegającej na prawdopodobnym wymyślaniu szczegółu.

Implementacja: prompt roli zawiera "gdy potrzebujesz konkretnych informacji, których ci nie podano, zapytaj odpowiednią rolę po imieniu przed wyprodukowaniem wyniku."

### Dlaczego weryfikator jest najważniejszy

Cemri i in. (MAST) prześledzili 1642 awarie wykonania systemów wieloagentowych. 21,3% stanowiły luki weryfikacyjne — system dostarczył odpowiedź, której nikt nie sprawdził. Pozostałe 79% często sprowadza się do "była kontrola, która zakończyła się cicho lub nigdy nie została uruchomiona." Weryfikacja jest kluczową rolą.

PwC odnotowało (wdrożenia CrewAI, 2025), że dodanie strukturalnej pętli walidacyjnej zwiększyło dokładność z 10% do 70%. 7-krotny wzrost dzięki jednej roli.

### Krytyk a weryfikator

- Krytyk to LLM recenzujący artefakt pod kątem jakości. Subiektywny. Może zostać oszukany przez wiarygodną prozę.
- Weryfikator to deterministyczny program uruchamiany na artefakcie. Obiektywny. Daje zaliczenie/niezaliczenie z dowodami.

Używaj obu. Krytyk wychwytuje problemy stylistyczne, których weryfikator nie potrafi wyrazić. Weryfikator wychwytuje błędy, których krytyk nie widzi, ponieważ ujawniają się dopiero w czasie wykonania.

### Antywzorzec

Każda rola w twoim systemie to LLM, a wynik każdej roli to "wygląda dobrze." Klasyczny tryb awarii MAST. Dodaj co najmniej jednego weryfikatora, którego zaliczenie/niezaliczenie jest rozstrzygane przez kod, a nie przez LLM.

### Mapowania frameworków

- **CrewAI** — `Agent(role, goal, backstory)` to podręcznikowa powierzchnia specjalizacji.
- **LangGraph** — węzły mogą mieć wyspecjalizowane prompty; krawędzie wymuszają pipeline.
- **AutoGen** — ConversableAgenty specyficzne dla roli z jednoznacznymi nazwami w GroupChat.
- **OpenAI Agents SDK** — narzędzia przekazania między wyspecjalizowanymi Agentami.

## Zbuduj To

`code/main.py` implementuje 4-rolowy pipeline budujący prostą funkcję Pythona:

- **Planista** tworzy specyfikację.
- **Wykonawca** generuje kod jako string.
- **Krytyk** (symulowany LLM) zgłasza oczywiste problemy.
- **Weryfikator** uruchamia wygenerowany kod w sandboksie (`exec`) względem przypadku testowego.

Demo uruchamia się dwukrotnie: raz, gdy wykonawca tworzy poprawny kod (krytyk + weryfikator zaliczają), raz, gdy wykonawca tworzy kod niezgodny ze specyfikacją (krytyk przeocza błąd, bo wygląda wiarygodnie, weryfikator go łapie, bo test nie przechodzi).

Uruchom:

```
python3 code/main.py
```

## Użyj Tego

`outputs/skill-role-designer.md` przyjmuje zadanie i tworzy zestaw ról (3-5 ról), schemat wejścia/wyjścia dla każdej roli oraz kontrolę weryfikatora. Użyj tego przed podłączeniem agentów do frameworka.

## Wdróż To

Lista kontrolna:

- **Co najmniej jeden deterministyczny weryfikator.** Nigdy nie wszystko-LLM.
- **Jawny schemat wejścia/wyjścia dla każdej roli.** Planista zwraca specyfikację, nie prozę; wykonawca czyta ten schemat.
- **Komunikatywna dehalucynacja.** Wykonawca musi zapytać planistę, gdy brakuje informacji; nigdy nie wymyślać.
- **Kolejność krytyk/weryfikator.** Uruchom krytyka najpierw (tani, wychwytuje problemy projektowe), weryfikatora drugiego (wolny, wychwytuje błędy).
- **Budżet pętli.** Maks. 2 rundy rewizji krytyk-wykonawca przed eskalacją do człowieka.

## Ćwiczenia

1. Uruchom `code/main.py` i zaobserwuj, jak weryfikator łapie błąd, który krytyk przeoczył. Dodaj kontrolę analizy statycznej (zliczanie wystąpień `return`) jako dodatkowy weryfikator. Co wychwytuje, czego nie wychwytuje test wykonawczy?
2. Dodaj 5. rolę: "analityk wymagań", który tłumaczy życzenie użytkownika na specyfikację gotową dla planisty. Jakie żądania komunikatywnej dehalucynacji powinny do niego trafiać?
3. Przeczytaj MetaGPT Sekcję 3 ("Agenci"). Wypisz schemat wejścia/wyjścia każdej z 5 ról MetaGPT.
4. Przeczytaj diagram łańcucha czatu ChatDev (arXiv:2307.07924, Rysunek 3). Zidentyfikuj, gdzie komunikatywna dehalucynacja przerywa pętlę, która w przeciwnym razie byłaby nieskończona.
5. 7-krotny wzrost dokładności PwC pochodził z pętli weryfikacyjnych. Hipotetyzuj trzy zadania, w których dodanie weryfikatora nie pomogłoby — gdzie deterministyczne sprawdzenie poprawności jest niemożliwe lub zbyt kosztowne.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|----------------|------------------------|
| Specjalizacja ról | "Różni agenci, różne zadania" | Odrębne prompty systemowe dostrojone do ról planista/wykonawca/krytyk/weryfikator. |
| Wzorzec SOP | "Zakodowana standardowa procedura operacyjna" | Ujęcie MetaGPT: ścisłe schematy wejścia/wyjścia dla każdej roli zamieniają zespół w pipeline. |
| Komunikatywna dehalucynacja | "Zapytaj zanim wymyślisz" | Wzorzec ChatDev: wykonawca pyta planistę, gdy brakuje szczegółu, zamiast go wymyślać. |
| Krytyk | "Recenzent LLM" | Subiektywny, opiniotwórczy recenzent. Wychwytuje problemy stylistyczne. Może być oszukany wiarygodną prozą. |
| Weryfikator | "Deterministyczna kontrola" | Oparta na kodzie kontrola zaliczenia/niezaliczenia. Runner testów, kontroler typów, walidator schematów. Nie można go oszukać. |
| Luka weryfikacyjna | "Nikt nie sprawdził" | 21,3% awarii MAST. Odpowiedź dostarczona bez kontroli, która wychwyciłaby błąd. |
| Pętla rewizji | "Krytyk odsyła z powrotem" | Odrzucenie przez krytyka wyzwala ponowne uruchomienie wykonawcy z informacją zwrotną. Wymaga budżetu. |
| Antywzorzec wszystko-LLM | "Wygląda dobrze" | Każda rola to LLM, brak deterministycznej kontroli. Klasyczna awaria MAST. |

## Dalsza Literatura

- [Hong i in. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — referencyjna publikacja o SOP jako promptach ról
- [Qian i in. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — łańcuch czatu + komunikatywna dehalucynacja
- [Cemri i in. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — taksonomia MAST; luki weryfikacyjne stanowią 21,3% awarii
- [Dokumentacja CrewAI — Role agentów](https://docs.crewai.com/en/introduction) — produkcyjna powierzchnia specyfikacji ról