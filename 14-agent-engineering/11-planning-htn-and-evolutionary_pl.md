# Planowanie z HTN i wyszukiwaniem ewolucyjnym

> Planowanie symboliczne obsługuje przypadki, w których plan jest możliwy do udowodnienia jako poprawny. Ewolucyjne wyszukiwanie kodu obsługuje przypadki, w których funkcja dopasowania jest sprawdzalna maszynowo. ChatHTN (2025) i AlphaEvolve (2025) pokazują, co każde z nich odblokowuje w połączeniu z LLM.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnij Hierarchiczne Sieci Zadań: zadania, metody, operatory, warunki wstępne, efekty.
- Opisz hybrydową pętlę ChatHTN — wyszukiwanie symboliczne z dekompozycją awaryjną LLM.
- Wyjaśnij pętlę ewolucyjną AlphaEvolve i dlaczego działa ona tylko z programowalnym ewaluatorem.
- Zaimplementuj zabawkowy planista HTN plus zabawkowe wyszukiwanie ewolucyjne w stdlib.

## The Problem

ReWOO (Lekcja 02), Plan-and-Execute i ReAct pokrywają większość planowania agentów. Dwa przypadki, których nie obejmują dobrze:

1. **Plany z możliwą do udowodnienia poprawnością.** Harmonogramowanie, planowanie lotów, przepływy pracy zgodności — plan musi być poprawny z założenia. Płynny plan LLM, który czasami halucynuje krok, jest nie do przyjęcia.
2. **Optymalizacje ze sprawdzalną maszynowo funkcją dopasowania.** Mnożenie macierzy, heurystyki harmonogramowania, przebiegi kompilatora — celem nie jest "poprawny plan", ale "najlepszy plan."

Planowanie HTN i AlphaEvolve rozwiązują te dwa różne problemy. Oba używają LLM jako wzmacniaczy, a nie zamienników.

## The Concept

### Hierarchiczne Sieci Zadań

HTN to:

- **Zadania** — złożone (do zdekomponowania) i prymitywne (bezpośrednio wykonywalne).
- **Metody** — sposoby dekompozycji zadania złożonego na podzadania, z warunkami wstępnymi.
- **Operatory** — akcje prymitywne z warunkami wstępnymi i efektami.
- **Stan** — zbiór faktów.

Planowanie: mając zadanie docelowe i stan początkowy, znajdź dekompozycję na operatory prymitywne, których warunki wstępne są spełnione sekwencyjnie.

HTN jest starsze niż LLM i wciąż stanowi punkt odniesienia dla planów o możliwej do udowodnienia poprawności.

### ChatHTN (Gopalakrishnan i in., 2025)

ChatHTN (arXiv:2505.11814) przeplata symboliczne HTN z zapytaniami LLM:

1. Spróbuj zdekomponować bieżące zadanie złożone za pomocą istniejących metod.
2. Jeśli żadna metoda nie ma zastosowania, zapytaj LLM: "jak zdekomponowałbyś `task` w stanie `s`?"
3. Przetłumacz odpowiedź LLM na kandydackie podzadania.
4. Zweryfikuj względem schematu operatora; odrzuć nieprawidłowe dekompozycje.
5. Rekurencja.

Główne twierdzenie artykułu: każdy wyprodukowany plan jest możliwy do udowodnienia jako poprawny, ponieważ sugestie LLM wchodzą tylko jako kandydackie dekompozycje, nigdy jako bezpośrednie edycje planu. Warstwa symboliczna odpowiada za poprawność; LLM rozszerza bibliotekę metod.

Uczenie metod online (OpenReview `gwYEDY9j2x`, kontynuacja 2025) dodaje uczący się, który uogólnia dekompozycje wyprodukowane przez LLM przez regresję — zmniejszając częstotliwość zapytań LLM nawet o 75%.

### AlphaEvolve (Novikov i in., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, czerwiec 2025) to inna bestia: ewolucyjne wyszukiwanie kodu orkiestrowane przez ensemble Gemini 2.0 Flash/Pro.

Pętla:

1. Zacznij od programu początkowego + programowalnego ewaluatora (zwraca wynik dopasowania).
2. Ensemble LLM proponuje mutacje.
3. Uruchom mutacje przez ewaluator.
4. Zachowaj najlepszą; mutuj ponownie.

Opublikowane osiągnięcia:

- Pierwsza poprawa względem Strassena dla mnożenia macierzy zespolonych 4x4 od 56 lat (48 mnożeń skalarnych).
- 0.7% odzyskanej mocy obliczeniowej Google poprzez heurystykę harmonogramowania Borg.
- 32% przyspieszenia FlashAttention na obciążeniu granicznym.

Twarde ograniczenie: funkcja dopasowania musi być sprawdzalna maszynowo. Wyszukiwanie ewolucyjne po odpowiedziach prozą nie jest zbieżne.

### Kiedy użyć którego

| Problem class | Use | Why |
|---------------|-----|-----|
| Harmonogramowanie z twardymi ograniczeniami | HTN + ChatHTN | Możliwa do udowodnienia poprawność |
| Optymalizacja kompilatora | AlphaEvolve | Sprawdzalna maszynowo funkcja dopasowania |
| Wykonanie wieloetapowego zadania | ReAct / ReWOO | LLM w pętli, bez formalnych gwarancji |
| Poprawa kodu z testami | AlphaEvolve | Testy są ewaluatorem |
| Automatyzacja związana polityką | HTN | Warunki wstępne kodują politykę |

### Gdzie ten wzór zawodzi

- **HTN bez operatorów.** Bez schematów warunków wstępnych/efektów twierdzenie o poprawności upada. "LLM sugeruje dekompozycję" w ChatHTN wymaga schematu do odrzucania nieprawidłowych ruchów.
- **AlphaEvolve bez prawdziwego ewaluatora.** "Zapytaj LLM, czy kod jest lepszy" to nie funkcja dopasowania. Ewaluator musi być deterministyczny i szybki.
- **Przeinżynierowanie.** Większość zadań agentów nie potrzebuje żadnego z nich. Sięgaj najpierw po ReAct lub ReWOO.

## Build It

`code/main.py` implementuje dwie zabawki:

- Planista HTN w stdlib z operatorami, metodami, warunkami wstępnymi, efektami i `LLMFallback`, który włącza się, gdy żadna metoda nie pasuje do zadania złożonego. "LLM" to skryptowany dekompozytor, więc planista działa offline.
- Wyszukiwanie ewolucyjne w stdlib po programach arytmetycznych: rozwijaj wyrażenia, których wynik minimalizuje `|f(x) - target|` na zbiorze testowym. Ewaluator jest deterministyczny.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje planistę HTN dekomponującego zadanie złożone (z awaryjnym LLM w trakcie planu) i pętlę ewolucyjną zbiegającą się do docelowego wyrażenia.

## Use It

- **Planisty HTN** — `pyhop`, `SHOP3` lub zbuduj własny dla egzekwowania polityki specyficznej dla domeny.
- **ChatHTN** — kod badawczy; wzór (symboliczny + awaryjny LLM) przenosi się czysto na dowolnego planistę HTN.
- **AlphaEvolve** — artykuł DeepMind; wzór (ensemble + ewaluator) jest powtarzalny. Pojawiają się OpenEvolve i podobne rozwidlenia open-source.
- **Frameworki agentów** — żaden nie dostarcza gotowego HTN ani AlphaEvolve. Zbuduj to jako podagenta lub pracownika tła.

## Ship It

`outputs/skill-hybrid-planner.md` generuje szkielet hybrydowego planisty (HTN lub ewolucyjnego) z jawnie określoną rolą LLM.

## Exercises

1. Rozszerz planistę HTN o wycofywanie: gdy warunek końcowy operatora nie powiedzie się w czasie wykonania, wycofaj się i spróbuj następnej metody.
2. Dodaj pamięć podręczną metod LLM do ChatHTN: gdy LLM dekomponuje zadanie `T` we wzorcu stanu `P`, zapisz wynik. Przy następnym wywołaniu najpierw sprawdź bibliotekę metod.
3. Zamień ewaluator wyszukiwania ewolucyjnego na prawdziwy zestaw testów. Wyewoluuj funkcję sortowania, która przechodzi 20 przypadków testowych; zgłoś liczbę pokoleń do zbieżności.
4. Przeczytaj notatki projektowe ewaluatora AlphaEvolve. Zaprojektuj ewaluator dla domeny, która Cię interesuje (optymalizacja zapytań SQL, minimalizacja zestawu testów, YAML wdrożeniowy).
5. Połącz: użyj HTN do dekompozycji złożonego zadania na podzadania, a następnie użyj wyszukiwania ewolucyjnego na operatorze prymitywnym każdego podzadania. Gdzie to błyszczy, gdzie to przeinżynieria?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| HTN | "Planista hierarchiczny" | Dekompozycja zadań z operatorami, warunkami wstępnymi, efektami |
| Metoda | "Reguła dekompozycji" | Sposób podziału zadania złożonego na podzadania |
| Operator | "Akcja prymitywna" | Konkretny krok z warunkiem wstępnym i efektem |
| ChatHTN | "LLM + HTN" | Planista symboliczny pyta LLM, gdy żadna metoda nie pasuje |
| AlphaEvolve | "Ewolucyjne wyszukiwanie kodu" | Ensemble LLM mutuje kod; deterministyczny ewaluator wybiera |
| Funkcja dopasowania | "Ewaluator" | Deterministyczny, sprawdzalny maszynowo wynik na wyjściach |
| Uczenie metod online | "Zbuforowana dekompozycja LLM" | Przechowuj + uogólniaj plany LLM, aby obniżyć koszt zapytań |

## Further Reading

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — hybrydowy planista symboliczny + LLM
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — ewolucyjne wyszukiwanie kodu z mutacjami LLM
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — kiedy sięgnąć po planistę, a kiedy po prostą pętlę