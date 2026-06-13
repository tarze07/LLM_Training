# Wzorzec Nadzorca / Orkiestrator-Pracownik

> Jeden wiodący agent planuje i deleguje; wyspecjalizowani pracownicy wykonują równolegle w oddzielnych kontekstach i raportują wyniki. To wzorzec stojący za systemem Research od Anthropic (Claude Opus 4 jako lider, Sonnet 4 jako subagenci), mierzony na +90,2% w porównaniu z pojedynczym agentem Opus 4 w wewnętrznych ewaluacjach badawczych. Post inżynieryjny Anthropic donosi, że 80% wariancji w BrowseComp wyjaśnione jest samym zużyciem tokenów — wieloagentowość wygrywa głównie dlatego, że każdy subagent dostaje świeże okno kontekstu. Ta lekcja buduje wzorzec nadzorcy z prymitywów i omawia lekcje inżynieryjne z wdrożeń produkcyjnych w 2026 roku.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Badania są prototypowym zadaniem, na którym zawodzą systemy jednoagentowe. Pytasz „co zmieniło się w systemach wieloagentowych między 2023 a 2026?" Pojedynczy agent czyta pięć artykułów sekwencyjnie, wypełnia połowę swojego kontekstu ich tekstem, a następnie musi rozumować o wszystkich razem. Zapomina pierwszy artykuł, zanim dotrze do piątego. Nie może zrównoleglać.

Wzorzec nadzorcy to naprawia: jeden wiodący agent planuje wyszukiwanie, deleguje każde podpytanie do pracownika i syntezuje. Każdy pracownik dostaje własne 200k-tokenowe okno dla wąskiego pytania. Lider nigdy nie widzi surowych artykułów — tylko podsumowania pracowników.

System Research od Anthropic w produkcji raportuje +90,2% w wewnętrznych ewaluacjach badawczych vs pojedynczy Opus 4. Ten sam post odnotowuje, że 80% wariancji BrowseComp wyjaśnione jest przez *samo zużycie tokenów*. Świeży kontekst na subagenta jest głównym mechanizmem.

## Concept

### Wzorzec

```
                 ┌──────────────┐
                 │   Lider      │ planuje, rozkłada,
                 │  (Opus 4)    │ syntezuje
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Pracown.│  │ Pracown.│  │ Pracown.│
      │    1    │  │    2    │  │    3    │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         świeży       świeży       świeży
         kontekst     kontekst     kontekst
```

Lider nigdy nie czyta surowych materiałów. Pracownicy nigdy nie widzą swojej nawzajem pracy, dopóki lider nie zsyntezuje. Każda strzałka to przekazanie z wąskim artefaktem.

### Dlaczego wygrywa

Trzy mechanizmy:

1. **Świeży kontekst na subagenta.** Pracownik badający „dziedzictwo FIPA-ACL" nie niesie 40k tokenów, które lider wydał na planowanie. Dostaje 200k okno na jedno pytanie.
2. **Specjalizacja przez prompt.** Prompt lidera to „rozkładaj i syntezuj", nie „badaj." Prompt każdego pracownika jest wąski: „znajdź, co zmieniło się w X." Skoncentrowane prompty produkują skoncentrowane wyniki.
3. **Równoległość.** Pracownicy działają współbieżnie. Rzeczywisty czas to mniej więcej `max(czasy_pracowników) + plan + synteza`, a nie `suma(czasy_pracowników)`.

### Lekcje inżynieryjne (Anthropic 2025)

Post Anthropic wymienia kilka lekcji produkcyjnych, które są nadal aktualne w 2026:

- **Dopasuj wysiłek do złożoności zapytania.** Proste zapytania: jeden agent, 3-10 wywołań narzędzi. Złożone zapytania: 10+ agentów. Lider musi to oszacować, nie wywołujący.
- **Szeroko potem wąsko.** Rozkładaj na szerokie podpytania najpierw, potem twórz więcej pracowników na podpytanie, jeśli odpowiedź wymaga głębi.
- **Tęczowe wdrożenia.** Agenci są długożyciowi i stanowi. Tradycyjne blue-green nie działa. Anthropic używa tęczy: stopniowe wdrażanie nowych wersji, podczas gdy stare są opróżniane.
- **Zużycie tokenów dominuje.** Wieloagentowość to ~15× tokenów pojedynczego agenta. Uruchamiaj ją tylko wtedy, gdy wartość zadania uzasadnia koszt.

### Zwrot LangGraph

LangGraph pierwotnie dostarczał bibliotekę `langgraph-supervisor` z wysokopoziomową pomocą `create_supervisor`. W 2025 LangChain przeniósł rekomendację na implementację wzorca nadzorcy przez bezpośrednie wywoływanie narzędzi, ponieważ wywołania narzędzi dają więcej kontroli nad tym, *co widzi nadzorca* (inżynieria kontekstu). Biblioteka wciąż działa; dokumentacja zaleca teraz formę wywołań narzędzi.

### Tryby awarii

- **Lider halucynuje plan.** Jeśli lider generuje podpytania, które nie rozkładają prawdziwego pytania, pracownicy prowadzą precyzyjne badania na złym celu.
- **Pracownicy nadmiernie eksplorują.** Bez wyraźnych granic zakresu, pracownicy dryfują poza przypisane podpytanie i zanieczyszczają krok syntezy.
- **Konflikty syntezy.** Dwóch pracowników zwraca sprzeczne fakty. Lider musi albo zapytać ponownie (dodać rundę), albo wyraźnie odnotować niezgodność. Ciche wybranie jednej strony jest najgorszą awarią: użytkownik nigdy nie wie, że doszło do niezgodności.

### Kiedy nadzorca jest zły

- **Zadania sekwencyjne.** Jeśli krok 2 dosłownie potrzebuje wyniku kroku 1, równoległość nic nie daje. Użyj potoku (CrewAI Sequential, LangGraph graf liniowy).
- **Proste zapytania.** Pojedynczy agent obsługuje je szybciej i taniej. Użyj sprawdzenia „dopasuj wysiłek" lidera przed tworzeniem pracowników.
- **Ścisły determinizm.** Nadzorca używa delegacji wybranej przez LLM. Statyczne grafy są lepsze, gdy audyt/odtworzenie są ważniejsze niż adaptowalność.

```figure
supervisor-hierarchy
```

## Build It

`code/main.py` implementuje nadzorcę trzech równoległych pracowników używając `threading`. Lider rozkłada zapytanie na podpytania, pracownicy działają współbieżnie na każdym podpytaniu, a lider syntezuje. Żadnych prawdziwych LLM — pracownicy są skryptowani, aby symulować fetch-and-summarize.

Kluczowa struktura:

- `Lead.plan(query)` dzieli zapytanie na 3 podpytania.
- `Worker.run(sub_q)` zwraca fałszywe podsumowanie (w produkcji może to być dowolny agent używający narzędzi).
- `Lead.run(query)` uruchamia pracowników w wątkach, łączy i syntezuje.

Uruchom:

```
python3 code/main.py
```

Wynik pokazuje plan, równoległe ślady pracowników ze znacznikami czasu rozpoczęcia/zakończenia i końcową syntezę. Możesz zobaczyć korzyści czasu rzeczywistego: trzech 0,3-sekundowych pracowników działa w ~0,35 sekundy, nie 0,9.

## Use It

`outputs/skill-supervisor-designer.md` przyjmuje zapytanie użytkownika i tworzy projekt wzorca nadzorcy: prompt systemowy lidera, role pracowników, reguły dekompozycji podpytań i szablon syntezy. Użyj go przed zbudowaniem nowego systemu agentowego typu research.

## Ship It

Lista kontrolna przed wdrożeniem wzorca nadzorcy:

- **Dobór modeli.** Lider na modelu klasy rozumującej (Opus class, `o3` class). Pracownicy na szybszym, tańszym modelu (Sonnet, `o4-mini`).
- **Timeout pracownika.** Każdy pracownik przekraczający 2× medianę czasu wykonania jest zabijany; lider albo odtwarza z węższym zakresem, albo kontynuuje bez niego.
- **Limit tokenów na pracownika.** Twardy limit (powiedzmy 10× oczekiwanego wejścia syntezy) zapobiega przekroczeniu budżetu przez niekontrolowanego pracownika.
- **Obserwowalność.** Śledź plan lidera, wywołania narzędzi każdego pracownika i syntezę. To podstawa wszelkiego debugowania post-hoc.
- **Tęczowe wdrożenie.** Stanowi, długożyciowi agenci potrzebują stopniowego przejścia wersji, nie gorącej zamiany.

## Exercises

1. Uruchom `code/main.py`, a następnie zmodyfikuj lidera, aby tworzył 5 pracowników zamiast 3. Zaobserwuj efekt na czasie rzeczywistym. Przy jakiej liczbie pracowników narzut tworzenia przewyższa oszczędności równoległe w tym demo?
2. Zaimplementuj timeout pracownika: zabij każdego pracownika, który działa dłużej niż 0,5 sekundy, i pozwól liderowi zsyntezować pozostałe wyniki. Jakiej obserwowalności potrzebujesz, aby wiedzieć, że pracownik został odcięty?
3. Dodaj krok wykrywania konfliktów do syntezy lidera: jeśli dwóch pracowników zwróci sprzeczne odpowiedzi, lider odnotowuje niezgodność, zamiast wybierać jedną. Jak wykrywasz sprzeczność bez wywoływania LLM?
4. Przeczytaj post inżynieryjny Anthropic o systemie Research. Wymień trzy praktyki, które to zabawkowe demo musiałoby przyjąć, aby działać w produkcji.
5. Porównaj `create_supervisor` LangGraph (starsze) z nową rekomendacją wywoływania narzędzi. Która daje lepszą kontrolę nad tym, co widzi nadzorca? Dlaczego Anthropic jawnie przekazuje tylko pododpowiedzi, a nie surowy kontekst pracownika do syntezy?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Nadzorca | „Agent lider" | Agent orkiestrator, który planuje, deleguje i syntezuje. Nie wykonuje samej pracy. |
| Pracownik | „Subagent" | Skoncentrowany agent wywoływany przez nadzorcę z wąskim zakresem i własnym oknem kontekstu. |
| Orkiestrator-pracownik | „Wzorzec nadzorcy" | To samo, inna nazwa. Literatura 2026 używa obu. |
| Świeży kontekst | „Czyste okno" | Kontekst pracownika zaczyna się od jego promptu systemowego i przypisanego pytania, nie od historii lidera. |
| Tęczowe wdrożenie | „Stopniowe wdrażanie" | Długożyciowi, stanowi agenci potrzebują wersjonowanego opróżniania i zastępowania, nie blue-green. |
| Dominacja tokenów | „Kontekst jest zmienną" | 80% wariancji ewaluacji badawczych pochodzi z całkowitej liczby użytych tokenów, nie wyboru modelu, według Anthropic. |
| Dopasuj wysiłek | „Dopasuj liczbę agentów do złożoności" | Lider szacuje trudność zapytania, tworzy odpowiednio 1 vs 10+ pracowników. |
| Konflikt syntezy | „Pracownicy się nie zgadzają" | Dwóch pracowników zwraca sprzeczne fakty; lider musi ujawnić niezgodność, nie cicho wybrać jednej. |

## Further Reading

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — produkcyjna referencja dla wzorca nadzorcy
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — nadzorca przez wywoływanie narzędzi jest teraz zalecaną formą
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — starsze narzędzie, wciąż używane w produkcji 2026
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — wariant nadzorcy oparty na przekazaniu