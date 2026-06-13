# CrewAI: Zespoły oparte na rolach i przepływy

> CrewAI to wieloagentowy framework oparty na rolach w 2026 roku. Cztery prymitywy: Agent, Task, Crew, Process. Dwa kształty najwyższego poziomu: Crews (autonomiczna, oparta na rolach współpraca) i Flows (sterowane zdarzeniami, deterministyczne). Dokumentacja jest szczera: "dla każdej aplikacji gotowej do produkcji zacznij od Flow."

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Learning Objectives

- Wymień cztery prymitywy CrewAI (Agent, Task, Crew, Process) i co każdy z nich posiada.
- Rozróżnij Sequential, Hierarchical i planowany Consensus; wybierz jeden na obciążenie.
- Rozróżnij Crews (autonomiczne oparte na rolach) od Flows (sterowane zdarzeniami deterministyczne) i wyjaśnij zalecenie produkcyjne dokumentacji.
- Podłącz narzędzia za pomocą dekoratora `@tool` i podklasy `BaseTool`; uzasadnij wybór między strukturalnymi wyjściami a wolnym tekstem.
- Wymień cztery typy pamięci CrewAI i kiedy każdy się opłaca.
- Zaimplementuj w stdlib trzyosobowy zespół (badacz, pisarz, redaktor) produkujący brief.
- Zidentyfikuj trzy tryby awarii CrewAI: rozdęcie promptu, podatek LLM menedżera, kruche przekazania.

## The Problem

Zespoły adoptujące wieloagentowe frameworki uderzają w tę samą ścianę. "Autonomiczna współpraca" brzmi świetnie w demo. Potem klient zgłasza błąd i potrzebujesz deterministycznego odtworzenia. Albo finanse pytają, ile kosztuje uruchomienie zespołu routowanego przez LLM. Albo dyżurny musi wiedzieć, który agent utknął o 3 nad ranem.

Swobodne zespoły routowane przez LLM nie odpowiadają czysto na żadne z tych pytań. Czyste DAG-i odpowiadają na wszystkie, ale tracą eksploracyjny kształt, którego potrzebuje agent burzy mózgów.

Podział CrewAI jest uczciwy co do kompromisu. Crews do pracy opartej na rolach, eksploracyjnej, opartej na współpracy. Flows do sterowanej zdarzeniami, należącej do kodu, audytowalnej produkcji. Ten sam framework, dwa kształty, wybierz na powierzchnię.

## The Concept

### Cztery prymitywy

Powierzchnia CrewAI jest mała. Zapamiętaj to, a reszta to konfiguracja.

- **Agent.** `role + goal + backstory + tools + (opcjonalnie) llm`. Backstory jest nośne. Kształtuje ton, osąd, kiedy agent się zatrzymuje. Narzędzia to funkcje, które agent może wywołać (więcej poniżej).
- **Task.** `description + expected_output + agent + (opcjonalnie) context + (opcjonalnie) output_pydantic`. Wielokrotnego użytku jednostka pracy. `expected_output` to kontrakt. `context` wymienia nadrzędne zadania, których wyniki są przekazywane. `output_pydantic` wymusza strukturalny kształt.
- **Crew.** Kontener. Posiada listę `agents`, listę `tasks`, `process` oraz opcjonalne ustawienia `memory` + `verbose` + `manager_llm`.
- **Process.** Strategia wykonania. Sequential, Hierarchical, Consensus (planowany). Wybiera kształt uruchomienia.

Agenci nie widzą się bezpośrednio. Zadania odnoszą się do agentów. Crew sekwencjonuje zadania. Process decyduje, kto wybiera następne zadanie. To cały model mentalny.

> **Zweryfikowano względem** CrewAI 0.86 (2026-05). Nowsze wersje mogą zmienić nazwy lub scalić typy procesów; sprawdź [dokumentację procesów CrewAI](https://docs.crewai.com/concepts/processes) przed poleganiem na konkretnym kształcie.

### Sequential vs Hierarchical vs Consensus

- **Sequential.** Zadania są uruchamiane w kolejności deklaracji. Wynik zadania N jest dostępny jako `context` dla zadania N+1. Najniższy koszt. Najbardziej przewidywalny. Używaj, gdy kolejność jest ustalona.
- **Hierarchical.** Menedżer Agent (oddzielne wywołanie LLM) routuje między specjalistami. CrewAI tworzy menedżera z twojej konfiguracji `manager_llm` lub domyślnej. Menedżer wybiera następne zadanie w każdej rundzie i może odmówić lub przekierować. Używaj, gdy masz czterech lub więcej specjalistów, a kolejność naprawdę zależy od poprzedniego wyniku.
- **Consensus.** Planowany, obecnie niezaimplementowany w publicznym API. Dokumentacja rezerwuje nazwę dla przyszłego procesu opartego na głosowaniu. Nie polegaj na nim dzisiaj.

Hierarchical dodaje wywołanie LLM na rundę (menedżer) na wierzchu każdego wywołania specjalisty. Koszt tokenów może się potroić przy pięcioetapowym uruchomieniu. Płać za to tylko wtedy, gdy potrzebujesz routingu.

### Crews vs Flows

To jest rama, z którą dokumentacja prowadzi w 2026.

- **Crew.** Autonomia sterowana LLM. Framework wybiera kształt w czasie wykonania. Dobre do: badań, burzy mózgów, pierwszych szkiców, wszędzie tam, gdzie ścieżka jest częścią odpowiedzi. Trudne do odtworzenia. Trudne do testowania. Tanie do prototypowania.
- **Flow.** Graf sterowany zdarzeniami, który posiadasz. `@start` oznacza wejście. `@listen(topic)` oznacza krok, który odpala, gdy inny krok emituje ten temat. Każdy krok to czysty Python (może wewnętrznie wywołać Crew). Dobre do: produkcji. Obserwowalne. Testowalne. Deterministyczne.

Zalecenie produkcyjne dokumentacji 2026: zacznij od Flow. Włączaj Crews jako wywołania `Crew.kickoff()` z wewnątrz kroków Flow, gdy autonomia zapracuje na swój koszt. Flow daje ślad audytu, Crew daje eksplorację. Komponuj, nie wybieraj.

### Integracja narzędzi

Trzy sposoby na danie Agentowi narzędzia. Wybierz najprostszy, który pasuje.

1. **Dekorator `@tool`.** Czyste funkcje stają się narzędziami. Sygnatura jest schematem; docstring jest opisem, który widzi LLM. Najlepsze do jednorazowych pomocników.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **Podklasa `BaseTool`.** Narzędzie oparte na klasie z jawnym schematem argumentów, wsparciem async, ponownymi próbami. Używaj, gdy narzędzie ma stan (klient, pamięć podręczna) lub potrzebuje strukturalnych argumentów.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Wbudowane zestawy narzędzi.** CrewAI dostarcza adaptery pierwszej strony: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. Podłączone jednym importem.

Strukturalne wyjścia używają Pydantic. Przekaż `output_pydantic=MyModel` na Task. CrewAI waliduje odpowiedź LLM względem modelu i albo wymusza, albo ponawia. Połącz to z wąskim stringiem `expected_output`. Wyjścia wolnym tekstem są w porządku do szkiców; strukturalne wyjścia są tym, co downstream Flows mogą konsumować.

### Haczyki pamięci

CrewAI dostarcza cztery typy pamięci gotowe do użycia. Komponują się: Crew może włączyć wszystkie cztery jednocześnie.

> **Zweryfikowano względem** CrewAI 0.86 (2026-05). Najnowsze wydania routują wszystko przez ujednolicony system `Memory`, który opakowuje te cztery magazyny. Model koncepcyjny poniżej wciąż obowiązuje, ale publiczna powierzchnia klas może się zwinąć do pojedynczego punktu wejścia `Memory` w nowszych wersjach; sprawdź [dokumentację pamięci CrewAI](https://docs.crewai.com/concepts/memory) dla bieżącego API.

- **Krótkoterminowa.** Bufor rozmowy w ramach jednego uruchomienia. Wyczyszczony na koniec.
- **Długoterminowa.** Utrwalana między uruchomieniami. Przechowywana w bazie wektorowej (Chroma domyślnie, wymienna). Wyszukiwana przez podobieństwo do bieżącego zadania.
- **Encji.** Fakty na encję. "Klient X jest na planie enterprise." Kluczowane po encji, nie przez podobieństwo. Utrwalane między uruchomieniami.
- **Kontekstowa.** Wyszukiwanie w czasie składania. Pobiera odpowiednią pamięć w momencie, gdy Agent jej potrzebuje, nie wstępnie załadowaną.

Włącz na Crew przez `memory=True` lub konfigurację per-typ. Wspierane przez dostawcę embeddingów, którego konfigurujesz (domyślnie OpenAI, wymienny na lokalny). Pamięć jest jednym z miejsc, gdzie CrewAI zarabia na swoją przewagę nad cieńszymi frameworkami; czysty LangGraph wymaga ręcznego podłączenia każdego z tych elementów.

### Kiedy CrewAI pasuje

- Trzech do sześciu agentów z nazwanymi rolami i przepływem pracy opartym na współpracy. Szkicowanie, przeglądanie, planowanie, burza mózgów.
- Routing, gdzie osąd LLM co do następnego kroku jest częścią wartości (Hierarchical).
- Wszędzie tam, gdzie zespół jest szczęśliwszy czytając `role + goal + backstory` niż czytając definicję grafu.

### Kiedy CrewAI nie pasuje

- Deterministyczne DAG-i ze ścisłym porządkowaniem. Użyj LangGraph (Lekcja 13). Kształt grafu jest właściwą abstrakcją; ramowanie ról CrewAI jest tarciem.
- Budżety opóźnień poniżej sekundy. Hierarchical dodaje rundy. Nawet Sequential serializuje prompty, które zawierają backstory i poprzednie wyniki.
- Pętle jednoagentowe. Pomiń framework; pętla agenta (Lekcja 1) plus rejestr narzędzi jest krótsza.

Lekcja 17 (Kompromisy frameworków agentów) przedstawia to w macierzy. Krótka wersja: CrewAI siedzi w rogu "oparta na rolach współpraca".

### Kształt zależności

Niezależny od LangChain. Python 3.10 do 3.13. Używa `uv`. Liczba gwiazdek: zobacz [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) (stan na 2026-05). Integracja z AWS Bedrock jest udokumentowana; raporty dostawców wskazują znaczące przyspieszenie vs LangGraph w obciążeniach QA, ale metodologia (zbiór danych, sprzęt, metryka ewaluacji) nie jest opublikowana, więc traktuj liczby od dostawców frameworków jako kierunkowe.

### Gdzie ten wzór zawodzi

- **Rozdęcie promptu z backstory.** 2000-słowne backstory na agenta i pięcioosobowy zespół spala budżet kontekstu przed pierwszym wywołaniem narzędzia. Utrzymuj backstory poniżej 200 słów. Używaj fraz wielokrotnie między agentami; nie powtarzaj stylu domowego pięć razy.
- **Podatek tokenowy menedżera LLM.** Proces Hierarchical dodaje wywołanie menedżera LLM przed każdym wywołaniem specjalisty. W pięciozadaniowym zespole to sześć wywołań LLM zamiast pięciu, a wywołanie menedżera niesie pełną listę zadań plus poprzednie wyniki. Przełącz na Sequential, chyba że routing zależy od wyniku.
- **Kruche przekazania.** `expected_output` zadania N to "zarys". Zadanie N+1 czyta go jako `context` i próbuje sparsować trzy sekcje. LLM wyprodukował cztery. Downstream Agent improwizuje. Napraw przez `output_pydantic` na zadaniu N, aby zadanie N+1 czytało typowany obiekt, a nie wolny tekst.
- **Crew jako prod.** Swobodny Crew wysłany do produkcji bez opakowania Flow. Zmienność wyników jest wysoka; odtworzenie jest niemożliwe; dyżurny nie może porównać złego uruchomienia z dobrym. Opakuj w Flow.

## Build It

`code/main.py` implementuje w stdlib wersje obu kształtów plus trzyosobowy zespół.

Kształt:

- Dataclasses `Agent`, `Task` odpowiadające powierzchni CrewAI.
- `SequentialCrew.kickoff(inputs)` uruchamia zadania w kolejności deklaracji, przekazując wyniki jako `context`.
- `HierarchicalCrew.kickoff(topic)` dodaje menedżera Agent wybierającego następnego specjalistę w każdej rundzie, zatrzymuje się na "done".
- `Flow` z dekoratorami `@start` i `@listen(topic)`, małą pętlą zdarzeń i śladem.
- Dekorator `tool(name)` odwzorowujący kształt `@tool` CrewAI.
- `Memory` z magazynami `short_term`, `long_term`, `entity`; symulowane podobieństwo używa numpy.
- Mockowane odpowiedzi LLM to zakodowane na stałe stringi kluczowane po roli plus prefiks wejścia. Brak sieci. Deterministyczne.

Konkretne demo: zespół badacz, pisarz, redaktor produkujący brief na temat "agent engineering 2026". Badacz pobiera (zamockowane) źródła. Pisarz szkicuje. Redaktor poprawia. Ten sam zespół uruchomiony przez Flow, aby pokazać deterministyczny kształt.

Uruchom:

```bash
python3 code/main.py
```

Ślad obejmuje: sequential crew przekazujący wyniki przez `context`, hierarchical crew z wyborami menedżera (badacz, pisarz, redaktor, potem "done"), flow uruchamiający te same trzy kroki z jawnymi tematami (`researched`, `drafted`, `edited`), wywołania narzędzi routowane przez `@tool` i pamięć długoterminową utrwaloną między dwoma kickoffami.

Ślad Crew jest płynny; menedżer mógłby w zasadzie zmienić kolejność. Ślad Flow jest ustalony. Ten wybór jest lekcją.

## Use It

- **CrewAI Flow** do produkcji. Nawet gdy Flow to jeden krok, który wywołuje `Crew.kickoff()`. Flow daje granicę audytu.
- **CrewAI Crew (Sequential)** do pracy opartej na współpracy z jasnym porządkowaniem, zwłaszcza pierwsze szkice i pętle przeglądowe.
- **CrewAI Crew (Hierarchical)** gdy routing zależy od wyniku i masz czterech lub więcej specjalistów.
- **LangGraph** (Lekcja 13) do jawnych maszyn stanów, trwałego wznawiania, ścisłego porządkowania.
- **AutoGen v0.4** (Lekcja 14) do współbieżności modelu aktora i izolacji błędów.
- **OpenAI Agents SDK** (Lekcja 16) do produktów OpenAI-first z przekazaniami i zabezpieczeniami.
- **Claude Agent SDK** (Lekcja 17) do produktów Claude-first z podagentami i magazynem sesji.

## Ship It

`outputs/skill-crew-or-flow.md` wybiera Crew vs Flow dla zadania i tworzy szkielet minimalnej implementacji. Twarde odrzucenia dla Crew-bez-backstory, Flow-bez-jawnych-tematów, Hierarchical z mniej niż trzema specjalistami.

## Pitfalls

- **Backstory jako smak.** Kształtuje wyniki. Przetestuj trzy warianty na agenta; wariancja jest realna. Wybierz jeden, zamroź go.
- **Pomijanie `expected_output`.** Bez kontraktu na zadanie, downstream zadania przechwytują cokolwiek LLM wyprodukował. Crew działa; audyt zawodzi.
- **Pamięć zawsze włączona.** Długoterminowe zapisy przy każdym uruchomieniu. Baza wektorowa rośnie. Wyszukiwanie staje się głośne. Ogranicz zapisy do zadań, gdzie fakt jest trwały.
- **Dryf promptu menedżera.** Prompt menedżera w Hierarchical jest domyślny. Jeśli routing robi się dziwny, zrzuć go w trybie verbose i przeczytaj.
- **Efekty uboczne narzędzi w Crews.** Crew może wywołać narzędzie więcej razy niż oczekiwano. POST, DELETE, płatności należą do kroku Flow, nigdy do narzędzia Crew.

## Exercises

1. Przekształć Sequential crew na Flow. Policz punkty styku, gdzie zmienność spada. Zanotuj, gdzie czytelność spadła.
2. Dodaj pamięć encji do zespołu: fakty o kliencie utrwalają się między kickoffami. Zweryfikuj, że wyszukiwanie pobiera właściwą encję.
3. Zaimplementuj proces Hierarchical, w którym menedżer odmawia routowania do redaktora, dopóki wynik pisarza nie ma co najmniej trzech akapitów. Prześledź ponowienie.
4. Podłącz podklasę `BaseTool` dla (zamockowanego) wyszukiwania w sieci. Porównaj kształt śladu vs wersja z dekoratorem `@tool`.
5. Dodaj `output_pydantic=Brief` do zadania redaktora, gdzie `Brief` ma `title`, `summary`, `sections`. Spraw, aby zadanie pisarza raz wyprodukowało nieprawidłowy JSON; zweryfikuj zachowanie ponawiania CrewAI w śladzie.
6. Przeczytaj wstęp do dokumentacji CrewAI. Przenieś zabawkę na prawdziwe API `crewai`. Których gwarancji brakowało w wersji stdlib?
7. Podłącz AgentOps lub Langfuse (Lekcja 24) do prawdziwego uruchomienia. Których śladów brakowało w wersji stdlib?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Persona" | Rola + cel + backstory + narzędzia |
| Task | "Jednostka pracy" | Opis + oczekiwany wynik + przypisany + opcjonalne strukturalne wyjście |
| Crew | "Zespół agentów" | Kontener dla Agentów + Zadań + Procesu |
| Process | "Strategia wykonania" | Sequential / Hierarchical / Consensus (planowany) |
| Flow | "Deterministyczny przepływ pracy" | Sterowany zdarzeniami, należący do kodu, testowalny |
| Backstory | "Prompt persony" | Kształtujący ton i osąd dla Agenta |
| `@tool` | "Narzędzie funkcyjne" | Dekorator zamieniający funkcję w narzędzie, które Agent może wywołać |
| `BaseTool` | "Narzędzie klasowe" | Narzędzie oparte na klasie ze schematem argumentów, ponownymi próbami, wsparciem async |
| Pamięć encji | "Fakty na encję" | Pamięć ograniczona do klienta / konta / zgłoszenia |
| Pamięć długoterminowa | "Pamięć między uruchomieniami" | Pamięć wspierana wektorowo, która utrwala się między kickoffami |
| Pamięć kontekstowa | "Wyszukiwanie na żądanie" | Pamięć pobierana w momencie, gdy Agent jej potrzebuje |
| Menedżer LLM | "Agent router" | Dodatkowy LLM w procesie Hierarchical, który wybiera następne zadanie |
| `expected_output` | "Kontrakt zadania" | String mówiący Agentowi (i audytowi), jaki kształt zwrócić |

## Further Reading

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): koncepcje i zalecana ścieżka produkcyjna
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): kształt sterowany zdarzeniami, `@start`, `@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, wbudowane zestawy narzędzi
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): krótkoterminowa, długoterminowa, encji, kontekstowa
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): kiedy multi-agent pomaga, a kiedy nie
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): alternatywa maszyny stanów