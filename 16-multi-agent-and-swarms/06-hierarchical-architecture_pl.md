# Architektura Hierarchiczna i Jej Tryb Awarii

> Hierarchia to zagnieżdżony nadzorca. Menedżerowie agentów nad pod-menedżerami nad pracownikami. `Process.hierarchical` w CrewAI jest podręcznikową wersją: `manager_llm` dynamicznie deleguje zadania i waliduje wyniki. Odpowiednikiem w LangGraph jest `create_supervisor(create_supervisor(...))`. Jest to naturalny wzorzec, gdy zadanie jest prawdziwą strukturą organizacyjną. Jest to również wzorzec najbardziej podatny na zapadnięcie się w menedżerskie zapętlenie — menedżerowie agentów źle przydzielają pracę, błędnie interpretują wyniki podległych agentów lub nie osiągają konsensusu. Sekwencyjne często go pokonuje.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern)
**Time:** ~60 minutes

## Problem

Gdy wzorzec nadzorcy już zaskoczy, naturalnym następnym krokiem jest „co jeśli pracownicy sami są nadzorcami?" Zespoły mają podzespoły; firmy mają departamenty departamentów. Architektury hierarchiczne to odzwierciedlają.

Problem: Menedżerowie LLM nie są tacy sami jak menedżerowie ludzie. Ludzki menedżer ma stabilne założenia wstępne o tym, co wiedzą jego podwładni. Menedżer LLM przemyśla na nowo organizację każdej tury na podstawie tego, co jest w jego kontekście. Niewielki dryf w tym kontekście i całe drzewo błędnie alokuje pracę.

## Concept

### Kształt

```
                 Menedżer
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Pod-Mgr A         Pod-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       P1  P2  P3         P4  P5
```

Każdy węzeł wewnętrzny planuje, deleguje i syntezuje. Tylko liście wykonują pracę.

### Gdzie się sprawdza

- **Jasne mapowanie organizacji.** Jeśli prawdziwe zadanie jest departamentalne („dział prawny przejrzyj dokument, dział finansowy przejrzyj dokument, dział inżynieryjny przejrzyj dokument, a następnie podsumuj dla zarządu"), hierarchia jest jawna.
- **Lokalne podsumowywanie.** Każdy pod-menedżer syntezuje wyniki swojego zespołu, zanim zobaczy je główny menedżer. Główny menedżer widzi trzy podsumowania pod-menedżerów, a nie piętnaście wyników pracowników.

### Gdzie się psuje

Trzy tryby awarii, które analizy powypadkowe z 2026 roku wciąż znajdują:

1. **Błąd przydziału zadań.** Menedżer czyta cel, halucynuje dekompozycję i deleguje do niewłaściwego pod-menedżera. Ponieważ pod-menedżer posłusznie pracuje nad tym, co mu dano, błąd wychodzi na jaw dopiero przy syntezie na szczycie — jeden poziom od miejsca, gdzie człowiek mógłby go złapać.
2. **Błędna interpretacja wyniku.** Pod-menedżer zwraca „nie można zweryfikować twierdzenia X." Główny menedżer podsumowuje jako „twierdzenie X niepotwierdzone." Znaczenie dryfuje na każdym poziomie.
3. **Pętle konsensusu.** Dwóch pod-menedżerów nie zgadza się; główny menedżer prosi ich o pogodzenie; oni delegują w dół; pracownicy uruchamiają się ponownie; pod-menedżerowie zwracają nieco inne odpowiedzi; pętla. `Process.hierarchical` w CrewAI chroni przed tym limitami kroków, ale sam limit jest teraz hiperparametrem.

### Decydujące pytanie

Sekwencyjne (liniowy potok) vs hierarchiczne: czy twoje zadanie faktycznie ma niezależne podzespoły, czy to jeden liniowy przepływ udający drzewo? Jeśli to drugie, użyj sekwencyjnego. Jeśli to pierwsze, użyj hierarchicznego, ale zaplanuj jawne reguły pojednania.

### Implementacja w CrewAI

`Process.hierarchical` łączy menedżera LLM nad wyspecjalizowanymi zespołami. Menedżer:

- otrzymuje zadanie najwyższego poziomu,
- przydziela podzadania zespołom,
- ocenia wyniki zespołów,
- decyduje, czy zaakceptować, ponownie delegować, czy iterować.

Dokumentacja: https://docs.crewai.com/en/introduction (szukaj „Hierarchical Process" w Core Concepts).

### Implementacja w LangGraph

LangGraph używa zagnieżdżonych wywołań `create_supervisor`. Wewnętrzny nadzorca ma własny graf; zewnętrzny nadzorca traktuje wewnętrzny graf jako nieprzezroczysty węzeł. Jest to czystsze niż CrewAI do debugowania (możesz przechodzić przez każdy graf osobno), ale trudniejsze do wyrażenia dynamicznego przekształcania drzewa.

Referencja: https://reference.langchain.com/python/langgraph-supervisor.

## Build It

`code/main.py` uruchamia 3-poziomową hierarchię:

- główny menedżer: dzieli zadanie na gałęzie „inżynieryjną" i „prawną",
- pod-menedżer inżynierii: dzieli na pracowników „frontend" i „backend",
- pod-menedżer prawny: jeden pracownik.

Demo kontrastuje ścieżkę szczęśliwą (wszyscy się zgadzają) ze **ścieżką zaburzoną**, w której dekompozycja głównego menedżera błędnie oznacza „prawny" jako „finansowy" i obserwuje kaskadę błędów — pod-menedżer posłusznie wykonuje pracę finansową, główny syntezator raportuje wyniki finansowe, a oryginalne pytanie prawne pozostaje bez odpowiedzi.

Uruchom:

```
python3 code/main.py
```

Wynik pokazuje obie ścieżki z wyraźnym porównaniem „o co pytano" vs „co dostarczono."

## Use It

`outputs/skill-hierarchy-fitness.md` ocenia, czy dane zadanie powinno używać hierarchicznego, sekwencyjnego czy płaskiego nadzorcy. Wejścia: opis zadania, struktura organizacji, budżet pojednania. Wynik: rekomendacja wzorca z konkretnymi trybami awarii, przed którymi należy się chronić.

## Ship It

Jeśli wdrażasz hierarchię:

- **Ogranicz głębokość drzewa do 2.** Trzy poziomy już ukrywają większość błędów przed obserwowalnością.
- **Jawny budżet pojednania.** Ustaw maksymalną liczbę rund, zanim główny menedżer musi się zdecydować. Zwykle 2.
- **Proweniencja na każdej syntezie.** Podsumowanie każdego węzła musi wskazywać, które wyniki liści je wyprodukowały.
- **Alert na dryf dekompozycji.** Rejestruj dekompozycję menedżera na każdym kroku; porównaj z zapytaniem użytkownika. Jeśli dekompozycja nie obejmuje już zapytania, uruchom alert.

## Exercises

1. Uruchom `code/main.py` i porównaj ścieżkę szczęśliwą z zaburzoną. Ile poziomów przekazania menedżera potrzeba, zanim końcowy wynik całkowicie odbiegnie od pytania użytkownika?
2. Dodaj trzeci poziom (główny → pod → pod-pod → pracownik). Zmierz, jak często ścieżka zaburzona koryguje się sama vs całkowicie odbiega wraz ze wzrostem głębokości.
3. Zaimplementuj pracownika „kanarka" przy każdym pod-menedżerze, któremu zawsze zadawane jest oryginalne pytanie użytkownika bez zmian. Użyj odpowiedzi kanarka do wykrywania dryfu dekompozycji. Jak menedżer powinien zareagować, gdy kanarek nie zgadza się z zsyntezowaną odpowiedzią?
4. Przeczytaj dokumentację `Process.hierarchical` CrewAI. Zidentyfikuj jedną konkretną barierę ochronną, którą stosuje CrewAI (limit kroków, ograniczenie manager_llm) i opisz, który tryb awarii ona celuje.
5. Porównaj zagnieżdżonych nadzorców LangGraph z hierarchią CrewAI. Która czyni pętle pojednania tańszymi do wykrycia?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Hierarchia | „Wzorzec struktury org." | Nadzorcy nad nadzorcami; tylko liście wykonują pracę. |
| Menedżer LLM | „Szef" | LLM, który rozkłada, przydziela i waliduje w węźle wewnętrznym. |
| Dryf dekompozycji | „Szef zgubił wątek" | Podział głównego menedżera nie obejmuje już oryginalnego pytania. |
| Pętla pojednania | „Niekończące się spotkania" | Pod-menedżerowie nie zgadzają się; główny ponownie deleguje; pracownicy uruchamiają się ponownie; pętla aż do wyczerpania budżetu. |
| Limit głębokości 2 | „Nie schodź głębiej niż 2 poziomy" | Empiryczna bariera: 3+ poziomów załamuje obserwowalność. |
| Pytanie kanarka | „Prawda podstawowa na każdym poziomie" | Pracownik, któremu zawsze zadawane jest oryginalne pytanie bez zmian, w celu wykrycia dryfu. |
| Łańcuch proweniencji | „Kto co powiedział" | Ślad od każdej syntezy do wyników liści, które ją wyprodukowały. |

## Further Reading

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — podręcznikowa hierarchia z menedżerem LLM
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — zagnieżdżony nadzorca przez `create_supervisor`
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — dlaczego Anthropic celowo wybrał płaskiego nadzorcę zamiast hierarchicznego
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — taksonomia MAST; sekcja o awariach koordynacji dokumentuje dryf dekompozycji