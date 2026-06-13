# Debat i współpraca wieloagentowa

> Du et al. (ICML 2024, "Society of Minds") uruchamia N instancji modelu, które niezależnie proponują odpowiedzi, a następnie iteracyjnie krytykują się nawzajem przez R rund, aby osiągnąć zbieżność. Poprawia faktyczność, przestrzeganie reguł i rozumowanie. Topologia rzadka bije pełną siatkę na koszcie tokenów.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 minutes

## Learning Objectives

- Wyjaśnij protokół debaty: N proponentów, R rund, zbieżność do wspólnej odpowiedzi.
- Opisz, dlaczego debata poprawia faktyczność, przestrzeganie reguł i rozumowanie.
- Wyjaśnij topologię rzadką: nie każdy uczestnik debaty musi widzieć każdego innego.
- Zaimplementuj debatę w stdlib na skryptowym LLM z wariantami pełnej siatki i rzadkim; zmierz koszt tokenów względem dokładności.

## Problem

Self-Refine (Lekcja 05) to jeden model krytykujący samego siebie — ryzyko myślenia grupowego. CRITIC (Lekcja 05) opiera krytykę na zewnętrznych narzędziach — nie zawsze dostępnych. Debaty wprowadzają trzeci tryb: wiele instancji, wzajemna krytyka, zbieżność przez niezgodność.

## Koncepcja

### Society of Minds (Du et al., ICML 2024)

- N instancji modelu niezależnie proponuje odpowiedzi na to samo pytanie.
- Przez R rund każdy model czyta propozycje innych i krytykuje je.
- Modele aktualizują swoje odpowiedzi na podstawie krytyki.
- Po R rundach zwracana jest zbieżna odpowiedź.

Oryginalne eksperymenty używały N=3, R=2 ze względu na koszt. Dokładność rośnie z większą liczbą agentów i rund w trudnych problemach (MMLU, GSM8K, ważność ruchów szachowych, generowanie biografii).

Kombinacje różnych modeli biją debaty jednego modelu: ChatGPT + Bard razem > każdy osobno.

### Topologia rzadka

"Improving Multi-Agent Debate with Sparse Communication Topology" (arXiv:2406.11776, 2024-2025) pokazało, że pełna siatka nie zawsze jest optymalna. Topologie rzadkie (gwiazda, pierścień, hub-and-spoke) mogą osiągnąć tę samą dokładność przy niższym koszcie tokenów. Każdy uczestnik widzi tylko podzbiór rówieśników.

Implikacje:

- Pełna siatka N=5, R=3 = 5 × 3 = 15 propozycji, każda czytająca 4 rówieśników = 60 operacji krytyki.
- Gwiazda N=5, R=3 (jeden hub + 4 szprychy) = 15 propozycji, szprychy czytają tylko hub = 12 operacji krytyki.

### Kiedy debata pomaga

- **Faktyczność.** N niezależnych propozycji, wzajemna weryfikacja zmniejsza halucynacje.
- **Przestrzeganie reguł.** Ważność ruchów szachowych — jeden model pomija regułę, inne ją wychwytują.
- **Rozumowanie otwarte.** Wielokrotne ujęcia zawężają się do poprawnej odpowiedzi.

### Kiedy debata szkodzi

- **Wrażliwość na opóźnienia.** N × R rund szeregowych to opóźnienie, którego możesz nie mieć.
- **Wrażliwość na koszt.** N × R tokenów na pytanie.
- **Proste wyszukiwanie faktów.** Jedno wyszukiwanie jest tańsze niż pięć debat.

### Praktyczne implementacje w 2026

- **Anthropic orchestrator-workers** (Lekcja 12) — jeden wariant debaty z krokiem syntezy.
- **LangGraph supervisor** (Lekcja 13) — centralny router + wyspecjalizowani agenci mogą implementować debatę jako węzeł.
- **OpenAI Agents SDK** (Lekcja 16) — agenci przekazują sobie sterowanie w tę i z powrotem dla iteracyjnej krytyki.
- **Ewaluacje wieloagentowe** — połącz debatę z evaluator-optimizer dla sygnału ewaluacyjnego.

### Gdzie ten wzór zawodzi

- **Zapaść zbieżności.** Wszyscy agenci zbiegają się do pierwszej błędnej odpowiedzi. Łagodź przez wymagane rundy niezgodności.
- **Awaria huba.** W topologii gwiazdy zły hub psuje wszystkich. Rotuj lub używaj wielu hubów.
- **Homogenizacja promptów.** Wszyscy agenci używają tego samego prompta; produkują te same odpowiedzi. Używaj zróżnicowanych promptów i/lub modeli.

## Build It

`code/main.py` implementuje debatę w stdlib:

- Klasa `Debater` (skryptowy LLM z dryfem opinii na uczestnika).
- Uruchamiacze `FullMeshDebate` i `SparseDebate`.
- Trzy pytania: jedno faktograficzne, jedno oparte na regułach, jedno wymagające rozumowania.
- Metryki: odpowiedź zbieżna, rundy do zbieżności, całkowita liczba operacji krytyki.

Uruchom:

```
python3 code/main.py
```

Wynik: dokładność i koszt dla każdego protokołu; rzadki dorównuje pełnej siatce w 2/3 pytań przy niższym koszcie.

## Use It

- **Anthropic orchestrator-workers** dla prostych debat z 2-3 pracownikami.
- **LangGraph** dla stanowej debaty wielorundowej z punktami kontrolnymi.
- **Własne** dla badań lub specjalistycznych gwarancji poprawności.

## Ship It

`outputs/skill-debate.md` tworzy szkielet debaty wieloagentowej z konfigurowalną topologią, N, R i regułą zbieżności.

## Exercises

1. Zaimplementuj regułę "wymuszonej niezgodności": w rundzie 1 każdy uczestnik musi przedstawić odrębną propozycję. Zmierz wpływ na szybkość zbieżności.
2. Dodaj agregację ważoną ufnością: uczestnicy zwracają (odpowiedź, ufność); agregator waży według ufności. Czy to pomaga?
3. Zamień jednego "agenta" na inny skryptowy LLM z innymi opiniami. Czy heterogeniczność poprawia dokładność?
4. Zmierz koszt tokenów dla pełnej siatki vs rzadkiej na swoich 3 pytaniach. Wykreśl koszt względem dokładności.
5. Przeczytaj artykuł Society of Minds. Rozszerz swoją zabawkę do N=5, R=3. Co się psuje? Co się poprawia?

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| Debate | "Wieloagentowa krytyka" | N proponentów, R rund wzajemnej krytyki, zbieżność |
| Full mesh | "Każdy czyta każdego" | Każdy uczestnik czyta każdego rówieśnika w każdej rundzie |
| Sparse topology | "Ograniczony widok rówieśników" | Uczestnicy czytają tylko podzbiór rówieśników |
| Hub-and-spoke | "Topologia gwiazdy" | Jeden centralny uczestnik, N-1 szprych czyta tylko hub |
| Convergence | "Zgodność" | Uczestnicy zbiegają się do wspólnej odpowiedzi |
| Society of Minds | "Artykuł Du et al. o debacie" | Metoda debaty wieloagentowej z ICML 2024 |

## Further Reading

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — kanoniczna debata wieloagentowa
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — wyniki topologii rzadkiej
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers jako wariant debaty
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — odpowiednik samokrytyki pojedynczego modelu