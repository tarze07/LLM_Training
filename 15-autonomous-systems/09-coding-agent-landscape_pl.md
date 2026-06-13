# Krajobraz Autonomicznych Agentów Kodujących (2026)

> SWE-bench Verified wzrósł z 4% do 80.9% w niespełna trzy lata. Ten sam Claude Sonnet 4.5 osiągnął 43.2% na SWE-agent v1 i 59.8% na autonomicznym Cline — scaffolding wokół modelu ma teraz tak duże znaczenie jak sam model. OpenHands (dawniej OpenDevin) jest najbardziej aktywną platformą na licencji MIT, a jego pętla CodeAct wykonuje akcje Pythona bezpośrednio w sandboxie zamiast wywołań narzędzi JSON. Liczby w nagłówkach ukrywają problem metodologiczny: 161 z 500 zadań SWE-bench Verified wymaga tylko zmiany 1–2 linii, a SWE-bench Pro (zadania z 10+ liniami) osiąga 23–59% dla tych samych modeli granicznych.

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## Problem

„Który agent kodujący jest najlepszy" to złe pytanie. Właściwe pytanie brzmi: na rozkładzie zadań, który pasuje do mojej pracy, z scaffoldingiem, który uruchomię w produkcji, jaką niezawodność end-to-end uzyskuję?

Między 2022 a 2026 rokiem dziedzina nauczyła się, że scaffolding — warstwa wyszukiwania, planista, sandbox, pętla edytuj-weryfikuj, format informacji zwrotnej — ma kluczowe znaczenie. Claude Sonnet 4.5 na SWE-agent v1 osiągnął 43.2% na SWE-bench Verified; ten sam model wewnątrz autonomicznego scaffoldingu Cline'a osiągnął 59.8%. 16.6 punktów różnicy, te same wagi. Bazowy model jest komponentem; pętla jest produktem.

Problemem towarzyszącym jest to, że nasycenie benchmarków ukrywa regresje. SWE-bench Verified jest bliski nasycenia, a ogon łatwych zadań (161 z 500 wymagających ≤2 linii) podciąga górne wyniki. Jakość w świecie rzeczywistym jest lepiej mierzona na rozkładach takich jak SWE-bench Pro (zmiany 10+ linii), gdzie ci sami liderzy wciąż osiągają 23–59%.

## Koncepcja

### SWE-bench, jeden akapit

SWE-bench (Jimenez et al.) bierze prawdziwe zgłoszenia GitHub z rzeczywistymi łatami i prosi agenta o wyprodukowanie łaty, która sprawi, że zestaw testów przejdzie. SWE-bench Verified (OpenAI, 2024) to ręcznie wyselekcjonowany podzbiór 500 zadań z usuniętymi niejednoznacznymi i wadliwymi zadaniami. SWE-bench Pro to trudniejszy następca — zadania wymagające zmian 10+ linii, gdzie obecne graniczne agenty osiągają 23–59%.

### Co krzywa 2022 → 2026 faktycznie pokazuje

- **2022**: modele badawcze przy ~4% na surowym SWE-bench.
- **2024**: GPT-4 + scaffolding w stylu Devin przy ~14%; SWE-agent przy ~12%.
- **2025**: Claude 3.5/3.7 Sonnet wewnątrz Aider i SWE-agent wchodzą w zakres 40–55%.
- **2026**: Claude Sonnet 4.5 i graniczni konkurenci przy 70–80%+ na SWE-bench Verified. Tablica liderów Epoch AI śledzi to na żywo.

Nachylenie pochodziło z trzech nakładających się źródeł: lepszych modeli bazowych, lepszego scaffoldingu (CodeAct, refleksja, pętle weryfikatora) i lepszych benchmarków (Verified usuwający szum).

### CodeAct a wywołania narzędzi JSON

OpenHands (All-Hands-AI, arXiv:2407.16741, dawniej OpenDevin) postawił na konkretny zakład architektoniczny: zamiast modelu emitującego wywołania narzędzi JSON, które host dekoduje i wykonuje, model emituje kod Pythona, a kernel w stylu Jupyter uruchamia go w sandboxie. Agent może zapętlać pliki, łączyć narzędzia i łapać własne wyjątki w ramach jednej akcji.

Kompromis:

- **Wywołania narzędzi JSON**: każda akcja to jedna tura; łatwa do audytu; ograniczona kompozycyjność; bezpieczna domyślnie, ponieważ każde wywołanie przechodzi przez jawny walidator.
- **CodeAct**: jedna akcja może być całym programem; kompozycyjna; wymaga wzmocnionego sandboksa (OpenHands używa izolacji Docker); tryby awarii obejmują wszystko, na co pozwala środowisko uruchomieniowe sandboksa.

Obie architektury są w produkcji. CodeAct dominuje w otwartych platformach (OpenHands, smolagents). Wywołania narzędzi JSON pozostają dominujące w zarządzanych usługach (Anthropic Managed Agents, OpenAI Assistants), gdzie dostawca kontroluje wykonawcę.

### Scaffoldy w krajobrazie 2026

| Scaffold | Licencja | Model wykonania | Właściwość wyróżniająca |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct w Docker | Najbardziej aktywna otwarta platforma; strumień zdarzeń do odtworzenia |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | Pierwszy end-to-end scaffold SWE-bench |
| Aider | Apache-2 | edycja-poprzez-diff w lokalnym repo | Minimalny scaffold, silna stabilność regresji |
| Cline | Apache-2 | Agent VS Code z polityką narzędzi | Najwyżej punktowany otwarty scaffold na Sonnet 4.5 |
| Devin (Cognition) | Własnościowa | Zarządzana VM + planista | Pierwsza kategoria produktu „inżynier oprogramowania AI" |
| Claude Code | Własnościowa | Tryby uprawnień + rutyny | Lekcja 10 opisuje pętlę agenta w szczegółach |

### Dlaczego scaffolding dominuje

Uruchomienie kodowania to trajektoria długiego horyzontu (Lekcja 1). Niezawodność kumuluje się na krokach. Trzy miejsca, gdzie scaffolding zdobywa punkty:

1. **Wyszukiwanie**: znalezienie właściwych plików do odczytania to ciche wąskie gardło. ACI SWE-agenta, indeks plików OpenHands i mapa repozytorium Aidera wszystkie to atakują.
2. **Pętla weryfikatora**: uruchamianie testów, czytanie śladów stosu i ponawianie prób to różnica 10+ punktów na SWE-bench.
3. **Kontenerowanie awarii**: sandbox, który wycofuje zmiany przy błędzie, zapobiega kumulacji szkód. Ten sam model z i bez pętli weryfikatora wygląda jak dwa różne produkty.

### Nasycenie benchmarku a rzeczywisty rozkład

Autorzy OpenHands i Epoch AI obaj wskazują, że SWE-bench Verified ma łatwy ogon: 161 z 500 zadań wymaga tylko 1–2 linii zmian. Wysokie wyniki są częściowo napędzane przez ten ogon. SWE-bench Pro ogranicza się do zmian 10+ linii i zwraca wyniki w zakresie 23–59% nawet dla systemów granicznych. Twój produkcyjny rozkład jest prawie na pewno bliższy Pro niż Verified.

Implikacja dla wyboru agenta: uruchom podzbiór podobny do Pro z własnego zaległego backlogu błędów. Wynik, który się liczy, to wynik na zadaniach reprezentatywnych dla tego, co wdrażasz.

## Użyj Tego

`code/main.py` porównuje dwa zabawkowe scaffoldy agentów na stałej mini-rozkladzie zadań:

1. Scaffold z **wywołaniami narzędzi JSON**, który wykonuje jedną akcję na turę.
2. Scaffold **CodeAct**, który może emitować mały fragment Pythona na akcję.

Oba używają atrapy „modelu" (deterministyczne reguły), aby porównanie izolowało scaffold od jakości modelu. Wynik pokazuje, że scaffold CodeAct rozwiązuje więcej zadań w mniejszej liczbie tur, kosztem większego promienia rażenia na akcję.

## Wdróż To

`outputs/skill-scaffold-audit.md` pomaga przeprowadzić audyt proponowanego scaffoldu agenta kodującego przed adopcją: jakość wyszukiwania, obecność weryfikatora, izolacja sandboksa i dopasowanie benchmarku do rozkładu.

## Ćwiczenia

1. Uruchom `code/main.py`. Ile tur zajmuje każdy scaffold na tym samym zestawie zadań? Jaki jest promień rażenia na akcję każdego z nich?

2. Przeczytaj artykuł OpenHands (arXiv:2407.16741). Artykuł argumentuje, że CodeAct bije wywołania narzędzi JSON w złożonych zadaniach. Zidentyfikuj jeden tryb awarii, który artykuł uznaje, i napisz jedno zdanie o tym, kiedy ten tryb dominowałby w produkcji.

3. Wybierz jedno zadanie ze swojego backlogu błędów, które wymagałoby 10+ linii zmian w dwóch plikach. Oszacuj prawdopodobieństwo sukcesu end-to-end dla modelu granicznego pod (a) wywołaniami narzędzi JSON i (b) CodeAct. Uzasadnij różnicę.

4. SWE-bench Verified ma 161 zadań z jednego pliku, 1–2 linii. Skonstruuj wynik, który je wyklucza. Jak zmienia się tablica liderów?

5. Przeczytaj „Introducing SWE-bench Verified" (OpenAI). Wyjaśnij konkretną metodologię używaną do usuwania niejednoznacznych zadań i wymień jedną kategorię, którą selekcja mogłaby pominąć.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|---|---|---|
| SWE-bench | „Benchmark kodowania" | Rzeczywiste zgłoszenia GitHub z prawdziwymi łatami i zestawami testów |
| SWE-bench Verified | „Oczyszczony podzbiór" | 500 ręcznie wyselekcjonowanych zadań, obecny łatwy ogon |
| SWE-bench Pro | „Trudniejszy podzbiór" | Zmiany 10+ linii; granica osiąga 23–59% |
| CodeAct | „Kod-jako-akcja" | Agent emituje Python; kernel w stylu Jupyter wykonuje w sandboxie |
| Wywołanie narzędzia JSON | „Wywoływanie funkcji" | Każda akcja to strukturalny ładunek JSON walidowany przed wykonaniem |
| Scaffold | „Framework agenta" | Wyszukiwanie + planista + wykonawca + pętla weryfikatora wokół modelu bazowego |
| ACI (Agent-Computer Interface) | „Format SWE-agenta" | Zestaw poleceń zaprojektowany dla ergonomii LLM, a nie ludzkich shelli |
| Pętla weryfikatora | „Testuj i próbuj ponownie" | Uruchom testy, przeczytaj wyniki, popraw łatę; największy pozamodelowy wzrost niezawodności |

## Dalsza Lektura

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — oryginalny benchmark i metodologia.
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — jak zbudowano wyselekcjonowany podzbiór.
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — architektura CodeAct i projekt strumienia zdarzeń.
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — wyniki śledzone na żywo.
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — ujęcie niezawodności długohoryzontowego agenta kodującego.
