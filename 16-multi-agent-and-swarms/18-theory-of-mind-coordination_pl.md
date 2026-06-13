# Teoria Umysłu i Emergencjalna Koordynacja

> Li i in. (arXiv:2310.10701) pokazali, że agenci LLM w kooperacyjnej grze tekstowej wykazują **emergencjalną Teorię Umysłu (ToM) wyższego rzędu** — rozumowanie o tym, co inny agent sądzi o przekonaniach trzeciego agenta — ale zawodzą w planowaniu długoterminowym z powodu zarządzania kontekstem i halucynacji. Riedl (arXiv:2510.05174) zmierzył synergię wyższego rzędu w populacji i odkrył, że **tylko** warunek promptu ToM wytwarza zróżnicowanie powiązane z tożsamością i komplementarność ukierunkowaną na cel; LLMy o niższej pojemności wykazują tylko pozorną emergencję. Oznacza to, że emergencja koordynacji jest warunkowa od promptu i zależna od modelu, a nie darmowa. Ta lekcja implementuje minimalnego agenta świadomego ToM, uruchamia zadanie kooperacyjne z promptem ToM i bez niego oraz mierzy deltę koordynacji względem protokołu Riedl 2025.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 minutes

## Problem

Wieloagentowa koordynacja często wygląda magicznie: agenci dzielą pracę, przewidują się nawzajem, unikają redundancji. Zazwyczaj ta „emergencja" jest artefaktem inżynierii promptów — ktoś powiedział agentom, żeby „koordynowali." Usuń prompt, usuń koordynację.

Odkrycie Riedla z 2025 roku jest bardziej rygorystyczne: w kontrolowanych warunkach koordynacja pojawia się tylko wtedy, gdy agenci są promptowani do rozumowania o **umysłach innych agentów** (ToM). Bez promptu ToM, nawet silne modele wykazują wzorce koordynacji, które nie przeżywają kontroli statystycznych. To ma znaczenie dla produkcji: zespoły wdrażają funkcje „wieloagentowej koordynacji", które są zależne od promptu i kruche.

Ta lekcja traktuje ToM jako konkretną zdolność (rozumowanie o przekonaniach o przekonaniach), buduje minimalnego agenta świadomego ToM i mierzy, jak wygląda prawdziwa koordynacja w porównaniu z tym, co wygląda jak koordynacja dzięki odpowiedniej oprawie promptu.

## Koncepcja

### Co oznacza ToM

Psychologia rozwojowa: 3-latek myśli, że wewnętrzny świat każdego odpowiada jego własnemu. 5-latek rozumie, że inni mają różne przekonania. 7-latek rozumuje o przekonaniach o przekonaniach („ona myśli, że ja myślę, że piłka jest pod kubkiem"). Są to odpowiednio ToM zerowego, pierwszego i drugiego rzędu.

Dla agentów LLM, rzędy ToM odpowiadają:

- **Rząd zerowy:** brak modelu innych. Agent działa tylko na podstawie własnych obserwacji.
- **Rząd pierwszy:** agent ma model przekonań każdego innego agenta. „Alicja uważa, że X."
- **Rząd drugi:** agent modeluje rekurencyjne przekonania. „Alicja uważa, że Bob uważa, że X."

Li i in. 2023 odkryli, że ToM pierwszego i drugiego rzędu pojawia się u agentów LLM w grach kooperacyjnych, ale degraduje się przy długim horyzoncie i zawodnej komunikacji.

### Test Sally-Anne w skrócie

Test fałszywych przekonań z 1985 roku: Sally wkłada kulkę do koszyka A, wychodzi. Anne przenosi ją do koszyka B. Gdzie Sally będzie szukać, gdy wróci? Dziecko z ToM pierwszego rzędu mówi koszyk A (przekonanie Sally różni się od rzeczywistości). Dziecko bez ToM mówi koszyk B.

LLMy z epoki GPT-4 zdają testy w stylu Sally-Anne, gdy są zadane wprost. Zawodzą, gdy narracja jest długa, scena zmienia się kilka razy lub pytanie jest zadane pośrednio. Taki jest praktyczny stan ToM w produkcyjnych LLM w 2026 roku.

### Pomiar koordynacji Riedla

Riedl (arXiv:2510.05174) zbudował test na skalę populacji: N agentów, cel kooperacyjny, zmienne warunki promptu. Mierz:

1. **Zróżnicowanie powiązane z tożsamością.** Czy agenci rozwijają stabilne rozróżnienia ról w czasie?
2. **Komplementarność ukierunkowaną na cel.** Czy działania agentów uzupełniają się (różne podzadania), a nie dublują?
3. **Synergię wyższego rzędu.** Statystyczna miara tego, czy grupa osiąga to, czego żaden podzbiór nie mógłby osiągnąć.

Wynik: tylko w warunku promptu ToM wszystkie trzy metryki dają sygnał powyżej baseline'u. Bez promptowania ToM, metryki oscylują wokół przypadku dla modeli o umiarkowanej pojemności. Duże modele wykazują pewną koordynację bez jawnego promptowania ToM, ale efekt jest mniejszy niż z jawnym promptowaniem.

### Iluzja koordynacji

Bez kontroli statystycznych, „emergencjalna koordynacja" w demach często odzwierciedla:

- Inżynierię promptów, która wbudowuje koordynację (prompty systemowe mówiące „pracujcie razem").
- Błąd obserwatora (widzimy wzorce, których się spodziewamy).
- Wybór udanych przebiegów po fakcie.

Systemy produkcyjne, które reklamują „emergencjalną koordynację" bez mierzalnego sygnału, należy traktować jako marketing. Mierz, zanim zaczniesz twierdzić.

### Minimalny agent świadomy ToM

Struktura:

```
stan agenta:
  własne_przekonania:     {fakty, w które agent wierzy}
  modele_innych:          {id_innego_agenta -> {przekonania_przypisane_mu_przez_agenta}}
  działania_ostatnie_N:   [historia działań innych]

aktualizacja obserwacji:
  - aktualizuj własne_przekonania na podstawie bezpośredniej obserwacji
  - aktualizuj modele_innych[id_agenta] na podstawie jego działania + poprzednich przekonań

wybór działania:
  - wylicz działania kandydujące
  - dla każdego, przewidź, co zrobi każdy inny agent na podstawie modelowanych przekonań
  - wybierz działanie maksymalizujące łączny wynik przy tych przewidywaniach
```

Atrybut `modele_innych` to stan ToM. ToM pierwszego rzędu utrzymuje tylko jeden poziom. ToM drugiego rzędu dodaje `modele_innych[i][modele_innych_j]` — co według mnie agent i myśli, że agent j uważa.

### Dlaczego długi horyzont szkodzi

Li i in. dokumentują: ograniczenia kontekstu powodują, że agenci zapominają, które przekonanie należy do kogo. Halucynacje dodają fałszywe przekonania do modeli innych agentów. Oba powodują błędy „myślałem, że on myślał X", które kumulują się w czasie.

Łagodzenia udokumentowane w artykule i w pracach uzupełniających z lat 2024-2026:

- **Jawny stan ToM w prompcie.** Strukturalny format: `{id_agenta: lista_przekonań}`. Wymusza zachowanie wiązania tożsamość-przekonanie przy wyszukiwaniu.
- **Krótsze łańcuchy rozumowania.** Mniej aktualizacji ToM na turę zmniejsza kumulację halucynacji.
- **Zewnętrzne przechowywanie ToM.** Utrzymuj model poza kontekstem LLM; wstrzykuj tylko istotne części na turę.

### Gdzie ToM zawodzi w produkcji

- **Ustawienia adversarialne.** Agenci z dobrą ToM są łatwiejsi do manipulowania (możesz modelować, co oni modelują o tobie, a potem wykorzystać).
- **Zespoły heterogeniczne.** Gdy modele są różne, model ToM działający dla jednego przeciwnika nie uogólnia się.
- **Zadania zależne od prawdy podstawowej.** ToM dotyczy przekonań; jeśli poprawność zależy od faktów, ToM może być rozpraszająca.

### Koordynacja, którą faktycznie możesz zmierzyć

Trzy praktyczne sygnały, że koordynacja zespołu jest prawdziwa, a nie tylko wynikiem promptu:

1. **Komplementarność w czasie.** W zadaniu wieloturowym, czy działania agentów pokrywają rozłączne podzadania?
2. **Antycypacja.** Czy działanie agenta A w turze T+1 zależy od przewidywania działania B w turze T+2, które okazało się poprawne?
3. **Korekta.** Gdy A źle odczytuje przekonanie B w turze T, czy A koryguje do tury T+2?

Są one mierzalne w logowanym systemie wieloagentowym. Stanowią one merytoryczną wersję narracji o „koordynacji."

## Build It

`code/main.py` implementuje:

- `ToMAgent` — śledzi własne przekonania i modele przekonań dla każdego innego agenta.
- Zadanie kooperacyjne: trzej agenci muszą zebrać trzy żetony z trzech pudełek; każde pudełko może pomieścić jeden żeton. Agenci nie mogą się komunikować; wnioskują o intencjach z działań innych.
- Dwie konfiguracje: `zeroth_order` (bez ToM) i `first_order` (ToM z modelami przekonań jednego poziomu).
- Pomiar w 200 randomizowanych próbach: wskaźnik ukończenia, wskaźnik duplikacji (dwóch agentów celujących w to samo pudełko), średnia liczba tur do ukończenia.

Uruchomienie:

```
python3 code/main.py
```

Oczekiwane wyjście: agenci rzędu zerowego dublują wysiłek w ~35% przypadków i kończą ~60% prób w 10 turach. Agenci z ToM pierwszego rzędu dublują w ~5% i kończą ~95%. Delta to mierzalny efekt koordynacji.

## Use It

`outputs/skill-tom-auditor.md` to umiejętność, która audytuje twierdzenie systemu wieloagentowego o „emergencjalnej koordynacji." Sprawdza obecność oprawy promptowej, istotność statystyczną względem kontroli i zmierzoną komplementarność.

## Ship It

Lista kontrolna dla twierdzeń o koordynacji:

- **Warunek kontrolny.** Wersja twojego systemu bez promptu koordynacji. Zmierz obie.
- **Test statystyczny.** Czy różnica między systemem a kontrolą jest istotna przy `p < 0.05` na twojej metryce?
- **Miarę komplementarności.** Rozłączność działań w czasie, a nie tylko końcowy sukces.
- **Log przypadków awarii.** Gdy agenci się źle koordynują, jak wygląda stan ToM?
- **Ujawnienie pojemności modelu.** Jeśli efekt znika na mniejszych modelach, powiedz o tym.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że ToM pierwszego rzędu zmniejsza wskaźnik duplikacji ~7x. Czy różnica utrzymuje się, gdy skalować do 5 agentów i 5 pudełek?
2. Zaimplementuj ToM drugiego rzędu (agent A modeluje, co B myśli o C). Czy poprawia się względem pierwszego rzędu? W jakich zadaniach?
3. Wstrzyknij **halucynację** do stanu ToM: losowo odwróć jedno przekonanie na turę. Jak bardzo degraduje to wydajność pierwszego rzędu?
4. Przeczytaj Li i in. (arXiv:2310.10701). Odtwórz odkrycie „degradacji przy długim horyzoncie": gdy tury rosną z 10 do 30, jak zmienia się wydajność twojego ToM pierwszego rzędu?
5. Przeczytaj Riedl 2025 (arXiv:2510.05174). Zaimplementuj statystykę synergii wyższego rzędu na swoich logach symulacji. Czy efekt jest obecny bez warunku promptu ToM?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Teoria Umysłu | „Rozumienie umysłów innych" | Zdolność do modelowania przekonań innego agenta. Gradowana według rzędu (0, 1, 2+). |
| Test Sally-Anne | „Test fałszywych przekonań" | Psychologia rozwojowa 1985; LLMy zdają proste wersje, zawodzą na złożonych. |
| ToM pierwszego rzędu | „A uważa X" | Modelowanie przekonań jednego innego o faktach. |
| ToM drugiego rzędu | „A uważa, że B uważa X" | Rekurencyjne modelowanie o jeden poziom głębiej. |
| Zróżnicowanie powiązane z tożsamością | „Stabilne role w czasie" | Metryka Riedla: role się utrzymują, nie są losowe. |
| Komplementarność ukierunkowana na cel | „Rozłączne działania" | Agenci celują w różne podzadania, a nie to samo. |
| Synergia wyższego rzędu | „Grupa przewyższa podzbiór" | Statystyczna miara Riedla dla prawdziwej koordynacji. |
| Iluzja koordynacji | „Wygląda na skoordynowane" | Pozór koordynacji przez oprawę promptową bez mierzalnego sygnału. |

## Dalsza Literatura

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — emergencjalna ToM w grach kooperacyjnych; tryby awarii przy długim horyzoncie
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — pomiar na skalę populacji; promptowanie ToM jest warunkiem nośnym
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — pochodzenie koncepcji ToM z 1978 roku
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — artykuł Sally-Anne (1985)