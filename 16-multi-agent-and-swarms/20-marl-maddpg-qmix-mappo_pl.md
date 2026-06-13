# MARL — MADDPG, QMIX, MAPPO

> Dziedzictwo uczenia przez wzmacnianie w koordynacji wieloagentowej, które wciąż wpływa na systemy agentów LLM w 2026 roku. **MADDPG** (Lowe i in., NeurIPS 2017, arXiv:1706.02275) wprowadziło scentralizowane trenowanie, zdecentralizowane wykonywanie (CTDE): każdy krytyk widzi stany i działania wszystkich agentów podczas trenowania; w czasie testu działają tylko lokalne aktory. Działa w ustawieniach kooperacyjnych, konkurencyjnych i mieszanych. **QMIX** (Rashid i in., ICML 2018, arXiv:1803.11485) to dekompozycja wartości z monotoniczną siecią mieszającą; per-agentowe Q łączą się we wspólne Q, więc `argmax` rozdziela się czysto — dominuje w StarCraft Multi-Agent Challenge (SMAC). **MAPPO** (Yu i in., NeurIPS 2022, arXiv:2103.01955) to PPO ze scentralizowaną funkcją wartości; „zaskakująco skuteczne" w świecie cząstek, SMAC, Google Research Football, Hanabi przy minimalnym dostrajaniu hiperparametrów. Algorytmy te stanowią podstawę trenowania polityk dla zespołów agentów, które muszą działać zdecentralizowanie. MAPPO jest **domyślnym baseline'em kooperacyjnego MARL w 2026 roku**. Ta lekcja buduje każdy z nich na małej siatce świata-zabawki i osadza trzy idee w pamięci mięśniowej przed dotknięciem trenowania agentów LLM.

**Type:** Learn
**Languages:** Python (stdlib, small NumPy-free implementations)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 minutes

## Problem

Systemy agentów LLM coraz częściej trenują polityki dla koordynacji międzyagentowej: kiedy oddelegować, kiedy działać, którego rówieśnika wywołać. Literatura, która mówi, jak trenować takie polityki, to Wieloagentowe Uczenie przez Wzmacnianie (MARL), które poprzedza falę LLM i ma mały zestaw dominujących algorytmów.

Czytanie artykułów MARL bez słownictwa wzorców jest bolesne. Scentralizowane trenowanie ze zdecentralizowanym wykonywaniem (CTDE), dekompozycja wartości i scentralizowani krytycy to nie modne hasła — to konkretne odpowiedzi na konkretne problemy:

- Niezależne RL (każdy agent uczy się sam) jest niestacjonarne z perspektywy każdego agenta. Źle.
- Scentralizowane RL (jeden agent kontroluje wszystko) nie skaluje się i narusza ograniczenia wykonawcze.
- CTDE łączy najlepsze z obu światów: trenuj z globalną informacją, wdrażaj z lokalnymi politykami.

## Koncepcja

### Trzy środowiska używane w artykułach

- **Świat cząstek (multi-agent particle env).** Prosta fizyka 2D z zadaniami kooperacyjnymi/konkurencyjnymi. Oryginalne środowisko testowe MADDPG.
- **StarCraft Multi-Agent Challenge (SMAC).** Kooperacyjne mikrozarządzanie, częściowa obserwacja. Środowisko testowe QMIX. Dyskretne akcje, ciągłe stany.
- **Google Research Football, Hanabi, MPE.** Baseline'y MAPPO.

Różne środowiska mają różne typy akcji/obserwacji. Algorytmy dobierają odpowiednio.

### MADDPG (2017) — wzorzec CTDE

Każdy agent `i` ma aktor `mu_i(o_i)`, który mapuje własną obserwację na akcję. Każdy agent ma także krytyka `Q_i(x, a_1, ..., a_n)`, który widzi wszystkie obserwacje i wszystkie akcje podczas trenowania. Aktor jest aktualizowany przez gradient polityki względem oceny krytyka.

```
aktualizacja aktora:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) przy a_i=mu_i(o_i)]
aktualizacja krytyka:   TD na Q_i(x, a_1..n) na podstawie wspólnego oszacowania następnego stanu
```

Dlaczego CTDE: w czasie trenowania znamy działania wszystkich; używamy tego do redukcji wariancji każdego krytyka. W czasie wdrożenia każdy agent widzi tylko `o_i` i wywołuje `mu_i(o_i)`.

Tryb awarii: krytycy rosną z N agentów (wejście obejmuje wszystkie akcje). Nie skaluje się powyżej ~10 agentów bez przybliżeń.

### QMIX (2018) — dekompozycja wartości

Tylko kooperacyjny. Globalna nagroda jest sumą monotonicznej funkcji per-agentowych wartości Q:

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

Monotoniczność gwarantuje, że `argmax_a Q_tot` może być obliczone przez każdego agenta wybierającego `argmax_{a_i} Q_i` niezależnie. To jest **dokładnie właściwość zdecentralizowanego wykonywania**, której potrzebujesz. W czasie trenowania sieć mieszająca produkuje `Q_tot` z per-agentowych Q.

Dlaczego QMIX wygrywa na SMAC: kooperacyjne mikrozarządzanie w StarCraft ma homogenicznych agentów, lokalne obserwacje, globalną nagrodę — idealne dopasowanie do dekompozycji wartości.

Tryb awarii: ograniczenie monotoniczności jest restrykcyjne; niektóre zadania mają struktury nagród, które nie są monotonicznie dekomponowalne (jeden agent poświęca się dla zespołu). Rozszerzenia (QTRAN, QPLEX) rozluźniają to.

### MAPPO (2022) — przeoczony domyślny

Wieloagentowe PPO: PPO ze scentralizowaną funkcją wartości. Każdy agent ma własną politykę; wszyscy agenci dzielą (lub mają per-agentowe) funkcje wartości, które widzą pełny stan. Yu i in. 2022 porównali MAPPO z MADDPG, QMIX i ich rozszerzeniami na pięciu benchmarkach i stwierdzili:

- MAPPO dorównuje lub bije metody off-policy MARL w świecie cząstek, SMAC, Google Research Football, Hanabi, MPE.
- Wymagane minimalne dostrajanie hiperparametrów.
- Stabilne trenowanie; powtarzalne między ziarnami.

Społeczność nie doceniała on-policy MARL aż do tego artykułu. W 2026 roku MAPPO jest domyślnym baseline'em dla kooperacyjnego MARL; każda nowa metoda musi go pokonać.

### Dlaczego inżynierowie agentów LLM powinni się tym przejmować

Trzy bezpośrednie zastosowania:

1. **Trenowanie routera.** Meta-agent wybiera, który sub-agent obsługuje zadanie. To problem MARL z N zdecentralizowanymi sub-agentami i jednym scentralizowanym routerem. MAPPO pasuje.
2. **Emergencja ról.** W symulacjach agentów generatywnych, trenowanie agentów do przyjmowania komplementarnych ról w czasie to problem MARL w przebraniu. Dekompozycja wartości w stylu QMIX wymusza komplementarność przez konstrukcję.
3. **Wieloagentowe użycie narzędzi.** Gdy agenci dzielą narzędzia i konkurują o budżet, trenowanie ich przez CTDE produkuje wdrażalne lokalne polityki, które respektują ograniczenia zasobów.

Praktyczne zastrzeżenie: w 2026 roku większość produkcyjnych systemów agentów LLM promptuje swoje polityki, zamiast je trenować. MARL wchodzi w grę, gdy masz (a) dużo danych interakcyjnych, (b) jasny sygnał nagrody i (c) gotowość do inwestycji w infrastrukturę treningową.

### CTDE jako wzorzec projektowy poza RL

Nawet bez trenowania, CTDE jest użytecznym wzorcem architektonicznym:

- Podczas *projektowania* zakładaj pełną widoczność zespołu.
- W *czasie wykonania* wymuszaj zdecentralizowane wykonywanie: każdy agent widzi tylko `o_i`.

Wzorzec wymusza utrzymywanie stanu per-agentowego w jawnej postaci i myślenie o częściowej obserwowalności z góry. Wiele produkcyjnych systemów wieloagentowych po cichu zakłada wspólny stan wszędzie — dyscyplina CTDE temu zapobiega.

### Problem niestacjonarności

Gdy wielu agentów uczy się jednocześnie, środowisko każdego agenta (które obejmuje polityki innych) jest niestacjonarne. Klasyczne dowody RL dla pojedynczego agenta zawodzą. Algorytmy MARL w tej lekcji wszystkie rozwiązują ten problem:

- MADDPG: globalny krytyk widzi wszystkie akcje, więc jego oszacowanie wartości jest stacjonarne.
- QMIX: dekompozycja wartości przenosi uczenie się do wspólnej przestrzeni Q, gdzie optymalność jest dobrze zdefiniowana.
- MAPPO: scentralizowana funkcja wartości tłumi wariancję ze zmian polityk innych.

W systemach agentów LLM niestacjonarność manifestuje się jako „mój agent działał w zeszłym miesiącu, teraz ten inny agent upstream się zmienił, mój źle się zachowuje." Trenowanie MARL z CTDE to zasadnicza naprawa; poprawki na poziomie promptu są szybsze, ale mniej trwałe.

### Czego ta lekcja NIE obejmuje

Trenowanie rzeczywistych sieci to temat Fazy 09. Ta lekcja buduje wersje ze skryptowanymi politykami, które demonstrują wzorce CTDE, dekompozycji wartości i scentralizowanej wartości bez aktualizacji gradientowych. Celem jest internalizacja wzorców, zanim sięgniesz po pełną bibliotekę MARL (PyMARL, MARLlib, RLlib multi-agent).

## Build It

`code/main.py` implementuje trzy demonstracje wzorców, wszystkie na małej kooperacyjnej siatce świata-zabawki z 2 agentami:

- Środowisko: 2 agentów na siatce 4x4, jedna kulka nagrody. Nagroda = 1 jeśli któryś agent dotrze do kulki; zadanie kończy się.
- `IndependentAgents` — każdy agent traktuje innych jako środowisko. Baseline.
- `MADDPGStyle` — scentralizowany krytyk oblicza wspólną wartość; polityki aktorów aktualizują się na jej podstawie. Skryptowana poprawa polityki.
- `QMIXStyle` — dekompozycja wartości z monotonicznym miksere.
- `MAPPOStyle` — scentralizowana funkcja wartości; polityki aktualizują się względem wspólnego baseline'u.

Wszystkie cztery uruchamiają te same epizody i raportują średnią liczbę kroków do celu. Warianty CTDE zbiegają się do krótszych ścieżek niż niezależny baseline.

Uruchomienie:

```
python3 code/main.py
```

Oczekiwane wyjście: niezależni agenci potrzebują średnio ~6 kroków; warianty CTDE zbiegają się do ~3.5 kroków (optymalne dla siatki 4x4 to 3). Różnica wzorców uwidacznia się mimo skryptowanych polityk.

## Use It

`outputs/skill-marl-picker.md` to umiejętność, która wybiera algorytm MARL dla danego zadania wieloagentowego: kooperacyjne vs. konkurencyjne, homogeniczne vs. heterogeniczne, typ przestrzeni akcji, skala, sygnał nagrody.

## Ship It

MARL w produkcji jest rzadki. Gdy go używasz:

- **Zacznij od MAPPO.** Artykuł z 2022 roku ustanowił to jako baseline; odtworzenie go najpierw oszczędza tygodnie gonienia za bardziej wymyślnymi metodami.
- **Loguj strumień obserwacji i akcji każdego agenta.** Debugowanie MARL bez śladów per-agentowych jest beznadziejne.
- **Oddziel kod trenowania od kodu wykonywania.** CTDE to dyscyplina; upewnij się, że ścieżka wykonania naprawdę widzi tylko `o_i`.
- **Ostrzeżenie o kształtowaniu nagrody.** MARL jest niezwykle wrażliwy na projekt nagrody. Jeden błąd koordynacji w kształtowaniu i agenci uczą się go wykorzystywać. Przeprowadź testy adversarialne.
- **Dla agentów LLM**, rozważ najpierw polityki na poziomie promptu. Inwestuj w trenowanie MARL tylko wtedy, gdy obecne są dane interakcyjne + sygnał nagrody + infrastruktura.

## Ćwiczenia

1. Uruchom `code/main.py`. Zmierz różnicę w krokach do celu między niezależnymi agentami a agentami w stylu MAPPO. Czy różnica rośnie czy maleje na siatce 6x6?
2. Zaimplementuj wariant konkurencyjny: dwóch agentów, jedna kulka, tylko pierwszy, który dotrze, dostaje nagrodę. Który wzorzec radzi sobie czysto z konkurencją? Historycznie MADDPG.
3. Przeczytaj MADDPG (arXiv:1706.02275) Sekcja 3. Zaimplementuj dokładną regułę aktualizacji krytyka symbolicznie w pseudokodzie własnymi słowami.
4. Przeczytaj MAPPO (arXiv:2103.01955). Dlaczego autorzy argumentują, że scentralizowana wartość + PPO bije off-policy MARL na ich benchmarkach? Wymień trzy najsilniejsze twierdzenia.
5. Zastosuj CTDE jako wzorzec projektowy do hipotetycznego systemu agentów LLM (np. agent badawczy + sumaryzator + programista). Jaka jest wspólna informacja dostępna w czasie projektowania, która nie jest dostępna w czasie wykonania?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| MARL | „Wieloagentowe RL" | Uczenie przez wzmacnianie dla systemów wieloagentowych. |
| CTDE | „Scentralizowane trenowanie, zdecentralizowane wykonywanie" | Trenuj z globalną informacją; wdrażaj z lokalnymi politykami. |
| MADDPG | „Wieloagentowe DDPG" | CTDE z per-agentowym krytykiem widzącym wszystkie obserwacje + akcje. |
| QMIX | „Dekompozycja wartości" | Monotoniczne mieszanie per-agentowych Q. Kooperacyjny. |
| MAPPO | „Wieloagentowe PPO" | PPO ze scentralizowaną funkcją wartości. Domyślny baseline 2026. |
| Dekompozycja wartości | „Suma indywidualnych Q" | Wspólne Q reprezentowane jako monotoniczna funkcja per-agentowych Q. |
| Niestacjonarność | „Ruchome cele" | Środowisko każdego agenta zmienia się, gdy inni się uczą. Główny problem MARL. |
| On-policy / off-policy | „Ucz się z bieżących / z replayu" | PPO jest on-policy (MAPPO); DDPG i Q-learning są off-policy. |
| SMAC | „StarCraft Multi-Agent Challenge" | Benchmark kooperacyjnego mikrozarządzania; rodzinny teren QMIX. |

## Dalsza Literatura

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG; NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX; ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO; NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — czytelne przedstawienie wyniku MAPPO
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge