# Optymalizacja Rojowa dla LLM (PSO, ACO)

> Inspirowana biologią optymalizacja powraca w kontekście LLM. **LMPSO** (arXiv:2504.09247) używa PSO, gdzie prędkość każdej cząstki to prompt, a LLM generuje kolejnego kandydata; działa dobrze na strukturalnych wyjściach sekwencyjnych (wyrażenia matematyczne, programy). **Model Swarms** (arXiv:2410.11163) traktuje każdego eksperta LLM jako cząstkę PSO na rozmaitości wag modelu i raportuje **średni zysk 13.3%** w stosunku do 12 baseline'ów na 9 zbiorach danych przy zaledwie 200 instancjach. **SwarmPrompt** (ICAART 2025) hybrydyzuje PSO + Grey Wolf dla optymalizacji promptów. **AMRO-S** (arXiv:2603.12933) to specjaliści feromonowi inspirowani ACO do routingu wieloagentowego LLM — **4.7x przyspieszenie**, interpretowalne dowody routingu, asynchroniczna aktualizacja z bramką jakości, która oddziela wnioskowanie od uczenia się. Ta lekcja implementuje PSO w przestrzeni parametrów promptu i ACO w routingu agentów, mierzy, dlaczego te klasyczne algorytmy pasują do ery LLM i kiedy nie.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## Problem

Masz prompt, który osiąga 62% w ewaluacji zadania. Chcesz go poprawić. Naiwnym podejściem jest ręczne dostrajanie bez gradientu, które źle się skaluje. Uczenie przez wzmacnianie potrzebuje sygnałów nagrody i wystarczającej liczby przebiegów do treningu. Propagacja wsteczna przez prompt nie jest tak naprawdę możliwa — prompt to dyskretny ciąg znaków, a nie różniczkowalny parametr.

Klasyczna inspirowana biologią optymalizacja — PSO dla ciągłych przestrzeni poszukiwań, ACO dla wyboru ścieżki — została zaprojektowana właśnie dla tego reżimu: bezgradientowa, oparta na populacji, tania na ocenę. Połącz je z LLM dla bezgradientowego kroku poszukiwań, a otrzymasz zaskakująco praktyczny optymalizator.

Te same wzorce odnoszą się do *routingu* agentów w systemach wieloagentowych. Ślad feromonowy w stylu ACO rejestruje, który agent działał najlepiej na którym typie zadania, pozwala routerowi wykorzystywać ślad i wygasza feromony, aby trasy mogły być odkrywane na nowo.

## Koncepcja

### Przypomnienie PSO (Kennedy & Eberhart 1995)

Optymalizacja rojem cząstek: populacja cząstek w ciągłej przestrzeni poszukiwań. Każda cząstka ma pozycję `x_i` i prędkość `v_i`. Każda iteracja:

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
oceń dopasowanie(x_i)
aktualizuj p_best_i jeśli poprawa
aktualizuj g_best jeśli globalnie najlepszy
```

Gdzie `p_best` to najlepsza pozycja cząstki, `g_best` to najlepsza pozycja roju, `w, c1, c2` to wagi inercji + poznawcza + społeczna, `r1, r2` to czynniki losowe.

### PSO na wyjściach LLM — LMPSO

arXiv:2504.09247 adaptuje PSO dla strukturalnych wyjść generowanych przez LLM (wyrażenia matematyczne, programy). Każda cząstka to wyjście kandydujące. Prędkość to *prompt* opisujący, jak zmodyfikować bieżące wyjście w kierunku osobistego/globalnego najlepszego. LLM generuje nowe wyjście z promptu prędkości. „Inercja" prędkości to prompt taki jak „wprowadź małe, stopniowe zmiany."

Działa to dobrze, gdy:
- Wyjście jest strukturalne (do sparsowania, do oceny).
- Dopasowanie jest automatyczne (przebiegi testowe, ocena arytmetyczna).
- Populacja jest mała (~10-30 cząstek), więc całkowita liczba wywołań LLM pozostaje zarządzalna.

Nie działa dobrze, gdy dopasowanie wymaga przeglądu ludzkiego — koszt na iterację staje się zaporowy.

### Model Swarms

arXiv:2410.11163 przenosi PSO z warstwy wyjściowej do warstwy *modelu*. Każda „cząstka" to ekspert LLM (parametry). Rój przesuwa parametry w kierunku zbiorowego optimum poprzez aktualizację bezgradientową. Raportowany: średni zysk 13.3% w stosunku do 12 baseline'ów na 9 zbiorach danych, przy zaledwie 200 instancjach na iterację.

Kluczowym spostrzeżeniem jest to, że modele eksperckie LLM są już blisko siebie we wspólnej rozmaitości parametrów (wagi adapterów, delty LoRA). PSO na tej niskowymiarowej podprzestrzeni jest tanie i efektywne.

### Przypomnienie ACO (Dorigo 1992)

Optymalizacja kolonią mrówek: mrówki przemierzają graf; każda ścieżka ma ślad feromonowy. Prawdopodobieństwa ruchu mrówek są ważone siłą feromonu. Mrówki, które kończą zadanie, odkładają feromon proporcjonalnie do jakości rozwiązania. Feromon zanika w czasie.

### AMRO-S — ACO dla routingu agentów

arXiv:2603.12933 używa ACO do routingu wieloagentowego. Każdy typ zadania to „cel"; każdy agent to możliwa trasa. Feromony wzmacniają trasy, które produkują dobre wyniki. Kluczowe wkłady:

- **Interpretowalne dowody routingu.** Siła feromonu to czytelny dla człowieka sygnał.
- **Asynchroniczna aktualizacja z bramką jakości.** Feromony aktualizują się dopiero po przejściu kontroli jakości, oddzielając wnioskowanie od uczenia się.
- **4.7x przyspieszenie** w benchmarku routingu wieloagentowego.

Bramka jakości ma znaczenie: bez niej, szybkie, ale błędne agenty gromadzą feromon, a system blokuje się na złych trasach.

### Kiedy używać PSO / ACO dla LLM

**Użyj PSO, gdy:**
- Przestrzeń poszukiwań jest ciągła lub mapuje się na ciągłe parametry (embeddingi promptów, wagi LoRA, numeryczne parametry generacji).
- Dopasowanie jest tanie i automatyczne.
- Populacja może być mała (10-30).

**Użyj ACO, gdy:**
- Masz problem routingu lub wyboru ścieżki.
- Decyzje wzmacniają się w czasie (te same typy zadań powracają).
- Potrzebujesz interpretowalnych dowodów dla decyzji routingu.

**Nie używaj żadnego z nich, gdy:**
- Dopasowanie wymaga przeglądu ludzkiego (zbyt kosztowne na iterację).
- Przestrzeń poszukiwań jest dyskretna i kombinatoryczna w sposób, którego PSO nie obejmuje (użyj algorytmów genetycznych).
- Decyzje w czasie rzeczywistym wymagają ścisłego opóźnienia (PSO/ACO zbiegają się wolno względem heurystyk jednoprzebiegowych).

### Dlaczego inspirowane biologią wciąż wygrywa

Metody gradientowe potrzebują różniczkowalnych sygnałów. Wyjścia LLM i decyzje routingu nie są trywialnie różniczkowalne. Metody pseudogradientowe (routery uczone przez wzmacnianie, strojniki promptów w stylu DPO) działają, ale potrzebują drogiego treningu.

PSO i ACO potrzebują tylko funkcji *oceniającej*. Jeśli możesz ocenić wyjście kandydujące lub decyzję routingu, możesz optymalizować w przestrzeni. To sprawia, że próg stosowalności jest znacznie niższy.

### Praktyczne ograniczenia

- **Budżet populacji.** N cząstek × T iteracji × koszt na ocenę. Dla ewaluacji LLM przy ~$0.02 / wywołanie, PSO z 20 cząstkami uruchomione na 50 iteracji kosztuje ~$20. Planuj odpowiednio.
- **Eksploracja vs. eksploatacja.** Tempo zaniku feromonu i inercja PSO to kompromis; zbyt szybki zanik → zapominanie rozwiązań; zbyt wolny → utknięcie na wczesnych lokalnych optimach.
- **Katastroficzny dryft.** Oba algorytmy mogą zbiec się, a potem rozbiec, jeśli krajobraz dopasowania się zmieni (nowa dystrybucja danych). Monitoruj stabilność najlepszego dopasowania.

## Build It

`code/main.py` implementuje:

- `LMPSO` — PSO nad numerycznymi parametrami promptów (temperatura, wagi top_k). „Generacja LLM" każdej cząstki jest symulowana jako skryptowana funkcja dopasowania. Uruchamia algorytm na 30 iteracji i pokazuje zbieżność g_best.
- `AMRO_S` — routing w stylu ACO. 3 agentów, 4 typy zadań, macierz feromonów, 100 routowanych zadań. Wyświetla rozkład (typ_zadania → wybór agenta) w czasie, aby pokazać tworzenie się śladów.
- Porównanie: routing losowy vs. routing ACO na tym samym strumieniu zadań. Mierzy jakość i opóźnienie.

Uruchomienie:

```
python3 code/main.py
```

Oczekiwane wyjście:
- LMPSO: dopasowanie g_best poprawia się od losowego do blisko optymalnego w ciągu 30 iteracji.
- AMRO-S: tabela feromonów stabilizuje się na właściwym agencie dla każdego typu zadania; routing ACO bije losowy o ~30-40% na jakości i także redukuje opóźnienie (mniej ponowień).

## Use It

`outputs/skill-swarm-optimizer.md` pomaga wybrać między PSO, ACO, algorytmami genetycznymi a optymalizatorami gradientowymi dla problemów optymalizacji LLM / agentów.

## Ship It

- **Zacznij od małego.** 10-20 cząstek, 20-50 iteracji. Skaluj tylko wtedy, gdy krzywa zbieżności pokazuje wyraźny zysk.
- **Loguj feromony lub g_best na iterację.** Debugowanie optymalizatorów rojowych bez śladu jest bolesne.
- **Aktualizacje z bramką jakości.** Zwłaszcza dla routingu ACO: szybcy, ale błędni agenci nie mogą gromadzić feromonu.
- **Zresetuj zanik przy zmianie dystrybucji.** Gdy twoja dystrybucja ewaluacyjna się zmienia, starsze feromony są nieaktualne; zresetuj lub tymczasowo podwój tempo zaniku.
- **Ogranicz koszt na iterację.** Emituj metrykę kosztu na iterację. PSO, które kosztuje $500 / iterację i zyskuje 0.5%, nie jest wdrażalne.

## Ćwiczenia

1. Uruchom `code/main.py`. Zaobserwuj zbieżność LMPSO. Zmieniaj rozmiar populacji 5, 10, 20, 50. Przy jakim rozmiarze czas do zbieżności się nasyca?
2. Zaimplementuj eksperyment „katastroficznego dryftu": po iteracji 30 zmień funkcję dopasowania. Jak szybko PSO się adaptuje? Czy resetowanie `p_best` pomaga?
3. Dodaj bramkę jakości do AMRO-S: odkładanie feromonu tylko dla przebiegów z wynikiem ewaluacji > 0.7. Jak zmienia to zbieżność w porównaniu z wersją bez bramki?
4. Przeczytaj LMPSO (arXiv:2504.09247). Odnieś „prędkość jako prompt" z artykułu do swojej numerycznej prędkości. Co jest tracone w symulacji, a co zachowane?
5. Przeczytaj AMRO-S (arXiv:2603.12933). Zaimplementuj odseparowaną „szybką ścieżkę wnioskowania" z asynchroniczną aktualizacją feromonów. Jak zmienia to opóźnienie systemu pod stałym obciążeniem?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| PSO | „Optymalizacja rojem cząstek" | Kennedy-Eberhart 1995. Optymalizator bezgradientowy oparty na populacji. |
| ACO | „Optymalizacja kolonią mrówek" | Dorigo 1992. Optymalizacja ścieżek/tras przez ślady feromonowe. |
| LMPSO | „PSO z generacją LLM" | arXiv:2504.09247. Prędkość to prompt; LLM produkuje kandydatów. |
| Model Swarms | „PSO na wagach eksperckich" | arXiv:2410.11163. Aktualizacja bezgradientowa w podprzestrzeni parametrów modelu. |
| AMRO-S | „ACO dla routingu agentów" | arXiv:2603.12933. Macierz feromonów nad typ_zadania × agent. |
| p_best / g_best | „Osobiste / globalne najlepsze" | Najlepsze rozwiązania znalezione dla cząstki i całego roju. |
| Feromon | „Pamięć routingu" | Siła na krawędzi; zanika w czasie; odkładany przy jakości. |
| Aktualizacja z bramką jakości | „Ucz się tylko z dobrych przebiegów" | Odkładanie feromonu warunkowane kontrolą jakości. |
| Katastroficzny dryft | „Zmiana dystrybucji" | Krajobraz dopasowania się zmienia; stare p_best i feromony stają się nieaktualne. |

## Dalsza Literatura

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — artykuł PSO z 1995 roku
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — podstawy ACO z 1992 roku
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — PSO dla strukturalnych wyjść LLM
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — PSO w podprzestrzeni wag modelu
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — routing napędzany feromonami z bramką jakości