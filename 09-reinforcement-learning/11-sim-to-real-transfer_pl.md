# Transfer Symulacja-Rzeczywistość

> Polityka wytrenowana w symulatorze, która zawodzi na sprzęcie, to polityka, która zapamiętała symulator. Randomizacja domeny, adaptacja domeny i identyfikacja systemu to trzy narzędzia, dzięki którym wyuczone sterowniki pokonują przepaść rzeczywistości.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 minutes

## Problem

Trenowanie prawdziwego robota jest wolne, niebezpieczne i drogie. Dwunożny robot potrzebuje milionów epizodów treningowych, aby nauczyć się chodzić; prawdziwy robot, który przewróci się nawet raz, niszczy sprzęt. Symulacja daje nieograniczone resetowanie, deterministyczną powtarzalność, równoległe środowiska i brak fizycznych uszkodzeń.

Ale symulatory są niedokładne. Łożyska mają więcej tarcia niż modele MuJoCo. Kamery mają dystorsję obiektywu, której symulator nie zawiera. Silniki mają opóźnienia, luz i nasycenie, które 99% modeli symulacyjnych pomija. Wiatr, kurz i zmienne oświetlenie sabotują politykę wytrenowaną na sterylnym renderowaniu. **Luka rzeczywistości** — systematyczna różnica między rozkładem symulacji a rozkładem rzeczywistym — to centralny problem wdrożonego RL w robotyce.

Potrzebujesz polityki, która jest *odporna na przesunięcie rozkładu symulacja-rzeczywistość*. Trzy historyczne podejścia: randomizuj symulator (randomizacja domeny), dostosuj politykę z odrobiną rzeczywistych danych (adaptacja domeny / dostrajanie) lub zidentyfikuj parametry rzeczywistego systemu i dopasuj je (identyfikacja systemu). W 2026 dominujący przepis łączy wszystkie trzy z masywnie równoległą symulacją (Isaac Sim, Isaac Lab, Mujoco MJX na GPU).

## Koncepcja

![Trzy reżimy sym2real: randomizacja domeny, adaptacja, identyfikacja systemu](../assets/sim-to-real.svg)

**Randomizacja Domeny (DR).** Tobin i in. 2017, Peng i in. 2018. Podczas treningu, randomizuj każdy parametr symulacji, który może się różnić na prawdziwym robocie: masy, współczynniki tarcia, wzmocnienia PD silnika, szum czujników, pozycja kamery, oświetlenie, tekstury, modele kontaktu. Polityka uczy się rozkładu warunkowego "w którym symulatorze jestem dzisiaj" i generalizuje na cały zakres. Jeśli prawdziwy robot mieści się w kopercie treningowej, polityka działa.

- **Zaleta:** brak potrzeby rzeczywistych danych. Jeden przepis, wiele robotów.
- **Wada:** przetrenowany na zbyt dużym zakresie randomizacji produkuje "uniwersalną", ale nadmiernie ostrożną politykę. Zbyt dużo szumu ≈ zbyt dużo regularyzacji.

**Identyfikacja Systemu (SI).** Dopasuj parametry symulatora do danych ze świata rzeczywistego przed treningiem. Jeśli możesz zmierzyć tarcie w stawie ramienia na prawdziwym robocie, wprowadź to do symulacji. Następnie trenuj politykę, która oczekuje tych wartości. Wymaga dostępu do rzeczywistego systemu, ale bezpośrednio zmniejsza lukę rzeczywistości.

- **Zaleta:** precyzyjny, niskoszumny cel treningowy.
- **Wada:** resztkowy błąd modelu jest niewidoczny dla polityki; małe niezidentyfikowane efekty (np. martwa strefa silnika) wciąż załamują wdrożenie.

**Adaptacja Domeny.** Trenuj w symulacji, dostrój z małą ilością rzeczywistych danych. Dwa warianty:

- **Real2Sim2Real:** naucz się resztkowego symulatora `f(s, a, z) - f_sim(s, a)` używając rzeczywistych przebiegów, trenuj w poprawionej symulacji. Zamyka lukę bez dużej ilości rzeczywistych danych.
- **Adaptacja obserwacji:** trenuj politykę, która mapuje rzeczywiste obserwacje → obserwacje podobne do symulacji przez wyuczony ekstraktor cech (np. GAN pixel-do-piksela). Sterownik pozostaje w symulacji.

**Uczenie uprzywilejowane / nauczyciel-uczeń.** Miki i in. 2022 (ANYmal czworonóg). Trenuj *nauczyciela* w symulacji, który ma dostęp do uprzywilejowanych informacji (prawdziwe tarcie, wysokość terenu, dryf IMU). Dystyluj *ucznia*, który widzi tylko obserwacje z rzeczywistych czujników. Uczeń uczy się wnioskować o uprzywilejowanych cechach z historii, odporny na parametry fizyczne.

**Masywnie równoległa symulacja.** 2024–2026. Isaac Lab, Mujoco MJX, Brax — wszystkie uruchamiają tysiące równoległych robotów na jednym GPU. PPO z 4096 równoległymi humanoidami zbiera lata doświadczeń w godzinach. "Luka rzeczywistości" zmniejsza się, gdy rozkład treningowy się poszerza; DR staje się prawie darmowe, gdy każde z tych 4096 środowisk ma inne zrandomizowane parametry.

**Przepis na świat rzeczywisty w 2026 (przykład chodzenia czworonoga):**

1. Masywnie równoległa symulacja z zrandomizowaną grawitacją, tarciem, wzmocnieniami silnika, obciążeniem.
2. Nauczyciel trenowany z uprzywilejowanymi informacjami (mapa terenu, prawdziwa prędkość ciała).
3. Uczeń dystylowany z nauczyciela używający tylko propriocepcji (enkodery stawów nóg).
4. Opcjonalna adaptacja obserwacji przez autoenkoder na rzeczywistym IMU.
5. Wdrożenie. Zero-shot w 10+ środowiskach. Jeśli zawiedzie, wykonaj minuty dostrajania w świecie rzeczywistym z PPO ograniczonym bezpieczeństwem.

## Zbuduj To

Kod tej lekcji to mała demonstracja randomizacji domeny na GridWorld z *zaszumionymi* przejściami. Trenujemy politykę, która doświadcza zrandomizowanych prawdopodobieństw ześlizgu w "symulacji" i oceniamy na "rzeczywistości" z poziomem ześlizgu, którego nigdy nie widziała podczas treningu. Kształt mapuje się bezpośrednio na transfer MuJoCo-do-sprzętu.

### Krok 1: sparametryzowany sym

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` to parametr, który symulator udostępnia. W prawdziwej robotyce może to być tarcie, masa, wzmocnienie silnika — cokolwiek, co zmienia się między symulacją a rzeczywistością.

### Krok 2: trenuj z DR

Na początku każdego epizodu, próbkuj `slip ~ Uniform[0.0, 0.4]`. Trenuj PPO / Q-learning / cokolwiek. Rób to przez wiele epizodów.

### Krok 3: oceń zero-shot na "rzeczywistych" ześlizgach

Oceń na `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Pierwsze cztery mieszczą się w zakresie wsparcia treningu; `0.5` i `0.7` są poza. Polityka trenowana DR powinna pozostać blisko optymalnej wewnątrz wsparcia i degradować się łagodnie na zewnątrz. Polityka trenowana na stałym ześlizgu będzie krucha poza swoim ześlizgiem treningowym.

### Krok 4: porównaj z wąskim treningiem

Trenuj drugą politykę tylko z `slip = 0.0`. Oceń na tym samym zakresie `slip`. Powinieneś zobaczyć katastrofalny spadek, gdy tylko rzeczywisty ześlizg > 0.

## Pułapki

- **Zbyt dużo randomizacji.** Trenuj na `slip ∈ [0, 0.9]`, a twoja polityka będzie tak bardzo unikająca ryzyka, że nigdy nie spróbuje optymalnej ścieżki. Dopasuj *oczekiwany* rozkład rzeczywistego świata, a nie "cokolwiek może się zdarzyć."
- **Zbyt mało randomizacji.** Trenuj na wąskim wycinku, a polityka w ogóle nie będzie mogła generalizować. Użyj adaptacyjnego programu nauczania (Automatyczna Randomizacja Domeny), który poszerza rozkład w miarę poprawy polityki.
- **Błędnie zidentyfikowana przestrzeń parametrów.** Randomizuj niewłaściwą rzecz (odcień kamery, gdy rzeczywista luka to opóźnienie silnika), a DR nie pomoże. Najpierw profiluj prawdziwego robota.
- **Wyciek uprzywilejowanych informacji.** Nauczyciel używający globalnego stanu dla akcji, a nie tylko obserwacji, może wyprodukować ucznia, który nie może go dogonić. Upewnij się, że polityka nauczyciela jest możliwa do zrealizowania przez ucznia na podstawie historii obserwacji.
- **Niepowodzenie transferu sym-sym.** Jeśli twoja polityka nie jest odporna na trudniejszy wariant symulacji, nie będzie odporna na rzeczywisty świat. Zawsze testuj na wstrzymanym wariancie symulacji przed wdrożeniem.
- **Brak osłony bezpieczeństwa w świecie rzeczywistym.** Polityka, która działa w symulacji i "działa w rzeczywistości" bez niskopoziomowej tarczy bezpieczeństwa, wciąż może zniszczyć sprzęt. Dodaj ograniczenia prędkości, ograniczenia momentu obrotowego, ograniczenia stawów w nienauczonym sterowniku.

## Użyj Tego

Stos sym2real w 2026:

| Domena | Stos |
|--------|-------|
| Lokomocja czworonożna (ANYmal, Spot, humanoid) | Isaac Lab + DR + uprzywilejowany nauczyciel / uczeń |
| Manipulacja (zręczne dłonie, pick-and-place) | Isaac Lab + DR + DR-GAN dla wizji |
| Jazda autonomiczna | CARLA / NVIDIA DRIVE Sim + DR + rzeczywiste dostrajanie |
| Wyścigi dronów | RotorS / Flightmare + DR + adaptacja online |
| Manipulacja palcami/w dłoni | OpenAI Dactyl (DR w niespotykanej skali) |
| Ramiona przemysłowe | MuJoCo-Warp + SI + małe rzeczywiste dostrajanie |

Dla sterowania we wszystkich skalach, przepis pracy jest spójny: dopasuj symulator najlepiej jak potrafisz, randomizuj to, czego nie możesz dopasować, trenuj ogromne polityki, dystyluj, wdróż z tarczą bezpieczeństwa.

## Dostarcz To

Zapisz jako `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Ćwiczenia

1. **Łatwe.** Trenuj agenta Q-learning na GridWorld ze stałym ześlizgiem (slip=0.0). Oceń na slip ∈ {0.0, 0.1, 0.3, 0.5}. Narysuj zwrot vs slip.
2. **Średnie.** Trenuj agenta DR Q-learning próbkującego `slip ~ Uniform[0, 0.3]`. Oceń ten sam zakres. Ile zyskuje DR przy slip=0.5 (poza rozkładem)?
3. **Trudne.** Zaimplementuj program nauczania: zacznij od slip=0.0, poszerzaj zakres DR za każdym razem, gdy polityka osiągnie 90% optimum. Zmierz całkowitą liczbę kroków środowiska do osiągnięcia slip=0.3 zero-shot vs stała wartość bazowa DR.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| Luka rzeczywistości | "Różnica sym-real" | Przesunięcie rozkładu między fizyką/czujnikami treningu a wdrożenia. |
| Randomizacja domeny (DR) | "Trenuj na losowych symulatorach" | Randomizuj parametry symulacji podczas treningu, aby polityka generalizowała. |
| Identyfikacja systemu (SI) | "Zmierz rzeczywiste i dopasuj sym" | Oszacuj rzeczywiste parametry fizyczne; ustaw symulator, aby pasował. |
| Adaptacja domeny | "Dostrój na rzeczywistych danych" | Małe rzeczywiste dostrajanie po treningu symulacyjnym; może adaptować obserwacje lub dynamikę. |
| Uprzywilejowane informacje | "Prawdziwe dane dla nauczyciela" | Informacje dostępne tylko w symulacji; uczeń musi je wnioskować z historii obserwacji. |
| Nauczyciel/uczeń | "Dystyluj uprzywilejowane → obserwowalne" | Nauczyciel trenowany ze skrótami; uczeń uczy się naśladować bez nich. |
| ADR | "Automatyczna Randomizacja Domeny" | Program nauczania, który poszerza zakresy DR w miarę poprawy polityki. |
| Real2Sim | "Zamknij lukę rzeczywistymi danymi" | Naucz się residuum, aby symulator naśladował rzeczywiste przebiegi. |

## Dalsza Lektura

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — oryginalna praca DR (wizja dla robotyki).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR dla dynamiki, lokomocja czworonożna.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, ADR w skali.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — nauczyciel-uczeń dla ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — masywnie równoległa symulacja napędzająca wdrożenia 2025–2026.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — metoda programu nauczania ADR.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — ramy Dyna (użyj modelu do planowania + wykonań), które leżą u podstaw nowoczesnych pipeline'ów sym2real.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — taksonomia metod sym2real z wynikami benchmarkowymi.