# Kryteria Uczciwości — Grupowa, Indywidualna, Kontrfaktyczna

> Trzy rodziny strukturują literaturę uczciwości. Uczciwość grupowa: parytet demograficzny, wyrównane szanse, równość dokładności użycia warunkowego — równe wskaźniki w grupach chronionych średnio. Uczciwość indywidualna (Dwork et al. 2012): podobne osoby otrzymują podobne decyzje; warunek Lipschitza na mapie decyzyjnej. Uczciwość kontrfaktyczna (Kusner et al. 2017): decyzja jest uczciwa względem osoby, jeśli nie zmienia się, gdy wrażliwe atrybuty są kontrfaktycznie zmienione. Teoretyczny wynik 2024 (NeurIPS 2024): istnieje nieodłączny kompromis CF-vs-dokładność; metoda niezależna od modelu przekształca optymalny, ale nieuczciwy predyktor w CF z ograniczoną stratą dokładności. Kontrfaktyki cofające (arXiv:2401.13935, styczeń 2024): nowy paradygmat, który unika wymagania interwencji na prawnie chronionych atrybutach. Filozoficzne pojednanie (ICLR Blogposts 2024): z grafami przyczynowymi, spełnienie pewnych miar uczciwości grupowej pociąga za sobą uczciwość kontrfaktyczną.

**Type:** Learn
**Languages:** Python (stdlib, porównanie trzech kryteriów)
**Prerequisites:** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time:** ~60 minutes

## Learning Objectives

- Wymienić trzy kryteria uczciwości grupowej (parytet demograficzny, wyrównane szanse, równość dokładności użycia warunkowego) i jeden wynik niemożliwości.
- Opisać uczciwość indywidualną przez formułę Lipschitza Dwork et al. 2012.
- Opisać uczciwość kontrfaktyczną i jej zależność od grafu przyczynowego.
- Wyjaśnić kontrfaktyki cofające i dlaczego omijają problem interwencji na chronionym atrybucie.

## The Problem

Lekcja 20 dotyczyła mierzenia uprzedzenia. Lekcja 21 dotyczy definiowania standardu uczciwości, któremu pomiar ma służyć. Trzy rodziny dają strukturalnie różne standardy — model może być grupowo-uczciwy i indywidualnie-nieuczciwy, kontrfaktycznie-uczciwy i grupowo-nieuczciwy. Wybór standardu to decyzja polityczna; żaden standard nie jest uniwersalnie optymalny.

## The Concept

### Uczciwość grupowa

- **Parytet demograficzny.** P(Y=1 | A=a) = P(Y=1 | A=a') dla wszystkich grup. Równe wskaźniki akceptacji.
- **Wyrównane szanse.** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a'). Równe TPR i FPR między grupami.
- **Równość dokładności użycia warunkowego.** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a'). Równe wartości predykcyjne między grupami.

Niemożliwość (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): te trzy nie mogą być spełnione jednocześnie przy nierównych stawkach bazowych.

### Uczciwość indywidualna

Dwork et al. 2012. Mapa decyzyjna f jest indywidualnie uczciwa względem metryki podobieństwa specyficznej dla zadania d, jeśli |f(x) - f(x')| <= L * d(x, x') dla pewnej stałej Lipschitza L. Podobne osoby otrzymują podobne decyzje.

Wymaga zdefiniowania d. Pytanie polityczne, nie statystyczne.

### Uczciwość kontrfaktyczna

Kusner et al. 2017. Decyzja jest kontrfaktycznie uczciwa względem osoby i, jeśli w ramach modelu przyczynowego populacji, decyzja nie zmienia się, gdy wrażliwe atrybuty i są kontrfaktycznie zmienione.

Wymaga przyczynowego DAG. DAG jest wyborem modelowania. Uczciwość kontrfaktyczna jest tylko tak uzasadniona, jak DAG.

### Kompromis CF-vs-dokładność

Teoria NeurIPS 2024: istnieje nieodłączny kompromis między uczciwością kontrfaktyczną a dokładnością predykcyjną. Metoda niezależna od modelu może przekształcić optymalny, ale nieuczciwy predyktor w CF, przy ograniczonym koszcie dokładności. Koszt dokładności zależy od wielkości współczynnika wrażliwego atrybutu w optymalnym nieuczciwym predyktorze.

### Kontrfaktyki cofające

arXiv:2401.13935 (styczeń 2024). Tradycyjne kontrfaktyki wymagają interwencji na wrażliwym atrybucie — "czy decyzja zmieniłaby się, gdyby ta osoba była innej płci." Prawnie jest to problematyczne: chronione atrybuty nie mogą być przedmiotem interwencji w prawie klasyfikacyjnym.

Kontrfaktyki cofające odwracają kierunek: zamiast interweniować na atrybucie, pytają, jaka kombinacja faktycznych cech osoby wyprodukowałaby kontrfaktyczny wynik. To omija prawny zarzut.

### Filozoficzne pojednanie

ICLR Blogposts 2024. Z grafem przyczynowym w ręku, spełnienie pewnych miar uczciwości grupowej pociąga za sobą uczciwość kontrfaktyczną. Trzy rodziny nie są ortogonalne; są różnymi fasetami tej samej struktury przyczynowej.

To nie rozwiązuje twierdzeń o niemożliwości (nierówne stawki bazowe nadal uniemożliwiają jednoczesną uczciwość grupową). Ale pokazuje, że pozorna opozycja między "grupową" a "indywidualną / kontrfaktyczną" jest częściowo artefaktem nie bycia jawnym co do modelu przyczynowego.

### Miejsce w Fazie 18

Lekcja 20 to pomiar uprzedzenia. Lekcja 21 to definicja uczciwości. Lekcja 22 to prywatność (prywatność różnicowa). Lekcja 23 to znakowanie wodne. To lekcje sąsiadujące z alokacją, uzupełniające lekcje sąsiadujące z oszustwem 7-11.

## Use It

`code/main.py` buduje zabawkowy zbiór danych klasyfikacji binarnej z wrażliwym atrybutem i nierównymi stawkami bazowymi. Oblicz parytet demograficzny, wyrównane szanse i równość dokładności użycia warunkowego na prostym klasyfikatorze. Zaobserwuj, że trzy metryki są ze sobą w sprzeczności. Zastosuj przeważanie dla parytetu demograficznego i zaobserwuj jego koszt na pozostałych dwóch.

## Ship It

Ta lekcja produkuje `outputs/skill-fairness-criterion.md`. Dla twierdzenia lub polityki uczciwości, identyfikuje, które kryterium jest twierdzone, czy model może spełnić pozostałe kryteria przy twierdzonych nierównych stawkach bazowych i od jakiego przyczynowego DAG zależy twierdzenie.

## Exercises

1. Uruchom `code/main.py`. Raportuj trzy metryki grupowe na domyślnych danych. Zastosuj przeważanie ukierunkowane na parytet demograficzny i ponownie raportuj.

2. Zaimplementuj metrykę uczciwości indywidualnej Dwork et al. 2012 używając L2 na niewrażliwych cechach. Raportuj, ile par narusza Lipschitza ze stałą L=1.

3. Przeczytaj Kusner et al. 2017. Zbuduj prosty dwu-cechowy przyczynowy DAG dla oceny CV i zidentyfikuj warunek uczciwości kontrfaktycznej, który implikuje.

4. Artykuł o kontrfaktykach cofających z 2024 unika interwencji na chronionych atrybutach. Opisz scenariusz, w którym ma to znaczenie dla zgodności prawnej.

5. Pojednanie ICLR 2024 argumentuje, że uczciwość grupowa i kontrfaktyczna są fasetami tej samej struktury. Wybierz dwa z trzech kryteriów w `code/main.py` i wskaż założenie przyczynowe, które uczyniłoby je równoważnymi.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Parytet demograficzny | "równe wskaźniki" | P(Y=1 | A=a) równe między grupami |
| Wyrównane szanse | "równe TPR/FPR" | Równe wskaźniki prawdziwie pozytywnych i fałszywie pozytywnych między grupami |
| Równość dokładności użycia | "równe PPV/NPV" | Równe wartości predykcyjne między grupami |
| Uczciwość indywidualna | "warunek Lipschitza" | Podobne osoby otrzymują podobne decyzje |
| Uczciwość kontrfaktyczna | "niezmienność przy alteracji przyczynowej" | Decyzja niezmieniona pod kontrfaktyczną alteracją atrybutu |
| Kontrfaktyka cofająca | "wyjaśnij przez faktyczne" | Kontrfaktyka wnioskowana wstecz od wyniku, nie w przód od atrybutu |
| Twierdzenie o niemożliwości | "trzy są w konflikcie" | Chouldechova / KMR 2017: kryteria grupowe wzajemnie wykluczające się przy nierównych stawkach bazowych |

## Further Reading

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — uczciwość indywidualna
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — uczciwość kontrfaktyczna
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — niemożliwość
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — nowy paradygmat dla interwencji na chronionych atrybutach
