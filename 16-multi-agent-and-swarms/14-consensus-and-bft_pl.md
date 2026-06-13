# Konsensus i Bizantyjska Tolerancja Błędów dla Agentów

> Klasyczny BFT z systemów rozproszonych spotyka stochastyczne LLM. W latach 2025-2026 pojawiły się trzy kierunki badawcze: **CP-WBFT** (arXiv:2511.10400) waży każdy głos przez sondę pewności; **DecentLLMs** (arXiv:2507.14928) idzie bez lidera z równoległymi propozycjami pracowników i agregacją mediany geometrycznej; **WBFT** (arXiv:2505.05103) łączy ważone głosowanie z Hierarchicznym Grupowaniem Strukturalnym, aby podzielić węzły na Core i Edge. Uczciwy wynik empiryczny z "Can AI Agents Agree?" (arXiv:2603.01213) jest taki, że nawet zgoda skalarna jest dziś krucha — pojedynczy oszukańczy agent może skompromitować Mixture-of-Agents. BFT jest konieczny, ale niewystarczający. Ta lekcja buduje minimalny protokół BFT, wstrzykuje trzy ataki specyficzne dla agentów (kłamstwo bizantyjskie, konformizm pochlebczy, monokulturę skorelowanego błędu) i mierzy, jak każdy wariant konsensusu sobie radzi.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Problem

Masz N agentów LLM, każdy produkuje swoją odpowiedź. Są w niezgodzie. Głosowanie większościowe wybiera tę złą, ponieważ dwóch agentów jest skorelowanych (ten sam model bazowy, te same dane treningowe, te same tryby awarii). Trzeci agent jest przypadkowo błędny w nowatorski sposób — więc większość jest fałszywą większością.

Teraz dodaj oszukańczego agenta: kłamie celowo. Albo agenta pochlebcę: zgadza się z tym, kto mówił ostatnio. W klasycznym BFT założenie jest takie, że węzły bizantyjskie stanowią ułamek `f < n/3` i zachowują się arbitralnie. Rzeczywistość 2026 roku jest taka, że węzły LLM są stochastyczne nawet gdy uczciwe, skorelowane między modelami i pod wpływem wyników innych. Nie możesz traktować ich jako niezależnych wyborców Bernoulliego.

Klasyczny BFT (PBFT, 1999) nie jest błędny — jest niekompletny. Radzi sobie z arbitralnym odwracaniem bitów. Nie radzi sobie z "trzej uczciwi agenci dzielą halucynację, ponieważ dzielą dane treningowe." Ta lekcja buduje na fundamencie PBFT i nakłada trzy adaptacje z lat 2025-2026.

## Koncepcja

### Co daje ci klasyczny BFT

Praktyczna Bizantyjska Tolerancja Błędów (Castro i Liskov, OSDI 1999) toleruje `f < n/3` węzłów bizantyjskich. Protokół ma trzy fazy (pre-prepare, prepare, commit) i dwa prymitywy (podpisane wiadomości, certyfikaty kworum). Zgoda na pojedynczej wartości wśród `n >= 3f + 1` węzłów uczciwych lub złośliwych.

Gwarancje są silne, ale zakładają:

1. **Niezależne awarie.** Bizantyjczycy nie koordynują.
2. **Uczciwe węzły są naprawdę uczciwe.** Poprawność uczciwych wyników nie jest problemem; protokół tylko wyrównuje niezgodności.
3. **Pytanie ma odpowiedź zgodną z prawdą.** Konsensus co do błędnego faktu to wciąż konsensus.

Agenci LLM naruszają wszystkie trzy. Dwaj agenci uruchamiający ten sam model bazowy dzielą awarie. "Uczciwy" LLM wciąż halucynuje. A w przypadku pytań niejednoznacznych "prawda" jest tym, co agenci zdecydują — nie ma zewnętrznego wyroczni.

### Trzy ataki specyficzne dla LLM

**Kłamstwo bizantyjskie.** Jeden agent celowo podaje błędną odpowiedź. Klasyczny BFT radzi sobie z tym, jeśli `f < n/3`.

**Konformizm pochlebczy.** Jeden agent czyta odpowiedzi innych przed głosowaniem i dostosowuje się do tego, kto mówił ostatnio. Niezłośliwy, ale koreluje z najgłośniejszym głosem. Klasyczny BFT temu nie zapobiega, ponieważ agent przechodzi każdą kontrolę podpisu.

**Monokultura skorelowanego błędu.** Trzej agenci dzielą model bazowy. Halucynują tę samą błędną odpowiedź. Większość jest błędna. Klasyczny BFT nie pomaga, ponieważ wszyscy trzej "uczciwie" się zgadzają.

### Odpowiedzi z lat 2025-2026

**CP-WBFT** (arXiv:2511.10400) — Ważony BFT z Sondą Pewności. Każdy głosujący dołącza sondę pewności do swojej odpowiedzi (samozgłoszone prawdopodobieństwo lub przewidywanie osobnego modelu kalibracyjnego). Wagi głosów skalują się z pewnością. Zgłoszona +85.71% poprawa BFT na pełnych grafach. Mitygacja dla: konformizmu pochlebcza (konformistyczni agenci mają tendencję do niskiej pewności na swoim dobrowolnym stanowisku).

**DecentLLMs** (arXiv:2507.14928) — Bez lidera. Agenci-pracownicy proponują równolegle, agenci-ewaluatorzy oceniają propozycje, ostateczna odpowiedź to geometryczna mediana ocenionych pozycji. Solidny, gdy `f < n/2`. Mitygacja dla: kłamstwa bizantyjskiego i skorelowanych błędów (geometryczna mediana jest odporna na wartości odstające i ciągnie w kierunku gęstego klastra, a nie średniej obciążonej modelem).

**WBFT** (arXiv:2505.05103) — Ważony BFT z Hierarchicznym Grupowaniem Strukturalnym. Wagi głosów są przypisywane przez jakość odpowiedzi plus wynik zaufania uczony z historii. Grupuj agentów w Core i Edge; agenci Core muszą najpierw osiągnąć konsensus, agenci Edge podążają. Mitygacja dla: skalowalności (konsensus Core jest mały i szybki) i częściowo dla monokultury (Core może być wybrany dla różnorodności).

### Empiryczne: "Can AI Agents Agree?" (arXiv:2603.01213)

Publikacja mierzy zgodę skalarną (agenci LLM zgadzający się na pojedynczej wartości numerycznej) na wielu modelach granicznych. Wynik jest niewygodny:

- Nawet bez przeciwników, agenci LLM nie zgadzają się w kwestiach skalarnych w tempie powyżej 30% na wielu benchmarkach.
- Pojedynczy agent, który przyjmuje oszukańczą personę, może odciągnąć konsensus Mixture-of-Agents o 40+ punktów procentowych od uczciwej linii bazowej.
- Wskaźniki niezgodności korelują z różnorodnością modeli — heterogeniczne zespoły nie zgadzają się bardziej niż homogeniczne (dobrze: nieskorelowane błędy), ale także dryfują wolniej (źle: dłuższy czas do zgody).

Wniosek: BFT daje ci mechanizmy do wyrównywania wyników, ale nie mówi ci, czy wyrównany wynik jest prawidłowy. Połącz z weryfikacją (Faza 16 · 08 specjalizacja ról), różnorodnością (Faza 16 · 15 warianty debaty) i agentami ewaluatorami (Faza 16 · 24 benchmarki).

### Podstawowy protokół, okrojony

Minimalna runda BFT dla agentów LLM:

```
1. zadanie przychodzi; każdy agent i produkuje odpowiedź a_i
2. każdy agent dołącza sondę pewności c_i w [0, 1]
3. agregator zbiera (a_i, c_i) od wszystkich n agentów
4. agregator grupuje według klastra semantycznego (odpowiedniki odpowiedzi)
5. agregator oblicza wagę dla każdego klastra C:
      w(C) = suma_{i in C} c_i
6. zwycięzca = klaster z maksymalną wagą, jeśli max > próg * suma(c_i)
    w przeciwnym razie: ponów lub eskaluj
7. klastry mniejszościowe logowane z proweniencją do audytu post-hoc
```

Krok grupowania semantycznego to zwrot specyficzny dla LLM. Dwie odpowiedzi "badanie raportuje 4.2%" i "4.2% poprawy" to ten sam klaster. Naiwne sprawdzenie równości stringów by to przeoczyło. W produkcji użyj taniego modelu osadzania lub jawnej kanonizacji.

### Dostrajanie progu

Parametr `threshold` decyduje, kiedy zaakceptować, a kiedy ponowić. Zbyt niski: akceptujesz słabe większości. Zbyt wysoki: nigdy niczego nie akceptujesz. Zakres empiryczny: 0.5-0.67 dla `n=5-7` agentów, wyższy dla mniejszego `n`. Poniżej progu, eskaluj do człowieka lub do innego zespołu agentów.

### Gdzie konsensus nie pomaga

- **Pytania niejednoznaczne.** Jeśli pytanie nie ma prawdy obiektywnej, konsensus jest opinią. Nazwij to tak.
- **Pytania złożone.** "Napisz kod i wyjaśnij go" — dwie odpowiedzi. Głosuj nad każdą niezależnie.
- **Wielorundowe działanie adversarialne.** Jeśli agenci mogą obserwować poprzednie rundy i naśladować (debata Du 2023), zaczynają zgadzać się ze sobą niezależnie od prawdy. Ogranicz rundy (2-3 zazwyczaj).

## Zbuduj To

`code/main.py` implementuje:

- `AgentVoter` — skryptowa polityka z (odpowiedź, pewność).
- `MajorityVote` — klasyczna pluralność.
- `CPWBFT` — ważone głosowanie z grupowaniem semantycznym.
- `DecentLLMs` — agregacja mediany geometrycznej na ocenionych propozycjach.
- `Scenario` — uruchamia każdy agregator pod trzema wzorcami ataku.

Zaimplementowane wzorce ataku:

1. `byzantine`: jeden agent kłamie z wysoką pewnością.
2. `sycophancy`: jeden agent kopiuje pierwszą odpowiedź, którą widzi, z dopasowaną pewnością.
3. `monoculture`: trzech agentów dzieli błędną odpowiedź (skorelowany błąd) z umiarkowaną pewnością.

Uruchom:

```
python3 code/main.py
```

Oczekiwane wyjście: tabela (atak, agregator) -> ostateczna odpowiedź, z poprawną odpowiedzią wyróżnioną. Pluralność zawodzi w przypadku monokultury. Ważenie pewnością CPWBFT łagodzi pochlebstwo. Geometryczna mediana DecentLLMs ciągnie w kierunku uczciwego klastra, gdy monokultura stanowi mniej niż połowę populacji.

## Użyj Tego

`outputs/skill-consensus-designer.md` projektuje protokół konsensusu dla zespołu wieloagentowego: metodę grupowania, ważenie, próg i politykę eskalacji dla rund poniżej progu.

## Wdróż To

Przed wdrożeniem jakiegokolwiek mechanizmu konsensusu:

- **Przetestuj atakami co najmniej trzema wzorcami** powyżej. Twój protokół powinien zawodzić przewidywalnie, a nie po cichu.
- **Loguj każdy klaster mniejszościowy** z proweniencją. Klastry mniejszościowe to twój system wczesnego ostrzegania o skorelowanych błędach.
- **Wymuszaj ograniczone rundy.** Żadnego "debataj aż do zgody" — to nagradza pochlebstwo.
- **Oddziel zgodę od poprawności.** Wynik konsensusu trafia do weryfikatora; weryfikator jest niezależny od zespołu.
- **Monitoruj wskaźnik zgody.** Ostry wzrost oznacza błąd konformizmu; ostry spadek oznacza dryf modelu.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że pluralność zawodzi w przypadku ataku monokultury, ale CPWBFT częściowo go łagodzi, gdy pewność monokultury jest poniżej 0.7.
2. Dodaj czwarty wzorzec ataku: **ciche wstrzymanie się** — jeden agent odmawia odpowiedzi ("Nie wiem"). Jak każdy agregator powinien traktować wstrzymania? Zaimplementuj swój wybór.
3. Zamień grupowanie semantyczne z kanonizacji stringów na podobieństwo osadzania (użyj dowolnego otwartoźródłowego modelu osadzania). Co dzieje się z atakiem pochlebstwa?
4. Przeczytaj CP-WBFT (arXiv:2511.10400). Zaimplementuj krok kalibracji sondy pewności (osobny model kalibracyjny sprawdza samozgłoszoną pewność każdego agenta). Zmierz wzrost dokładności w scenariuszu monokultury.
5. Przeczytaj "Can AI Agents Agree?" (arXiv:2603.01213). Odtwórz uproszczony eksperyment zgody skalarnej: trzech agentów, jedno pytanie skalarne, prompt oszukańczej persony. Czy CPWBFT lub DecentLLMs to wychwytuje?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|----------------|------------------------|
| BFT | "Bizantyjska tolerancja błędów" | Protokół Castro-Liskov 1999 dla konsensusu z `f < n/3` arbitralnymi awariami. |
| Bizantyjski | "Dowolne złe zachowanie" | Węzeł, który może kłamać, upuszczać wiadomości, cicho zawodzić — wszystko oprócz bezpiecznej awarii. |
| Sonda pewności | "Jak bardzo jesteś pewny?" | Samozgłoszone lub przewidziane przez kalibrator prawdopodobieństwo dołączone do głosu. |
| Grupowanie semantyczne | "Ta sama odpowiedź, inne słowa" | Grupowanie równoważnych odpowiedzi przed liczeniem głosów. |
| Geometryczna mediana | "Solidne centrum" | Punkt minimalizujący sumę odległości do punktów próbki. Odporna na wartości odstające, w przeciwieństwie do średniej. |
| Monokultura | "Ten sam model, te same awarie" | Skorelowane błędy, gdy agenci dzielą dane treningowe lub model bazowy. |
| Konformizm pochlebczy | "Zgadzanie się z głośnym głosem" | Głos agenta skłania się ku temu, kto mówił pierwszy/najgłośniej. |
| Core/Edge | "Hierarchiczny BFT" | Podział WBFT: najpierw mały konsensus Core, węzły Edge podążają. Ogranicza opóźnienie. |

## Dalsza Literatura

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — fundament
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — ważenie głosów przez pewność
- [DecentLLMs — wieloagentowy konsensus bez lidera](https://arxiv.org/abs/2507.14928) — agregacja mediany geometrycznej
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — podział Core/Edge dla ograniczonego opóźnienia
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — kruchość zgody skalarnej i atak oszukańczą personą