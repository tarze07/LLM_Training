# Podążanie za instrukcjami jako sygnał alignmentu

> Każda późniejsza krytyka RLHF argumentuje przeciwko temu potokowi. Zanim zobaczymy, jak presja optymalizacyjna zniekształca proxy, trzeba poznać samo proxy. InstructGPT (Ouyang et al., 2022) zdefiniował architekturę referencyjną: nadzorowane dostrajanie na parach instrukcja-odpowiedź, model nagrody trenowany na rankingowych preferencjach parami oraz PPO względem modelu nagrody z karą KL do polityki SFT. InstructGPT w rozmiarze 1,3B był preferowany nad GPT-3 w rozmiarze 175B. Ten pojedynczy wynik jest powodem, dla którego każde laboratorium graniczne w 2026 roku wciąż dostarcza potok pozatrainingowy w kształcie RLHF.

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 minutes

## Learning Objectives

- Wymień trzy etapy potoku InstructGPT oraz funkcję straty używaną w każdym z nich.
- Wyjaśnij, dlaczego model 1,3B dostrojony do instrukcji pokonał surowy model GPT-3 175B w ocenie preferencji ludzkich.
- Określ, przed czym chroni kara KL w etapie 3 i dlaczego jej usunięcie prowadzi do zachowania faworyzującego dominanty.
- Opisz podatek alignmentu i mechanizm PPO-ptx zastosowany przez Ouyang et al. jako jego łagodzenie.

## The Problem

Wstępnie wytrenowane modele językowe uzupełniają tekst. Nie odpowiadają na pytania. Zapytaj GPT-3 „napisz funkcję w Pythonie, która odwraca listę", a często otrzymasz kolejny prompt, ponieważ większość dystrybucji treningowej to tekst internetowy kontynuujący tekst internetowy. Model działa prawidłowo — tylko zadanie jest niewłaściwe.

Proxy, którego każde poważne laboratorium użyło do rozwiązania tego problemu, to preferencje ludzkie. Dwie odpowiedzi trafiają do oceniającego; oceniający wybiera lepszą; model nagrody uczy się oceniającego. Następnie pętla RL przesuwa politykę w kierunku odpowiedzi, które model nagrody ocenia wysoko. To cała teza InstructGPT w trzech zdaniach. Reszta artykułu to inżynieria.

## The Concept

### Stage 1: supervised fine-tuning (SFT)

Zbierz pary prompt-odpowiedź, w których odpowiedź jest tym, co napisałby życzliwy człowiek. Ouyang et al. użyli 13 tys. promptów od etykieterów i API OpenAI. Dostrój model bazowy na tych danych ze standardową stratą cross-entropy.

Co daje SFT: model teraz odpowiada na pytania, zamiast je kontynuować. Czego nie daje: żadnego sygnału o tym, którą odpowiedź oceniający preferuje, gdy wiele odpowiedzi jest prawdopodobnych.

### Stage 2: reward model (RM)

Dla każdego promptu pobierz K odpowiedzi z modelu SFT. Oceniający rankingu je. Trenuj model nagrody, który punktuje dowolną parę prompt-odpowiedź tak, aby dla par, gdzie `y_w` było preferowane nad `y_l`:

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

Jest to para preferencji Bradleya-Terry'ego. RM jest zwykle inicjalizowany z modelu SFT, zastępując głowę LM głową skalarną.

Modele nagrody są małe: 6B wystarczyło dla InstructGPT 175B. Są też kruche — sekcja 5 artykułu dotyczy głównie zachowań polegających na hakowaniu nagrody, które ujawniły się w małej skali.

### Stage 3: PPO with a KL penalty

Zdefiniuj funkcję celu:

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

Maksymalizuj za pomocą PPO. Człon KL utrzymuje `pi` blisko polityki SFT. Bez niego optymalizator znajduje przykłady kontradyktoryjne — ciągi znaków, które uzyskują wysoką punktację od RM, ponieważ RM nigdy ich nie widział, nie dlatego, że ludzie faktycznie je preferują.

Współczynnik KL `beta` jest najważniejszym hiperparametrem RLHF. Zbyt niski: hakowanie nagrody. Zbyt wysoki: brak poprawy względem SFT.

### The alignment tax

Po RLHF model jest preferowany przez ludzi, ale regresuje na standardowych benchmarkach (SQuAD, HellaSwag, DROP). Ouyang et al. nazywają to podatkiem alignmentu i rozwiązują za pomocą PPO-ptx: domieszanie gradientów z pretreningu do celu RL, aby model nie zapomniał, jak wykonywać zadania downstream, za które nigdy nie był nagradzany.

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx stało się standardem. Anthropic, DeepMind i Meta używają jakiegoś wariantu.

### The result

InstructGPT 1,3B (SFT + RM + PPO-ptx) jest preferowany przez etykieterów nad bazowym GPT-3 175B w około 70% przypadków. Różnica zwiększa się na ukrytych promptach testowych z ruchu produkcyjnego. Dwie rzeczy do odczytania z tej liczby:

1. Alignment to inna oś niż capability. Model 175B miał większe capability; model 1,3B miał większy alignment; etykieterzy preferowali ten z alignmentem.
2. Poziom minimalny capability jest wyznaczony przez model bazowy. Nie można przez RLHF sprawić, by model bazowy poznał fakty, których nigdy nie widział.

### Why this is the reference point for Phase 18

Każda krytyka w późniejszych lekcjach — hakowanie nagrody (Lekcja 2), DPO (Lekcja 3), sykofancja (Lekcja 4), CAI (Lekcja 5), uśpione agenty (Lekcja 7), fałszowanie alignmentu (Lekcja 9) — argumentuje przeciwko pewnej części tego potoku. Hakowanie nagrody atakuje etap 2. DPO scala etapy 2 i 3. CAI zastępuje ludzkiego etykietera. Sykofancja pokazuje, że etykieter jest obciążonym sygnałem. Fałszowanie alignmentu pokazuje, że polityka może obejść etap 3 całkowicie. Nie można zrozumieć żadnej z tych krytyk bez wcześniejszego poznania tego potoku.

## Use It

`code/main.py` symuluje trzy etapy na zabawkowych danych preferencji. Bazowa „polityka" to obciążona moneta nad akcjami {A, B, C}. Etap 1 (SFT) naśladuje działania etykietera na 200 promptach. Etap 2 dopasowuje model nagrody Bradleya-Terry'ego z 500 rankingów parami. Etap 3 wykonuje uproszczoną aktualizację PPO z karą KL do polityki SFT. Możesz obserwować wzrost nagrody, wzrost dywergencji KL i dryf polityki — oraz wyłączyć człon KL, aby zobaczyć hakowanie nagrody w ciągu 50 kroków aktualizacji.

Na co zwrócić uwagę:

- Trajektoria nagrody dla `beta = 0.1` vs `beta = 0.0`.
- KL(pi || pi_SFT) w trakcie treningu.
- Końcowy rozkład akcji w porównaniu z preferencją etykietera.

## Ship It

Ta lekcja produkuje `outputs/skill-instructgpt-explainer.md`. Otrzymując opis potoku RLHF lub abstrakt artykułu, identyfikuje, który z trzech etapów jest modyfikowany, jaka funkcja straty jest używana na każdym etapie oraz czy występuje kara KL lub równoważny regulator.

## Exercises

1. Uruchom `code/main.py`. Ustaw `beta = 0.0` i zgłoś rozkład akcji po 200 krokach PPO. Wyjaśnij w jednym akapicie zachowanie faworyzujące dominanty.

2. Zmodyfikuj model nagrody, aby miał dodatkowe +0.5 dla akcji B (symulowany błąd nagrody). Uruchom PPO z `beta = 0.1`. Czy kara KL zapobiega wykorzystaniu błędu przez politykę? Przy jakiej `beta` wykorzystanie staje się widoczne?

3. Przeczytaj Ouyang et al. (arXiv:2203.02155) Rysunek 1. Odtwórz krzywą preferencji etykietera, uruchamiając PPO dla 1, 5, 20, 100 kroków i mierząc preferencję względem modelu SFT.

4. Sekcja 4.3 artykułu informuje, że InstructGPT 1,3B pokonuje GPT-3 175B w około 70% przypadków. Dlaczego stosunek ten byłby wyższy na ukrytych promptach produkcyjnych niż na promptach etykieterów?

5. Zastąp stratę PPO stratą DPO (Faza 10 · 08) na tych samych danych preferencji. Porównaj końcowy dryf polityki (KL do SFT) i końcową nagrodę. Która metoda dryfuje dalej przy wyrównanej nagrodzie?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| SFT | „dostrajanie instrukcji" | Etap 1: dostrajanie cross-entropy na parach prompt-odpowiedź |
| Reward model | „RM" | Regresor skalarny nad (prompt, odpowiedź) trenowany z Bradley-Terry na etykietach parami |
| Bradley-Terry | „strata preferencji parami" | -log sigmoid(r_w - r_l); redukuje rankingowanie parami do klasyfikacji binarnej |
| KL penalty | „regulator" | `beta * KL(pi || pi_SFT)` — utrzymuje politykę RL blisko kotwicy SFT |
| PPO-ptx | „PPO z domieszką pretreningu" | Dodaje ułamek log-prawdopodobieństwa pretreningu do celu PPO, aby skompensować podatek alignmentu |
| Alignment tax | „regresja po RLHF" | Spadek po RLHF na standardowych benchmarkach, których RLHF nie dotyczył |
| Labeler preference | „prawda podstawowa" | Próbka rankingów ludzkich; RM jest statystycznym proxy dla tego, nie dla „ludzkich wartości" |

## Further Reading

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — artykuł InstructGPT, podstawa każdego potoku RLHF, który nastąpił później
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — poprzednik RLHF do streszczania
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — oryginalne sformułowanie RL opartego na preferencjach
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — rozszerzenie HH potoku InstructGPT od Anthropic