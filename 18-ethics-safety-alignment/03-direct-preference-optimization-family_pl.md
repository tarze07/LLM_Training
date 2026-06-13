# Rodzina Bezpośredniej Optymalizacji Preferencji

> Rafailov et al. (2023) pokazali, że optimum RLHF ma postać zamkniętą w terminach danych preferencji, więc można pominąć jawny model nagrody i optymalizować politykę bezpośrednio. Ten wgląd zapoczątkował rodzinę — IPO, KTO, SimPO, ORPO, BPO — każdy naprawiający inny tryb awarii DPO. W 2026 roku algorytmy bezpośredniego alignmentu (DAA) dostarczają więcej potoków pozatrainingowych na granicy niż PPO. Ale krzywa nadoptymalizacji z Lekcji 2 wciąż obowiązuje: DAA nie uciekają przed Goodhartem, tylko przesuwają miejsce, w którym gryzie.

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## Learning Objectives

- Wyprowadź postać zamkniętą DPO z optimum RLHF-z-KL.
- Wymień tryb awarii, który każdy z IPO, KTO, SimPO, ORPO, BPO naprawia w DPO.
- Odróżnij „ukrytą lukę nagrody" od „siły preferencji" i wyjaśnij, dlaczego odwzorowanie tożsamościowe IPO ma znaczenie.
- Wyjaśnij, dlaczego Rafailov et al. (NeurIPS 2024) dowodzą, że DAA nadoptymalizują pomimo braku jawnego RM.

## The Problem

Funkcja celu RLHF (Lekcja 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

ma znane optimum:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

Zatem nagroda jest ukrycie zdefiniowana przez stosunek optymalnej polityki do referencyjnej:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Podstaw to do prawdopodobieństwa preferencji Bradleya-Terry'ego, a funkcja podziału `Z(x)` znika, ponieważ zależy tylko od `x`. Pozostaje strata w samych parametrach polityki — bez potrzeby modelu nagrody. To jest DPO.

Haczyk: wyprowadzenie zakłada, że optimum jest osiągalne, dane preferencji są w dystrybucji, a polityka referencyjna jest prawdziwą kotwicą dominanty. Żadne z tych założeń nie jest dokładnie spełnione. Każdy członek rodziny naprawia inne naruszone założenie.

## The Concept

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

Co może pójść źle:

- Ukryta luka nagrody `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` jest nieograniczona. Drobna preferencja może wygenerować dowolnie dużą lukę.
- Strata popycha log-prawdopodobieństwa wybranej i odrzuconej odpowiedzi w przeciwnych kierunkach. Może obniżać bezwzględne log-prawdopodobieństwo wybranej, o ile odrzucona spada szybciej. Jest to zjawisko Degradacji Wybranej Odpowiedzi.
- Preferencje spoza dystrybucji (rzadka para vs rzadka para) generują arbitralne ukryte nagrody.

### IPO (Azar et al., 2024)

Identity Preference Optimization zastępuje log-sigmoid odwzorowaniem tożsamościowym na prawdopodobieństwie preferencji. Strata staje się błędem kwadratowym na ograniczonym celu:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

Margines jest ograniczony przez `1/(2 beta)`. Siła preferencji i luka ukrytej nagrody są proporcjonalne. Brak eksplozji.

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimization całkowicie rezygnuje ze struktury par. Dla pojedynczej oznaczonej odpowiedzi i binarnego sygnału „pożądana" lub „niepożądana" odwzorowuje to na użyteczność z teorii perspektywy:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

z różnymi wagami dla zysków i strat (awersja do straty). Zaletą jest możliwość użycia niesparowanych danych, których jest znacznie więcej.

### SimPO (Meng et al., 2024)

Simple Preference Optimization dopasowuje sygnał treningowy do generowania. Usuwa całkowicie politykę referencyjną i normalizuje log-prawdopodobieństwo przez długość:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

z marginesem `gamma` dla stabilizacji. Normalizacja długości usuwa motywację do wykorzystywania trybu awarii obciążenia długością DPO (dłuższe `y_w` daje z definicji większą lukę log-prawdopodobieństwa).

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization dodaje człon preferencji do standardowego ujemnego logarytmu wiarygodności SFT:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

Brak polityki referencyjnej — człon SFT jest regulatorem. Trenuj w jednym etapie od modelu bazowego do modelu z alignmentem. Bez osobnego punktu kontrolnego SFT.

### BPO (ICLR 2026 submission, OpenReview id=b97EwMUWu7)

Identyfikuje problem Degradacji Wybranej Odpowiedzi: DPO zachowuje ranking `y_w > y_l`, ale bezwzględne log-prawdopodobieństwo `y_w` może spaść. BPO dodaje jednoliniową korektę, która karze spadki na wybranej odpowiedzi. Zgłoszona poprawa dokładności o +10,1% na Llama-3.1-8B-Instruct w rozumowaniu matematycznym względem DPO.

### The universal result: DAAs still over-optimize

Rafailov et al. „Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024) trenowali polityki z DPO, IPO, SLiC na wielu zbiorach danych w różnych budżetach KL. Krzywe złota-nagroda-vs-KL mają ten sam kształt szczytu-i-załamania co Gao et al. Ukryta nagroda odwołuje się do próbek spoza dystrybucji podczas treningu; regularyzacja KL tego nie stabilizuje.

DAA nie uciekają przed Goodhartem. Zmieniają powierzchnię, na której gryzie, z „model nagrody nadoptymalizowany" na „stosunek polityki referencyjnej nadoptymalizowany." Uniwersalne rozwiązanie — lepsze dane, ensemble, wczesne zatrzymanie — dotyczy obu.

### Choosing among them (2026)

- Jeśli masz duże sparowane dane preferencji: DPO z konserwatywnym beta, SimPO jeśli widoczne jest obciążenie długością.
- Jeśli masz niesparowane binarne opinie: KTO.
- Jeśli chcesz potok jednoetapowy z modelu bazowego: ORPO.
- Jeśli widzisz degradację log-prawdopodobieństw wybranych w logach DPO: BPO.
- Jeśli siły preferencji znacznie się różnią, a DPO się nasyca: IPO.

Każde laboratorium uruchamia wszystkie pięć na baterii i wybiera zwycięzcę per zadanie. Nie ma powodu, dla którego optimum jest takie samo dla rozumowania matematycznego i bezpieczeństwa.

```figure
dpo-margin
```

## Use It

`code/main.py` porównuje sześć funkcji straty (DPO, IPO, KTO, SimPO, ORPO, BPO) na zabawkowym zbiorze preferencji, gdzie prawdziwa siła preferencji różni się w zależności od pary. Każda strata jest optymalizowana na tej samej 500-parowej próbce z małą polityką softmax. Wykreśla końcowy wskaźnik wygranych, dryf log-prawdopodobieństwa wybranej i rozrzut ukrytej nagrody dla każdej metody.

## Ship It

Ta lekcja produkuje `outputs/skill-preference-loss-selector.md`. Otrzymując statystyki zbioru danych (sparowane vs niesparowane, zmienna vs jednolita siła preferencji, rozkład długości) i cel (jednoetapowy lub SFT-potem-preferencja), rekomenduje funkcję straty preferencji i raportuje tryb awarii, przed którym chroni.

## Exercises

1. Uruchom `code/main.py`. Zgłoś końcowy spadek log-prawdopodobieństwa wybranej odpowiedzi dla DPO i BPO. BPO powinno zachować wyższe bezwzględne prawdopodobieństwo wybranej — zweryfikuj to.

2. Zmodyfikuj dane preferencji tak, aby wszystkie pary miały równą siłę. Która z sześciu metod jest najbardziej odporna? Która degraduje? Wyjaśnij przewagę IPO w tym przypadku.

3. Spraw, aby odrzucone odpowiedzi były średnio 2 razy dłuższe niż wybrane. Nie zmieniając niczego innego, pokaż numerycznie wykorzystanie długości przez DPO i naprawę przez SimPO.

4. Rafailov et al. (NeurIPS 2024) twierdzą, że DAA nadoptymalizują. Odtwórz wersję jednopunktową: wykreśl dywergencję KL wybrana-minus-odrzucona i zaobserwuj nadoptymalizację w DPO przy dużym beta.

5. Przeczytaj abstrakt artykułu BPO (OpenReview b97EwMUWu7). Zapisz jednoliniową korektę, którą BPO dodaje do DPO. Potwierdź zgodność z implementacją w `code/main.py`.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DPO | „RLHF bez modelu nagrody" | Strata wyprowadzona z optimum RLHF w postaci zamkniętej; tylko parametry polityki |
| Implicit reward | „log-stosunek" | `beta * log(pi(y|x) / pi_ref(y|x))` — nagroda implikowana przez DPO |
| IPO | „ograniczone DPO" | Zastępuje log-sigmoid tożsamością; luka ukrytej nagrody ograniczona przez `1/(2 beta)` |
| KTO | „niesparowane DPO" | Użyteczność z teorii perspektywy nad pojedynczymi etykietami z awersją do straty |
| SimPO | „DPO bez referencji" | Znormalizowane długością log-prawdopodobieństwo + margines; brak polityki referencyjnej |
| ORPO | „jednoetapowe DPO" | NLL + człon preferencji ilorazu szans; trenuje z modelu bazowego w jednym przejściu |
| BPO | „DPO zachowujące wybraną" | DPO plus kara za zmniejszanie bezwzględnego log-prawdopodobieństwa wybranej odpowiedzi |
| Degraded Chosen | „wybrana spada" | DPO zmniejsza log-prawdopodobieństwo wybranej, o ile odrzucona spada szybciej |
| DAA | „algorytm bezpośredniego alignmentu" | Każda metoda straty preferencji, która pomija jawny RM |

## Further Reading

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)