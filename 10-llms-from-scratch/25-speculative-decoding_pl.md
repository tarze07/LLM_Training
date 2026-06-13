# Dekodowanie spekulatywne i EAGLE

> Graniczny LLM generujący jeden token wymaga pełnego przejścia w przód przez miliardy parametrów. To przejście w przód jest masowo nadmiarowe: przez większość czasu znacznie mniejszy model może poprawnie odgadnąć następne 3-5 tokenów, a duży model musi tylko *zweryfikować* zgadnięcie. Gdy zgadnięcie jest poprawne, dostajesz 5 tokenów w cenie jednego. Dekodowanie spekulatywne (Leviathan i in. 2023) uczyniło to dokładnym, a EAGLE-3 (2025) podniósł wskaźniki akceptacji do ~4.5 tokena na weryfikację — 4-5-krotne przyspieszenie przy dopasowanym rozkładzie wyjściowym.

**Type:** Build
**Languages:** Python (z numpy)
**Prerequisites:** Phase 10 Lesson 12 (optymalizacja wnioskowania), Phase 10 Lesson 04 (pretrenowanie Mini-GPT)
**Time:** ~75 minut

## Problem

Przepustowość dekodowania dla modelu klasy 70B na H100 wynosi zazwyczaj 40-80 tokenów/sekundę. Każdy token wymaga pełnego przejścia w przód, czytając wszystkie wagi modelu z HBM. Nie możesz zmniejszyć modelu bez zmiany jego wyjścia. Nie możesz zwiększyć rozmiaru partii poza pamięć. Utknąłeś — chyba że pozwolisz modelowi wyprowadzić więcej niż jeden token na przejście w przód.

Generowanie autoregresyjne wygląda na z natury sekwencyjne: `x_{t+1} = sample(p(· | x_{1:t}))`. Ale istnieje okazja do współbieżności. Gdybyś miał tani predyktor, który mówi „następne 4 tokeny to prawdopodobnie [a, b, c, d]", mógłbyś zweryfikować wszystkie 5 pozycji w **jednym przejściu w przód dużego modelu** i zaakceptować najdłuższy pasujący prefiks.

Leviathan, Kalai, Matias (2023, „Fast Inference from Transformers via Speculative Decoding") uczynili to dokładnym poprzez sprytną regułę akceptacji/odrzucenia, która zachowuje rozkład próbkowania modelu docelowego. Ten sam rozkład wyjściowy, 2-4× szybciej.

## Koncepcja

### Konfiguracja dwumodelowa

- **Model docelowy** `M_p`: duży, wolny, wysokiej jakości model, z którego faktycznie chcesz próbkować. Rozkład: `p(x)`.
- **Model szkicu** `M_q`: mały, szybki, niższej jakości model. Rozkład: `q(x)`. 5-30× mniejszy.

Na krok:

1. Model szkicu proponuje `K` tokenów autoregresyjnie: `x_1, x_2, ..., x_K ~ q`.
2. Model docelowy wykonuje JEDNO przejście w przód na wszystkich `K+1` pozycjach równolegle, produkując `p(x_k)` dla każdego proponowanego tokena.
3. Zaakceptuj/odrzuć każdy token od lewej do prawej za pomocą zmodyfikowanej reguły próbkowania z odrzuceniem poniżej. Zaakceptuj najdłuższy pasujący prefiks.
4. Jeśli jakikolwiek token zostanie odrzucony, próbkuj zastępstwo ze skorygowanego rozkładu i zatrzymaj się. W przeciwnym razie próbkuj jeden bonusowy token z `p(· | x_1...x_K)`.

Jeśli szkic idealnie pasuje do modelu docelowego, dostajesz K+1 tokenów na jeden forward modelu docelowego. Jeśli szkic jest błędny na pozycji 1, dostajesz tylko 1 token.

### Reguła dokładności

Dekodowanie spekulatywne jest **dowiedlne równoważne w rozkładzie próbkowaniu z p**. Reguła odrzucenia:

```
Dla każdego tokena ze szkicu x_t:
    r ~ Uniform(0, 1)
    jeśli r < p(x_t) / q(x_t):
        zaakceptuj x_t
    w przeciwnym razie:
        próbkuj zastępstwo z reszty: (p - q)+ / ||(p - q)+||_1
        zatrzymaj się
```

gdzie `(p - q)+` oznacza dodatnią część różnicy punktowej. Gdy szkic i model docelowy są zgodne (`p ≈ q`), akceptacja jest bliska 1. Gdy się różnią, rozkład resztowy jest skonstruowany tak, że ogólna próbka jest wciąż dokładnie `p`.

**Przypadek zachłanny.** Dla próbkowania temperature=0 wystarczy sprawdzić `argmax(p) == x_t`. Jeśli tak, zaakceptuj; jeśli nie, wypisz `argmax(p)` i zatrzymaj się.

### Oczekiwane przyspieszenie

Jeśli wskaźnik akceptacji na poziomie tokena modelu szkicu wynosi `α`, oczekiwana liczba tokenów wyprodukowanych na jedno przejście w przód modelu docelowego wynosi:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = długość szkicu, α w [0, 1]
```

Przy `α = 0.8, K = 4`: `(1 - 0.8^5)/(1 - 0.8) = 3.36` tokena na forward. Pojedynczy forward modelu docelowego kosztuje z grubsza `cost_q * K + cost_p` (K kroków szkicu plus jedna weryfikacja modelu docelowego). Jeśli `cost_p >> cost_q * K`, stosunek przyspieszenia wynosi `3.36× / 1 = 3.36×` na przepustowości.

Jedynym prawdziwym parametrem jest `α`, który zależy wyłącznie od dopasowania szkicu do modelu docelowego. Dobry szkic to wszystko.

### Trenowanie szkicu: destylacja

Losowy mały model daje słaby szkic. Standardowa recepta to destylacja z modelu docelowego:

1. Wybierz małą architekturę (~1B dla modelu docelowego 70B, ~500M dla modelu docelowego 7B).
2. Uruchom model docelowy na dużym korpusie tekstowym; przechowuj jego rozkłady następnego tokena.
3. Trenuj szkic z dywergencją KL względem rozkładu modelu docelowego (nie względem tokenów prawdy podstawowej).

Wynik: `α` zazwyczaj 0.6-0.8 przy kodowaniu, 0.7-0.85 przy czacie w języku naturalnym. Przyspieszenia 2-3× w produkcji.

### EAGLE: szkicowanie drzewiaste + ponowne użycie cech

Li, Wei, Zhang, Zhang (2024, „EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") zaobserwowali dwie nieefektywności w standardowym dekodowaniu spekulatywnym:

1. Szkic wykonuje K kroków sekwencyjnych, każdy pełny. Ale szkic mógłby ponownie użyć cech modelu docelowego (stanów ukrytych) z ostatniej weryfikacji — model docelowy już obliczył bogate reprezentacje, które szkic wyprowadza od zera.
2. Szkic wyprowadza liniowy łańcuch. Gdyby szkic mógł wyprowadzić *drzewo* kandydatów (każdy węzeł wiele zgadnięć), pojedyncze przejście w przód modelu docelowego mogłoby zweryfikować wiele ścieżek kandydackich równolegle za pomocą drzewiastej maski attention i wybrać najdłuższą zaakceptowaną gałąź.

Zmiany EAGLE-1:
- Wejście szkicu = końcowy stan ukryty modelu docelowego na pozycji t, a nie surowe tokeny.
- Architektura szkicu = 1 warstwa dekodera transformera (nie osobny mały model).
- Wyjście = drzewo K = 4-8 kandydatów na głębokość, głębokość 4-6.

EAGLE-2 (2024) dodaje dynamiczną topologię drzewa: drzewo rozszerza się tam, gdzie szkic jest niepewny, i pozostaje wąskie tam, gdzie jest pewny. Podnosi `α_effective` bez zwiększania kosztu weryfikacji.

EAGLE-3 (Li i in. 2025, „EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") usuwa zależność od stałej górnej warstwy cech i trenuje szkic z nową stratą „symulacji w czasie testu" — szkic jest trenowany na wyjściach pasujących do rozkładu modelu docelowego w czasie testu, a nie do rozkładu treningowego wymuszonego przez nauczyciela. Wskaźnik akceptacji rośnie z 0.75 (EAGLE-2) do 0.82 (EAGLE-3), a średnia liczba tokenów/weryfikację z 3.0 do 4.5.

### Weryfikacja drzewiastą attention

Gdy szkic wyprowadza drzewo, model docelowy weryfikuje je w jednym przejściu w przód za pomocą **drzewiastej maski attention** — maski przyczynowej, która koduje topologię drzewa, a nie czystą linię. Każdy token zwraca uwagę tylko na swoich przodków w drzewie. Przejście weryfikacyjne to wciąż jeden forward, jedno mnożenie macierzy; maska topologiczna kosztuje tylko kilka dodatkowych wpisów KV.

```
        korzeń
       /    \
      a      b
     / \    / \
    c  d   e   f
```

Jeśli `a, b` to konkurujące kandydatury pierwszego tokena, a `c, d, e, f` to kandydatury drugiego tokena, wszystkie sześć pozycji jest weryfikowanych w jednym przejściu w przód. Wynikiem jest najdłuższy prefiks wzdłuż dowolnej zaakceptowanej ścieżki.

### Kiedy wygrywa, kiedy nie

**Wygrywa:**
- Czat / dokończanie z przewidywalnym tekstem (kod, popularny angielski, ustrukturyzowane wyjście). `α` jest wysokie.
- Ustawienia z niewykorzystanymi obliczeniami GPU podczas dekodowania (faza ograniczona pamięcią). Szkicowanie drzewiaste wykorzystuje dostępne FLOPsy.

**Przegrywa / brak zysku:**
- Wysoce stochastyczne wyjścia (kreatywne pisanie przy wysokiej temperaturze). `α` spada w kierunku `1/|vocab|`.
- Obsługa wsadowa z bardzo wysoką współbieżnością — grupowanie już wypełnia FLOPsy, mało miejsca na weryfikację drzewiastą.
- Bardzo małe modele docelowe, gdzie szkic nie jest dużo mniejszy.

Produkcyjne sklepy zazwyczaj raportują 2-3× przyspieszenie czasu ściennego na czacie, 3-5× na generowaniu kodu i blisko zera na kreatywnym pisaniu.

```figure
speculative-decoding
```

## Zbuduj

`code/main.py`:

- Referencyjna `speculative_decode(target, draft, prompt, K, temperature)`, która implementuje dokładną regułę odrzucenia i weryfikuje, że zachowuje rozkład modelu docelowego (empiryczna KL < 0.01 vs zwykłe próbkowanie modelu docelowego).
- Szkicownik drzewiasty w stylu EAGLE, który buduje drzewo o głębokości K z rozgałęzieniem top-p.
- Konstruktor drzewiastej maski attention, który produkuje odpowiedni wzór przyczynowy dla weryfikatora.
- Uprząż wskaźnika akceptacji, która uruchamia oba na małym LM (destyluj jeden GPT-2-small z modelu docelowego GPT-2-medium).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """Jedna runda dekodowania spekulatywnego. Zwraca listę zaakceptowanych tokenów."""
    # 1. Szkicuj K tokenów
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Model docelowy oblicza p na każdej pozycji szkicu + 1 dodatkowej
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Zaakceptuj/odrzuć od lewej do prawej
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. Wszystkie K zaakceptowane → próbkuj bonusowy token z modelu docelowego
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Użyj

- **vLLM** i **SGLang** dostarczają dekodowanie spekulatywne pierwszej klasy. Flag: `--speculative_model`, `--num_speculative_tokens`. Obsługa EAGLE-2/3 przez flagę `--spec_decoding_algorithm eagle`.
- **NVIDIA TensorRT-LLM** natywnie obsługuje drzewa Medusa i EAGLE.
- **Referencyjne modele szkicu**: `Qwen/Qwen3-0.6B-spec` (szkice dla Qwen3-32B), `meta-llama/Llama-3.2-1B-Instruct-spec` (szkice dla 70B).
- **Głowice Medusa** (Cai i in. 2024, „Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): zamiast modelu szkicu, dodaj K równoległych głowic predykcyjnych do samego modelu docelowego. Prostsze we wdrożeniu, nieco niższa akceptacja niż EAGLE.

## Dostarcz

Ta lekcja produkuje `outputs/skill-speculative-tuning.md` — umiejętność, która profiluje obciążenie modelu docelowego i wybiera: model szkicu, K (długość szkicu), szerokość drzewa, temperaturę i kiedy wrócić do zwykłego dekodowania.

## Ćwiczenia

1. Zaimplementuj dokładną regułę odrzucenia i empirycznie ją zweryfikuj. Uruchom 10K próbek przez `speculative_decode` i przez zwykłe próbkowanie modelu docelowego; oblicz odległość TV między dwoma rozkładami wyjściowymi. Powinna być < 0.01.

2. Oblicz wzór na przyspieszenie. Dla ustalonego `α` i `K`, wykreśl oczekiwane tokeny na forward modelu docelowego. Znajdź optymalne K dla α ∈ {0.5, 0.7, 0.9}.

3. Wytrenuj mały szkic. Weź model docelowy GPT-2 124M i destyluj szkic GPT-2 30M na 100M tokenów ze stratą KL. Zmierz `α` na wstrzymanym tekście. Oczekiwane: 0.6-0.7.

4. Zaimplementuj szkicowanie drzewiaste w stylu EAGLE. Zamiast łańcucha, niech szkic wyprowadza top-3 gałęzie na każdej głębokości. Zbuduj drzewiastą maskę attention. Zweryfikuj, że model docelowy akceptuje najdłuższą poprawną gałąź.

5. Zmierz tryby awarii. Uruchom dekodowanie spekulatywne przy temperature=1.5 (wysoka stochastyczność). Pokaż, że α załamuje się i algorytm jest wolniejszy niż zwykłe dekodowanie z powodu narzutu szkicu.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|-----------------|------------------------|
| Model docelowy | „Duży model" | Wolny, wysokiej jakości model, z którego chcesz próbkować (rozkład p) |
| Model szkicu | „Spekulator" | Mały, szybki predyktor (rozkład q); 5-30x mniejszy |
| K / długość szkicu | „Spoglądanie w przód" | Liczba spekulowanych tokenów na jedną weryfikację |
| α / wskaźnik akceptacji | „Współczynnik trafień" | Prawdopodobieństwo na token, że propozycja szkicu zostanie zaakceptowana |
| Dokładna reguła odrzucenia | „Test akceptacji" | Porównanie r < p/q, które zachowuje rozkład modelu docelowego |
| Rozkład resztowy | „Skorygowane p-q" | (p - q)+ / ||(p - q)+||_1, rozkład do próbkowania przy odrzuceniu |
| Szkicowanie drzewiaste | „Rozgałęziająca spekulacja" | Szkic wyprowadza drzewo kandydatów, weryfikowane w jednym przejściu z drzewiastą maską attention |
| Drzewiasta maska attention | „Maska topologiczna" | Maska przyczynowa kodująca topologię drzewa, tak aby każdy węzeł zwracał uwagę tylko na swoich przodków |
| Głowice Medusa | „Równoległe głowice" | K dodatkowych głowic predykcyjnych na samym modelu docelowym; brak osobnego modelu szkicu |
| Ponowne użycie cech EAGLE | „Szkic na stanie ukrytym" | Wejściem szkicu jest ostatni stan ukryty modelu docelowego, a nie surowe tokeny, co zmniejsza szkic |
| Strata symulacji w czasie testu | „Trenowanie EAGLE-3" | Trenuj szkic na wyjściach pasujących do rozkładu modelu docelowego w czasie testu, a nie wymuszania przez nauczyciela |

## Dalsze czytanie

- [Leviathan, Kalai, Matias, 2023 — „Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — dokładna reguła odrzucenia i teoretyczna analiza przyspieszenia
- [Chen, Borgeaud, Irving et al., 2023 — „Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — równoczesny artykuł o próbkowaniu spekulatywnym w DeepMind
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — „Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — alternatywa z równoległymi głowicami zamiast modelu szkicu
- [Li, Wei, Zhang, Zhang, 2024 — „EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — ponowne użycie cech i szkicowanie drzewiaste
- [Li et al., 2024 — „EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — dynamiczna topologia drzewa
- [Li et al., 2025 — „EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — dopasowanie czasu trenowania do czasu testu
- [Fu, Haotian, Peng et al., 2024 — „Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — dekodowanie Jacobiego/wyprzedzające, alternatywa bez spekulatora