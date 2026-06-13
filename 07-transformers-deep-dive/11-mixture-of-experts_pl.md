# Mixture of Experts (MoE)

> Gęsty transformer 70B aktywuje każdy parametr dla każdego tokena. MoE 671B aktywuje tylko 37B na token i bije go na każdym benchmarku. Rzadkość to najważniejsza idea skalowania tej dekady.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## Problem

FLOP-y gęstego transportera podczas wnioskowania są równe jego liczbie parametrów (razy 2 dla przepustu w przód). Skaluj gęsty model w górę, a każdy token płaci pełny rachunek. Do 2024 roku granica możliwości uderzyła w ścianę obliczeniową: aby być znacząco mądrzejszym, potrzebowałeś wykładniczo więcej FLOP-ów na token.

Mixture of Experts zrywa to powiązanie. Zastąp każdy FFN przez `E` niezależnych ekspertów + router, który wybiera `k` ekspertów na token. Całkowita liczba parametrów = `E × rozmiar_FFN`. Aktywne parametry na token = `k × rozmiar_FFN`. Typowa konfiguracja z 2026: `E=256`, `k=8`. Pamięć skaluje się z `E`, obliczenia skalują się z `k`.

Granica możliwości w 2026 roku to prawie wyłącznie MoE: DeepSeek-V3 (671B całkowitych / 37B aktywnych), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. W niezależnym rankingu Artificial Analysis, top 10 modeli open-source to wszystkie MoE.

## Koncepcja

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### Zamiana FFN

Blok gęstego transportera:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

Blok MoE:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # wybierz k z E na token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Każdy ekspert to niezależny FFN (zazwyczaj SwiGLU). Router to pojedyncza warstwa liniowa. Każdy token wybiera własne `k` ekspertów i otrzymuje bramkowaną mieszankę ich wyjść.

### Problem równoważenia obciążenia

Jeśli router kieruje 90% tokenów przez eksperta 3, pozostali eksperci głodują. Wypróbowano trzy poprawki:

1. **Pomocnicza strata równoważenia obciążenia** (Switch Transformer, Mixtral). Dodaj karę proporcjonalną do wariancji w użyciu ekspertów. Działa, ale dodaje hiperparametr i drugi sygnał gradientu.
2. **Pojemność eksperta + odrzucanie tokenów** (wczesny Switch). Każdy ekspert przetwarza co najwyżej `C × N/E` tokenów; nadmiarowe tokeny pomijają warstwę. Szkodzi jakości.
3. **Równoważenie bez straty pomocniczej** (DeepSeek-V3). Dodaj uczone odchylenie na eksperta, które przesuwa wybór top-k routera. Odchylenie jest aktualizowane poza funkcją straty. Brak kary na główny cel. Wielkie odblokowanie 2024 roku.

Podejście DeepSeek-V3: po każdym kroku treningowym, dla każdego eksperta, sprawdź, czy jego użycie jest powyżej czy poniżej celu. Przesuń odchylenie o `±γ`. Wybór używa `scores + bias`. Prawdopodobieństwa ekspertów używane do bramkowania to surowe `scores` bez zmian. Odłącza kierowanie od ekspresji.

### Współdzieleni eksperci

DeepSeek-V2/V3 dodatkowo dzielą ekspertów na *współdzielonych* i *kierowanych*. Każdy token przechodzi przez wszystkich współdzielonych ekspertów. Kierowani eksperci są wybierani przez top-k. Współdzieleni eksperci przechwytują wspólną wiedzę; kierowani eksperci specjalizują się. V3 uruchamia 1 współdzielonego eksperta plus top-8 z 256 kierowanych.

### Drobnoziarniści eksperci

Klasyczny MoE (GShard, Switch): każdy ekspert jest tak szeroki jak pełny FFN. `E` jest małe (8–64), `k` jest małe (1–2).

Nowoczesny drobnoziarnisty MoE (DeepSeek-V3, Qwen-MoE): każdy ekspert jest węższy (1/8 rozmiaru FFN). `E` jest duże (256+), `k` jest większe (8+). Te same całkowite parametry, ale kombinacje skalują się znacznie szybciej. `C(256, 8) = 400 bilionów` możliwych "ekspertów" na token. Jakość rośnie, opóźnienie pozostaje stałe.

### Profil kosztów

Na token, na warstwę:

| Konfiguracja | Aktywne parametry / token | Całkowite parametry |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (gęsty) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 bije Llamę 3 70B (gęsty) na prawie każdym benchmarku, wykonując **mniej aktywnych FLOP-ów na token**. Więcej parametrów = więcej wiedzy. Więcej aktywnych FLOP-ów = więcej obliczeń na token. MoE rozdziela je.

### Haczyk: pamięć

Wszyscy eksperci żyją na GPU niezależnie od tego, którzy zostają uruchomieni. Model 671B potrzebuje ~1.3 TB VRAM dla wag fp16. Wdrożenie MoE na granicy możliwości wymaga równoległości ekspertów — rozdziel ekspertów na wiele GPU, kieruj tokeny przez sieć. Opóźnienie jest zdominowane przez komunikację all-to-all, nie przez mnożenie macierzy.

## Zbuduj To

Zobacz `code/main.py`. Kompaktowa warstwa MoE w czystej bibliotece standardowej z:

- `n_experts=8` ekspertami przypominającymi SwiGLU (jeden liniowy każdy, dla ilustracji)
- kierowaniem top-k=2
- wagami bramkowania znormalizowanymi softmax
- równoważeniem bez straty pomocniczej poprzez odchylenie na eksperta

### Krok 1: router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax nad ORYGINALNYMI wynikami wybranych ekspertów
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Odchylenie wpływa na wybór, nie na wagę bramki. To sztuczka DeepSeek-V3 — odchylenie koryguje nierównowagę obciążenia bez sterowania przewidywaniami modelu.

### Krok 2: przepuść 100 tokenów przez router

Śledź, jak często każdy ekspert jest uruchamiany. Bez odchylenia użycie jest wypaczone. Z pętlą aktualizacji odchylenia (`-γ` dla nadużywanych ekspertów, `+γ` dla niedocenianych), użycie zbiega się do rozkładu jednostajnego w ciągu kilku iteracji.

### Krok 3: porównanie liczby parametrów

Wypisz "gęsty odpowiednik" konfiguracji MoE. W kształcie DeepSeek-V3: 256 kierowanych + 1 współdzielony, 8 aktywnych, d_model=7168. Całkowita liczba parametrów robi wrażenie. Aktywna liczba to siódma część gęstego Llama 3 70B.

## Użyj Tego

Ładowanie HuggingFace:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

Produkcyjne wnioskowanie w 2026: vLLM obsługuje kierowanie MoE natywnie. SGLang ma najszybszą ścieżkę równoległości ekspertów. Oba automatycznie obsługują wybór top-k i równoległość ekspertów.

**Kiedy wybrać MoE:**
- Chcesz jakości z pogranicza możliwości przy niższym koszcie wnioskowania na token.
- Masz infrastrukturę VRAM / równoległości ekspertów.
- Twój workload jest ciężki od tokenów (czat, kod), a nie od kontekstu (długie dokumenty).

**Kiedy NIE wybierać MoE:**
- Wdrożenie na urządzeniach brzegowych — płacisz za pełną pamięć za każdy aktywny FLOP.
- Obsługa pojedynczego użytkownika krytyczna pod względem opóźnienia — kierowanie ekspertów dodaje narzut.
- Małe modele (<7B) — przewaga jakościowa MoE pojawia się dopiero powyżej progu obliczeniowego (~6B aktywnych parametrów).

## Dostarcz To

Zobacz `outputs/skill-moe-configurator.md`. Umiejętność wybiera E, k i układ współdzielonych ekspertów dla nowego MoE na podstawie budżetu parametrów, tokenów treningowych i celu wdrożenia.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Obserwuj, jak aktualizacja odchylenia bez straty pomocniczej wyrównuje użycie ekspertów w ciągu 50 iteracji.
2. **Średnie.** Zastąp uczony router routerem opartym na haszowaniu (deterministyczny, bez uczenia). Porównaj jakość i równowagę. Dlaczego uczony router jest lepszy?
3. **Trudne.** Zaimplementuj kierowanie "rollout-matched" w stylu GRPO (sztuczka DeepSeek-V3.2): zarejestruj, którzy eksperci są uruchamiani podczas wnioskowania, wymuś to samo kierowanie podczas obliczania gradientu. Zmierz efekt na zabawkowym układzie policy-gradient.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Ekspert (expert) | "Jeden FFN spośród wielu" | Niezależna sieć feed-forward; parametry dedykowane rzadkiemu wycinkowi obliczeń FFN. |
| Router | "Bramka" | Malutka warstwa liniowa, która ocenia każdy token względem każdego eksperta; wybór top-k. |
| Kierowanie top-k | "k aktywnych ekspertów na token" | Obliczenia FFN każdego tokena przechodzą przez dokładnie k ekspertów, ważone bramką. |
| Strata pomocnicza (auxiliary loss) | "Kara za równoważenie obciążenia" | Dodatkowy człon straty karzący wypaczone użycie ekspertów. |
| Bez straty pomocniczej (auxiliary-loss-free) | "Sztuczka DeepSeek-V3" | Równoważenie poprzez odchylenie na eksperta tylko przy wyborze routera; brak dodatkowego gradientu. |
| Współdzielony ekspert (shared expert) | "Zawsze włączony" | Dodatkowy ekspert, przez który przechodzi każdy token; przechwytuje wspólną wiedzę. |
| Równoległość ekspertów (expert parallelism) | "Dziel według eksperta" | Rozmieść różnych ekspertów na różnych GPU; kieruj tokeny przez sieć. |
| Rzadkość (sparsity) | "Aktywne parametry < całkowite parametry" | Stosunek `k × rozmiar_eksperta / (E × rozmiar_eksperta)`; 37/671 ≈ 5.5% dla DeepSeek-V3. |

## Dalsza Lektura

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — pomysł.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, klasyczny MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + MoE bez straty pomocniczej + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — artykuł o równoważeniu opartym na odchyleniu.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — podział na drobnoziarnistych + współdzielonych ekspertów, którego używa router w tej lekcji.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — oryginalny artykuł o współdzielonych ekspertach.