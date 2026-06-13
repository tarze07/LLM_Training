# Long-Context Evaluation — NIAH, RULER, LongBench, MRCR (Ewaluacja długiego kontekstu — NIAH, RULER, LongBench, MRCR)

> Gemini 3 Pro reklamuje 10M tokenów kontekstu. Przy 1M tokenów, 8-igłowy MRCR spada do 26.3%. Reklamowane ≠ użyteczne. Ewaluacja długiego kontekstu mówi ci rzeczywistą pojemność modelu, który wdrażasz.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## Problem

Masz 200-stronicowy kontrakt. Model deklaruje kontekst 1M tokenów. Wklejasz kontrakt i pytasz: "Jaka jest klauzula wypowiedzenia?" Model odpowiada — ale ze strony tytułowej, bo klauzula wypowiedzenia znajduje się 120k tokenów w głąb, poza miejscem, gdzie model faktycznie skupia uwagę.

To jest luka pojemności kontekstu w 2026. Arkusze danych mówią 1M lub 10M. Rzeczywistość mówi, że 60-70% z tego jest użyteczne, a "użyteczne" zależy od zadania.

- **Wyszukiwanie (pojedyncza igła w stogu siana):** prawie idealne do reklamowanego maksimum w modelach z pierwszej ligi.
- **Wieloskokowe / agregacja:** gwałtownie spada po ~128k w większości modeli.
- **Rozumowanie nad rozproszonymi faktami:** pierwsze zadanie, które zawodzi.

Ewaluacja długiego kontekstu mierzy te osie. Ta lekcja wymienia benchmarki, co każdy faktycznie mierzy i jak zbudować niestandardowy test igły dla swojej domeny.

## Koncepcja

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Umieść fakt ("magiczne słowo to ananas") na kontrolowanej głębokości w długim kontekście. Poproś model o wydobycie go. Przeskanuj głębokość × długość. Oryginalny benchmark długiego kontekstu. Modele z pierwszej ligi teraz go nasycają; jest to konieczny, ale niewystarczający baseline.

**RULER (Nvidia, 2024).** 13 typów zadań w 4 kategoriach: wyszukiwanie (pojedynczy / wiele kluczy / wiele wartości), śledzenie wieloskokowe (śledzenie zmiennych), agregacja (częstotliwość słów), QA. Konfigurowalna długość kontekstu (4k do 128k+). Ujawnia modele, które nasycają NIAH, ale zawodzą na zadaniach wieloskokowych. W wydaniu 2024, tylko połowa z 17 modeli deklarujących 32k+ kontekstu utrzymała jakość przy 32k.

**LongBench v2 (2024).** 503 pytania wielokrotnego wyboru, konteksty 8k-2M słów, sześć kategorii zadań: QA z jednego dokumentu, QA z wielu dokumentów, długie uczenie w kontekście, długa rozmowa, repozytorium kodu, długie dane strukturalne. Produkcyjny benchmark dla rzeczywistego zachowania na długim kontekście.

**MRCR (Multi-Round Coreference Resolution).** Wieloobrotowa rezolucja koreferencji na skalę. Warianty 8-igłowy, 24-igłowy, 100-igłowy. Ujawnia, ile faktów model może żonglować, zanim uwaga ulegnie degradacji.

**NoLiMa.** "Nieleksykalna igła." Igła i zapytanie nie mają żadnego dosłownego nakładania się; wyszukiwanie wymaga jednego kroku rozumowania semantycznego. Trudniejsze niż NIAH.

**HELMET.** Łączy wiele dokumentów, zadaje pytanie z dowolnego z nich. Testuje selektywną uwagę.

**BABILong.** Osadza łańcuchy rozumowania bAbI w nieistotnych stogach siana. Testuje rozumowanie w stogu siana, nie tylko wyszukiwanie.

### Co faktycznie raportować

- **Reklamowane okno kontekstu.** Liczba z arkusza danych.
- **Efektywna długość wyszukiwania.** Próg NIAH na pewnym progu (np. 90%).
- **Efektywna długość rozumowania.** Próg wieloskokowy lub agregacji na tym progu.
- **Krzywa degradacji.** Dokładność vs długość kontekstu, wykreślone według typu zadania.

Dwie liczby do arkusza danych: efektywność wyszukiwania i efektywność rozumowania. Zazwyczaj efektywność rozumowania wynosi 25-50% reklamowanego okna.

## Zbuduj To

### Krok 1: niestandardowy NIAH dla twojej domeny

Zobacz `code/main.py`. Szkielet:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Przeskanuj `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Narysuj mapę ciepła. To jest karta NIAH dla twojego docelowego modelu.

### Krok 2: wariant wieloigłowy

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Pytania typu "Jakie są trzy magiczne słowa?" wymagają wydobycia wszystkich trzech. Sukces na pojedynczej igle nie przewiduje sukcesu na wielu igłach.

### Krok 3: wieloskokowe śledzenie zmiennych (w stylu RULER)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Odpowiedź wymaga połączenia trzech przypisań. Modele z pierwszej ligi przy 128k często spadają do 50-70% dokładności tutaj.

### Krok 4: LongBench v2 na twoim stosie

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Raportuj dokładność według kategorii. Zagregowane wyniki ukrywają duże różnice między zadaniami.

## Pułapki

- **Tylko ewaluacja NIAH.** Przejście NIAH przy 1M tokenów nic nie mówi o zadaniach wieloskokowych. Zawsze uruchom RULER lub niestandardowy test wieloskokowy.
- **Jednolite próbkowanie głębokości.** Wiele implementacji testuje tylko głębokość=0.5. Testuj głębokość=0, 0.25, 0.5, 0.75, 1.0 — efekt "lost in the middle" jest realny.
- **Leksykalne nakładanie się z wypełniaczem.** Jeśli igła dzieli słowa kluczowe z wypełniaczem, wyszukiwanie staje się trywialne. Użyj igieł NoLiMa bez nakładania się.
- **Ignorowanie opóźnienia.** Prompt 1M tokenów zajmuje 30-120 sekund na prefill. Mierz czas do pierwszego tokena wraz z dokładnością.
- **Liczby zgłaszane przez dostawców.** OpenAI, Google, Anthropic publikują własne wyniki. Zawsze uruchamiaj ponownie niezależnie na swoim przypadku użycia.

## Użyj Tego

Stos w 2026:

| Sytuacja | Benchmark |
|-----------|-----------|
| Szybkie sprawdzenie poprawności | Niestandardowy NIAH na 3 głębokościach × 3 długościach |
| Wybór modelu do produkcji | RULER (13 zadań) na twojej docelowej długości |
| Rzeczywista jakość QA | LongBench v2, podzbiór single-doc-QA |
| Rozumowanie wieloskokowe | BABILong lub niestandardowe śledzenie zmiennych |
| Konwersacyjne / dialog | MRCR 8-igłowy na twojej docelowej długości |
| Regresja przy uaktualnieniu modelu | Stały wewnętrzny harness NIAH + RULER, uruchamiany na każdym nowym modelu |

Reguła kciuka dla produkcji: nigdy nie ufaj oknu kontekstu, dopóki nie masz NIAH + 1 zadania rozumowania na zamierzonej długości.

## Dostarcz To

Zapisz jako `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Ćwiczenia

1. **Łatwe.** Zbuduj NIAH z 3 głębokościami (0.25, 0.5, 0.75) × 3 długościami (1k, 4k, 16k). Uruchom na dowolnym modelu. Narysuj mapę ciepła 3×3 wskaźnika przejścia.
2. **Średnie.** Dodaj wariant 3-igłowy. Zmierz wyszukiwanie wszystkich 3 na każdej długości. Porównaj ze wskaźnikiem przejścia dla pojedynczej igły przy tej samej długości.
3. **Trudne.** Skonstruuj zadanie śledzenia zmiennych (X1 → X2 → X3, z 3 skokami) osadzone w 64k wypełniacza. Zmierz dokładność na 3 modelach z pierwszej ligi. Raportuj efektywną długość rozumowania na model.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|-----------------|-----------------------|
| NIAH | Igła w stogu siana | Umieść fakt w wypełniaczu, poproś model o wydobycie go. |
| RULER | NIAH na sterydach | 13 typów zadań: wyszukiwanie / wieloskok / agregacja / QA. |
| Efektywny kontekst | Rzeczywista pojemność | Długość, przy której dokładność wciąż utrzymuje się powyżej progu. |
| Lost in the middle | Bias głębokości | Modele poświęcają mniej uwagi treści w środku długich wejść. |
| Multi-needle | Wiele faktów naraz | Wiele umieszczeń; testuje żonglowanie uwagą, nie tylko wyszukiwanie. |
| MRCR | Wieloobrotowa koreferencja | Koreferencja 8, 24 lub 100-igłowa; ujawnia nasycenie uwagi. |
| NoLiMa | Nieleksykalna igła | Igła i zapytanie nie dzielą dosłownych tokenów; wymaga rozumowania. |

## Dalsza Literatura

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — the original NIAH repo.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — the multi-task benchmark.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — real-world long-context eval.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — harder needles.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — the depth-bias paper.