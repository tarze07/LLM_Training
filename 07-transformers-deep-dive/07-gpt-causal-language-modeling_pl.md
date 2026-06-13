# GPT — Przyczynowe Modelowanie Języka (Causal Language Modeling)

> BERT widzi obie strony. GPT widzi tylko przeszłość. Trójkątna maska to najbardziej wpływowa pojedyncza linia kodu we współczesnej AI.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## Problem

Model językowy odpowiada na jedno pytanie: mając pierwsze `t-1` tokenów, jaki jest rozkład prawdopodobieństwa dla tokenu `t`? Trenuj na tym sygnale — przewidywaniu następnego tokenu — a otrzymasz model, który może generować dowolny tekst po jednym tokenie na raz.

Aby trenować go od końca do końca na całej sekwencji równolegle, potrzebujesz, aby predykcja każdej pozycji zależała tylko od wcześniejszych pozycji. W przeciwnym razie model trywialnie oszukuje, zaglądając do odpowiedzi.

Maska przyczynowa (causal mask) to robi. Jest to pojedyncza górnotrójkątna macierz wartości `-inf` dodawana do wyników attention przed softmaxem. Po softmaxie te pozycje stają się 0. Każda pozycja może zwracać uwagę tylko na siebie i wcześniejsze pozycje. A ponieważ stosujesz ją raz do całej sekwencji, otrzymujesz N równoległych predykcji następnego tokenu w jednym przejściu w przód.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — wszystkie to dekoderowe, przyczynowe transformery z tą samą główną pętlą. Tylko większe, z lepszymi danymi i lepszym RLHF.

## Koncepcja

![Maska przyczynowa tworzy trójkątną macierz attention](../assets/causal-attention.svg)

### Maska

Mając sekwencję długości `N`, zbuduj macierz `N × N`:

```
M[i, j] = 0       jeśli j <= i
M[i, j] = -inf    jeśli j > i
```

Dodaj `M` do surowych wyników attention przed softmaxem. `exp(-inf) = 0`, więc maskowane pozycje mają zerową wagę. Każdy wiersz macierzy attention to rozkład prawdopodobieństwa tylko nad poprzednimi pozycjami.

Koszt implementacji: jedno wywołanie `torch.tril()`. Czas obliczenia: nanosekundy. Wpływ na dziedzinę: wszystko.

### Równoległe trenowanie, szeregowe wnioskowanie

Trenowanie: wykonaj przejście w przód całej sekwencji `(N, d_model)` raz, oblicz N strat entropii krzyżowej (jedna na pozycję), zsumuj, wsteczna propagacja. Równolegle wzdłuż sekwencji. Dlatego trenowanie GPT skaluje się — przetwarzasz 1M tokenów w partii w jednym przejściu GPU.

Wnioskowanie: generujesz token po tokenie. Podaj `[t1, t2, t3]`, otrzymaj `t4`. Podaj `[t1, t2, t3, t4]`, otrzymaj `t5`. Podaj `[t1, t2, t3, t4, t5]`, otrzymaj `t6`. Pamięć podręczna KV (KV cache, Lekcja 12) zapisuje stany ukryte `t1…tn`, aby nie przeliczać ich przy każdym kroku. Ale szeregowa głębokość przy wnioskowaniu = długość wyjścia. To jest podatek autoregresyjny i dlatego dekodowanie jest wąskim gardłem opóźnienia każdego LLMa.

### Funkcja straty — przesunięcie o jeden

Mając tokeny `[t1, t2, t3, t4]`:

- Wejście: `[t1, t2, t3]`
- Cele: `[t2, t3, t4]`

Dla każdej pozycji `i`, oblicz `-log P(cele_i | wejście[:i+1])`. Sumuj. To jest entropia krzyżowa dla całej sekwencji.

Każdy transformerowy model językowy, o którym słyszałeś, trenuje na tej stracie. Pretrenowanie, dostrajanie, SFT — ta sama strata, różne dane.

### Strategie dekodowania

Po trenowaniu wybór próbkowania ma większe znaczenie, niż ludzie myślą.

| Metoda | Co robi | Kiedy używać |
|--------|---------|--------------|
| Zachłanne (greedy) | Argmax w każdym kroku | Zadania deterministyczne, uzupełnianie kodu |
| Temperatura | Podziel logity przez T, próbkuj | Zadania kreatywne, wyższe T = większa różnorodność |
| Top-k | Próbkuj tylko z top-k tokenów | Zabija ogony o niskim prawdopodobieństwie |
| Top-p (nucleus) | Próbkuj z najmniejszego zbioru o łącznym p ≥ p | Domyślne od 2020+; dostosowuje się do kształtu rozkładu |
| Min-p | Zachowaj tokeny z `p > min_p * max_p` | 2024+; lepiej odrzuca długie ogony niż top-p |
| Dekodowanie spekulacyjne | Model szkicowy proponuje N tokenów, duży model weryfikuje | 2–3× redukcja opóźnienia przy tej samej jakości |

W 2026 roku min-p + temperatura 0.7 to rozsądne ustawienie domyślne dla modeli z otwartymi wagami. Dekodowanie spekulacyjne to standard branżowy dla każdego produkcyjnego stosu wnioskowania.

### Co sprawiło, że "przepis GPT" zadziałał

1. **Tylko dekoder.** Żadnego narzutu enkodera. Jedno przejście attention + FFN na warstwę.
2. **Skalowanie.** 124M → 1.5B → 175B → biliony. Prawa skalowania Chinchilli (Lekcja 13) mówią, jak wydawać moc obliczeniową.
3. **Uczenie się w kontekście.** Pojawiło się około 6B–13B. Model może podążać za przykładami few-shot bez dostrajania.
4. **RLHF.** Potrening na ludzkich preferencjach przekształcił surowy pretrenowany tekst w asystentów czatu.
5. **Pre-norm + RoPE + SwiGLU.** Stabilne trenowanie na dużą skalę.

Podstawowa architektura nie zmieniła się znacząco od GPT-2. Wszystko, co interesujące, wydarzyło się w danych, skali i potreningu.

```figure
causal-mask
```

## Zbuduj to

### Krok 1: maska przyczynowa

Zobacz `code/main.py`. Jedno-linijka:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Dodaj ją do wyników attention przed softmaxem. To cały mechanizm.

### Krok 2: 2-warstwowy model w stylu GPT

Ułóż dwa bloki dekodera (maskowany self-attention + FFN, bez cross-attention). Dodaj osadzenie tokenów, kodowanie pozycyjne i odosadzenie (unembedding, powiązane z macierzą osadzenia tokenów — standardowy trik od GPT-2).

### Krok 3: przewidywanie następnego tokenu, od końca do końca

Na zabawkowym słowniku 20 tokenów wyprodukuj logity na każdej pozycji. Oblicz stratę entropii krzyżowej względem celu przesuniętego o jeden. Bez gradientu — to sprawdzenie poprawności przejścia w przód.

### Krok 4: próbkowanie

Zaimplementuj metody: zachłanna, temperatura, top-k, top-p, min-p. Uruchom każdą na stałym podpowiedzi i porównaj wyniki. Funkcja próbkowania to 10 linii.

## Użyj tego

PyTorch, idiom z 2026 roku:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Pod maską `generate()` wykonuje przejście w przód, pobiera logity z ostatniej pozycji, próbkuje następny token, dodaje go i powtarza. Każdy produkcyjny stos wnioskowania LLM (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) implementuje tę samą pętlę z intensywną optymalizacją — wsadowe prefill, ciągłe grupowanie, stronicowanie pamięci podręcznej KV, dekodowanie spekulacyjne.

**GPT vs BERT, po jednej linii:** GPT przewiduje `P(x_t | x_{<t})`. BERT przewiduje `P(x_{maskowany} | x_{niemaskowany})`. Funkcja straty decyduje o tym, czy model może generować.

## Dostarcz to

Zobacz `outputs/skill-sampling-tuner.md`. Umiejętność (skill) dobiera parametry próbkowania dla nowego zadania generowania i wskazuje, kiedy wymagane jest deterministyczne dekodowanie.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` i zweryfikuj, że macierz attention przyczynowej jest dolnotrójkątna po softmaxie. Sprawdź punktowo: wiersz 3 powinien mieć wagi tylko w kolumnach 0–3.
2. **Średnie.** Zaimplementuj wyszukiwanie wiązkowe (beam search) o szerokości 4. Porównaj perplexity beam-4 vs zachłanne na 10 krótkich podpowiedziach. Czy beam zawsze wygrywa? (Podpowiedź: zwykle tak dla tłumaczenia, nie dla otwartego czatu.)
3. **Trudne.** Zaimplementuj dekodowanie spekulacyjne: użyj małego 2-warstwowego modelu jako szkicownika i 6-warstwowego jako weryfikatora. Zmierz przyspieszenie ścienne na 100 uzupełnieniach o długości 64. Potwierdź, że wyniki są zgodne z zachłannym weryfikatorem.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Maska przyczynowa | "Trójkąt" | Górnotrójkątna macierz `-inf` dodawana do wyników attention, aby pozycja `i` widziała tylko pozycje `≤ i`. |
| Przewidywanie następnego tokenu | "Funkcja straty" | Entropia krzyżowa rozkładu modelu względem prawdziwego następnego tokenu na każdej pozycji. |
| Autoregresyjność | "Generuj jeden na raz" | Podawaj wyjście z powrotem jako wejście; równoległość tylko podczas trenowania, nie podczas generowania. |
| Logity | "Wyniki przed softmaxem" | Surowe wyjście głowy modelu językowego przed softmaxem; próbkowanie odbywa się na nich. |
| Temperatura | "Pokrętło kreatywności" | Podziel logity przez T; T→0 = zachłanne, T→∞ = jednostajne. |
| Top-p | "Próbkowanie jądrowe (nucleus)" | Obetnij rozkład do najmniejszego zbioru sumującego się do ≥p; próbkuj z tego, co zostało. |
| Min-p | "Lepsze niż top-p" | Zachowaj tokeny, gdzie `p ≥ min_p × max_p`; dostosowuje próg do ostrości rozkładu. |
| Dekodowanie spekulacyjne | "Szkicuj + weryfikuj" | Tani model proponuje N tokenów; duży model weryfikuje równolegle. |
| Teacher forcing | "Triki treningowe" | Podczas trenowania podawaj prawdziwy poprzedni token, a nie predykcję modelu. Standard dla każdego modelu językowego seq2seq. |

## Dalsze czytanie

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 i uczenie się w kontekście.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — artykuł o dekodowaniu spekulacyjnym.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — kanoniczny kod referencyjny causal-LM.