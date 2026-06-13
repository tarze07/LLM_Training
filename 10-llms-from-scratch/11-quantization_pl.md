# Kwantyzacja: Dopasowanie Modeli

> Model 70B w FP16 potrzebuje 140GB. Dwa A100 tylko dla wag. Kwantyzacja do FP8: jedna karta 80GB. INT4: MacBook.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lessons 01-10 (LLMs from Scratch)
**Time:** ~120 minutes

## Learning Objectives

- Zaimplementować kwantyzację symetryczną i asymetryczną z FP16 do INT8 i INT4, w tym skalowanie na tensor i na kanał
- Obliczyć oszczędności pamięci z kwantyzacji i określić, która precyzja mieści się w pamięci VRAM danej karty GPU
- Wyjaśnić różnicę między kwantyzacją potreningową (PTQ) a treningiem świadomym kwantyzacji (QAT)
- Zastosować GPTQ lub AWQ do kwantyzacji rzeczywistego modelu i zmierzyć kompromis dokładność-pamięć na benchmarku

## The Problem

Llama 3 70B ma 70 miliardów parametrów. Każdy parametr to 16-bitowa liczba zmiennoprzecinkowa. To 140 miliardów bajtów. 140GB. Pojedynczy A100 ma 80GB VRAM. Nie możesz nawet załadować wag, a co dopiero uruchomić inferencję, na jednej karcie GPU. Potrzebujesz dwóch A100 za $2/godzinę każdy, aby obsłużyć jeden model.

Ale 16 bitów na parametr jest marnotrawstwem. Większość wag w sieci neuronowej skupia się w pobliżu zera. Pełny zakres dynamiczny FP16 (od 0,000000059 do 65 504) jest prawie całkowicie nieużywany. Jeśli zmierzysz rzeczywistą dystrybucję wag w Llama 3 70B, 95% z nich znajduje się między -0,1 a +0,1. Marnujesz 16 bitów na reprezentowanie wartości, które zmieściłyby się w 4.

Kwantyzacja zastępuje liczby wysokiej precyzji tymi o niższej precyzji. FP16 do FP8 przecina pamięć na pół. FP16 do INT4 przecina ją do ćwierci. Ten model 140GB staje się 35GB. Mieści się na jednej konsumenckiej karcie GPU. Przy kwantyzacji 2-bitowej (agresywnej, stratnej, ale użytecznej dla niektórych zadań) ten sam model działa na laptopie z 16GB.

Kosztem jest dokładność. Każdy usunięty bit niszczy informację. Pytanie brzmi, ile dokładności tracisz i gdzie. Dobrze skwantowany model INT4 zachowuje 95-99% jakości oryginału na większości benchmarków. Naiwna kwantyzacja do INT4 może całkowicie zniszczyć model. Różnica polega na technice.

Kwantyzacje społecznościowe Llama 3 do INT4 z GPTQ pokazują około 1-2 punkty perplexity stracone na WikiText. Mistral wydał punkty kontrolne FP8 Mixtral 8x22B z zerową mierzalną utratą jakości na MMLU. Format GGUF zasila llama.cpp, uruchamiając modele 70B na MacBookach z układami M-series. Kwantyzacja nie jest hackiem. Jest standardową ścieżką wdrożenia dla każdego modelu większego niż 7B.

## The Concept

### Formaty Liczbowe: Co Robi Każdy Bit

Każda liczba zmiennoprzecinkowa ma trzy części: znak, wykładnik i mantysę (zwaną też znaczącą). Znak to jeden bit. Wykładnik określa zakres (jak duża lub mała może być liczba). Mantysa określa precyzję (ile miejsc po przecinku dostajesz).

```
FP32:  [1 sign] [8 exponent] [23 mantissa]  = 32 bits
FP16:  [1 sign] [5 exponent] [10 mantissa]  = 16 bits
BF16:  [1 sign] [8 exponent] [7  mantissa]  = 16 bits
FP8:   [1 sign] [4 exponent] [3  mantissa]  = 8  bits (E4M3)
FP8:   [1 sign] [5 exponent] [2  mantissa]  = 8  bits (E5M2)
INT8:  [1 sign] [7 value]                   = 8  bits (uniform steps)
INT4:  [1 sign] [3 value]                   = 4  bits (16 levels total)
```

**FP32** to pełna precyzja. 23 bity mantysy dają około 7 cyfr dziesiętnych precyzji. Zakres: około 1,2 x 10^-38 do 3,4 x 10^38. Trening kiedyś odbywał się wyłącznie w FP32. Wciąż tak jest dla agregacji (sum bieżących podczas mnożenia macierzy).

**FP16** przecina bity na pół. 10 bitów mantysy daje około 3,3 cyfry dziesiętne. Wykładnik kurczy się do 5 bitów, drastycznie zmniejszając zakres (maksymalna wartość ~65 504). Jest to w porządku dla wag (które skupiają się w pobliżu zera), ale niebezpieczne dla aktywacji i gradientów, które mogą skakać podczas treningu. Trening FP16 wymaga skalowania straty, aby zapobiec niedomiarowi.

**BF16** (Brain Float 16) zachowuje 8-bitowy wykładnik z FP32, ale kurczy mantysę do 7 bitów. Ten sam zakres co FP32, mniejsza precyzja niż FP16. Google zaprojektowało go specjalnie do głębokiego uczenia. Intuicja: zakres ma większe znaczenie niż precyzja dla sieci neuronowych. Gradient 10^-20, który w FP16 ulega niedomiaraowi do zera, w BF16 przetrwa. Waga 0,07342, która w BF16 zaokrągla się do 0,0734, jest wystarczająco blisko. Każdy nowoczesny trening używa BF16 lub mieszanki BF16/FP32.

**FP8** występuje w dwóch wariantach. E4M3 (4 wykładnik, 3 mantysa) jest używane dla wag i aktywacji podczas inferencji. E5M2 (5 wykładnik, 2 mantysa) jest używane dla gradientów podczas treningu, gdzie zakres ma większe znaczenie niż precyzja. Inferencja FP8 na kartach H100 osiąga 30-50% przyspieszenia w stosunku do FP16 z znikomą utratą jakości.

**INT8** to format całkowitoliczbowy. Żadnego wykładnika, żadnej mantysy. Tylko 256 równomiernie rozmieszczonych wartości od -128 do 127. Potrzebujesz współczynnika skali, aby odwzorować zmiennoprzecinkowe wagi w ten zakres. Zaleta: arytmetyka całkowitoliczbowa jest szybsza i bardziej energooszczędna niż zmiennoprzecinkowa. Mnożenie macierzy INT8 na A100 działa z wydajnością 624 TOPS wobec 312 TFLOPS dla FP16.

**INT4** idzie dalej. Tylko 16 możliwych wartości. Współczynnik skali robi ciężką pracę. Jakość zależy wyłącznie od tego, jak wybierzesz skalę i które wagi kwantyzujesz. Najnowocześniejsze metody INT4 (GPTQ, AWQ) zachowują 95%+ oryginalnej jakości modelu.

```mermaid
graph LR
    subgraph Formats["Number Format Landscape"]
        direction TB
        FP32["FP32\n32 bits\n4 bytes/param\nTraining gold standard"]
        BF16["BF16\n16 bits\n2 bytes/param\nTraining default"]
        FP16["FP16\n16 bits\n2 bytes/param\nInference baseline"]
        FP8["FP8\n8 bits\n1 byte/param\n30-50% faster"]
        INT8["INT8\n8 bits\n1 byte/param\n2x throughput"]
        INT4["INT4\n4 bits\n0.5 bytes/param\n4x compression"]
    end

    FP32 -->|"training"| BF16
    BF16 -->|"inference"| FP16
    FP16 -->|"H100 native"| FP8
    FP16 -->|"server deploy"| INT8
    FP16 -->|"edge/laptop"| INT4

    style FP32 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style BF16 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style FP16 fill:#1a1a2e,stroke:#ffa500,color:#fff
    style FP8 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style INT8 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style INT4 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Jak Działa Kwantyzacja

Podstawowa operacja jest prosta. Weź tensor zmiennoprzecinkowych wartości, znajdź współczynnik skali, pomnóż, zaokrąglij do najbliższej liczby całkowitej i przechowuj liczby całkowite wraz ze współczynnikiem skali.

**Kwantyzacja:**
```
scale = max(abs(tensor)) / max_int_value
quantized = round(tensor / scale)
```

**Dekwantyzacja:**
```
reconstructed = quantized * scale
```

Dla INT8 z symetrycznym zakresem (-127 do 127):
```
scale = max(abs(tensor)) / 127
quantized = clamp(round(tensor / scale), -128, 127)
```

Błąd to błąd zaokrąglenia. Każda wartość może być odchylona co najwyżej o `scale / 2`. Całkowity błąd w warstwie zależy od tego, ile wag masz i jak wrażliwy jest model na perturbacje w tych wagach.

**Kwantyzacja na tensor vs na kanał.** Na tensor używa jednego współczynnika skali dla całej macierzy wag. Prosta, ale stratna: jeśli jedna kolumna ma duże wartości, a inna małe, małe wartości tracą większość swojej precyzji. Na kanał używa jednego współczynnika skali na kanał wyjściowy (na wiersz lub kolumnę macierzy wag). Więcej narzutu (przechowujesz N współczynników skali zamiast 1), ale dramatycznie lepsza jakość. Każda produkcyjna metoda kwantyzacji używa na kanał lub drobniejszej ziarnistości.

**Kwantyzacja asymetryczna** dodaje przesunięcie punktu zerowego: `quantized = round(tensor / scale) + zero_point`. Obsługuje dystrybucje, które nie są wyśrodkowane wokół zera. Aktywacje ReLU, na przykład, są zawsze nieujemne. Kwantyzacja symetryczna marnuje połowę zakresu całkowitoliczbowego na ujemne wartości, które nigdy nie występują. Kwantyzacja asymetryczna odwzorowuje rzeczywisty zakres [min, max] na pełny zakres całkowitoliczbowy.

### Hierarchia Wrażliwości

Nie wszystko w modelu toleruje kwantyzację jednakowo. Istnieje wyraźna hierarchia.

**Wagi (najbardziej odporne).** Wagi modelu zmieniają się powoli podczas treningu i mają w przybliżeniu rozkład Gaussa wyśrodkowany wokół zera. Kwantyzują się dobrze. Wagi INT8 ze skalami na kanał dają prawie bezstratne wyniki. INT4 wymaga bardziej wyrafinowanych metod, ale działa.

**Aktywacje (umiarkowana wrażliwość).** Aktywacje to wartości pośrednie przepływające przez sieć podczas inferencji. Mają szerszy zakres dynamiczny niż wagi i zawierają wartości odstające. Pojedyncza głowa uwagi może produkować wartości aktywacji 100x większe niż średnia. Te wartości odstające są krytyczne dla jakości modelu. Kwantyzowanie ich naiwnie niszczy informację. Rozwiązania: utrzymuj kanały odstające w wyższej precyzji (LLM.int8()), używaj skal aktywacji na token lub na kanał.

**Pamięć podręczna KV (wysoka wrażliwość).** Pamięć podręczna klucz-wartość przechowuje stany uwagi dla wszystkich poprzednich tokenów. Przy długich kontekstach pamięć KV dominuje pamięć. Dla modelu 70B przy kontekście 32K, sama pamięć KV to 40GB w FP16. Kwantyzowanie pamięci KV do FP8 lub INT8 oszczędza ogromną ilość pamięci, ale każdy błąd kumuluje się we wszystkich przyszłych obliczeniach uwagi. Wpływ na jakość skaluje się z długością sekwencji.

**Logity uwagi (najbardziej wrażliwe).** Softmax w uwadze jest bardzo wrażliwy na małe zmiany w swoich wejściach. Błąd kwantyzacji 0,01 w logicie przed softmaxem może znacząco przesunąć rozkład uwagi. Większość schematów kwantyzacji utrzymuje obliczenia uwagi w wyższej precyzji (FP16 lub BF16), nawet gdy wszystko inne jest skwantowane.

```mermaid
graph TD
    subgraph Sensitivity["Quantization Sensitivity (Low to High)"]
        direction LR
        W["Weights\nGaussian, near zero\nINT4 works well"]
        A["Activations\nWider range, outliers\nINT8 with care"]
        KV["KV Cache\nErrors compound\nFP8 or INT8"]
        ATT["Attention Logits\nSoftmax amplifies error\nKeep in FP16"]
    end

    W -->|"safe"| A
    A -->|"careful"| KV
    KV -->|"dangerous"| ATT

    style W fill:#1a1a2e,stroke:#51cf66,color:#fff
    style A fill:#1a1a2e,stroke:#ffa500,color:#fff
    style KV fill:#1a1a2e,stroke:#e94560,color:#fff
    style ATT fill:#1a1a2e,stroke:#ff0000,color:#fff
```

### PTQ vs QAT

**Kwantyzacja Potreningowa (PTQ)** kwantyzuje już wytrenowany model. Żadnego ponownego treningu. Bierzesz wagi FP16, obliczasz współczynniki skali, zaokrąglasz i wdrażasz. Szybko (minuty do godzin) i tanio. Działa dobrze dla INT8 i FP8. Dla INT4 naiwny PTQ często zawodzi katastrofalnie, ponieważ błędy zaokrągleń się kumulują. Zaawansowane metody PTQ (GPTQ, AWQ) używają danych kalibracyjnych, aby zminimalizować błąd kwantyzacji.

**Trening Świadomy Kwantyzacji (QAT)** wstawia fikcyjne operacje kwantyzacji do przejścia w przód podczas treningu. Model uczy się umieszczać swoje wagi tam, gdzie błędy zaokrągleń są małe. Gradienty przepływają przez fikcyjną kwantyzację przy użyciu estymatora prostego przejścia (STE): udawaj, że operacja zaokrąglania ma gradient 1. QAT produkuje lepsze modele INT4 i INT2 niż PTQ, ale wymaga pełnego przebiegu treningowego. Google użył QAT dla wydajnego serwowania Gemini. Meta użyła QAT dla niektórych celów wdrożenia Llama.

| Aspekt | PTQ | QAT |
|--------|-----|-----|
| Koszt | Minuty do godzin | Pełny przebieg treningowy |
| Jakość przy INT8 | Doskonała (< 0.1% straty) | Doskonała |
| Jakość przy INT4 | Dobra z GPTQ/AWQ (1-3% straty) | Lepiej (< 1% straty) |
| Jakość przy INT2 | Słaba | Użyteczna dla niektórych zadań |
| Dane kalibracyjne | 128-1024 przykładów | Pełny zbiór danych treningowych |
| Kiedy używać | Wdrożenie, iteracja | Maksymalna jakość przy niskiej szerokości bitowej |

### GPTQ, AWQ, GGUF

**GPTQ (GPT Quantization)** to jednorazowa metoda PTQ. Kwantyzuje wagi jedna warstwa na raz, używając małego zestawu danych kalibracyjnych (128 przykładów to typowe) do pomiaru Hesjanu (informacji drugiego rzędu o tym, jak wrażliwe jest wyjście na każdą wagę). Wagi, które Hesjan wskazuje jako ważne, są kwantyzowane ostrożniej. GPTQ było pierwszą metodą, która uczyniła kwantyzację INT4 praktyczną dla LLM. TheBloke na Hugging Face spopularyzował GPTQ, wydając skwantowane wersje setek modeli.

**AWQ (Activation-Aware Weight Quantization)** obserwuje, że mały ułamek wag (około 1%) jest nieproporcjonalnie ważny, ponieważ mnoży się z dużymi wartościami aktywacji. AWQ identyfikuje te istotne wagi przy użyciu danych kalibracyjnych i skaluje je w górę przed kwantyzacją (następnie skaluje odpowiednie aktywacje w dół). To utrzymuje ważne wagi w zakresie, w którym kwantyzacja INT4 jest dokładna. AWQ zazwyczaj dorównuje lub nieco przewyższa jakość GPTQ, będąc jednocześnie 1,5-2x szybszym w zastosowaniu.

**GGUF (GPT-Generated Unified Format)** to format pliku używany przez llama.cpp i jego ekosystem. Obsługuje mieszaną kwantyzację: różne warstwy dostają różne szerokości bitowe. Pierwsza i ostatnia warstwa (embedding i głowa wyjściowa) są zazwyczaj utrzymywane w wyższej precyzji. Środkowe warstwy dostają INT4 lub INT3. Pliki GGUF są samodzielne: wagi, tokenizer, metadane wszystko w jednym pliku. Format jest zaprojektowany do inferencji na CPU i Apple Silicon, gdzie załadowanie całego modelu do pamięci i uruchomienie mnożenia macierzy na CPU lub GPU Metal jest standardową ścieżką. Q4_K_M to najpopularniejszy wariant kwantyzacji GGUF, równoważący jakość i rozmiar.

```mermaid
graph TD
    subgraph Methods["Quantization Methods"]
        direction TB
        GPTQ_["GPTQ\nHessian-guided\nPer-layer optimization\nPopular on HuggingFace"]
        AWQ_["AWQ\nActivation-aware\nSalient weight scaling\n1.5-2x faster than GPTQ"]
        GGUF_["GGUF\nMixed precision\nCPU + Metal optimized\nllama.cpp ecosystem"]
    end

    subgraph Use["Best For"]
        GPU["GPU inference\n(CUDA, ROCm)"]
        EDGE["Edge / Laptop\n(CPU, Metal)"]
    end

    GPTQ_ --> GPU
    AWQ_ --> GPU
    GGUF_ --> EDGE

    style GPTQ_ fill:#1a1a2e,stroke:#ffa500,color:#fff
    style AWQ_ fill:#1a1a2e,stroke:#51cf66,color:#fff
    style GGUF_ fill:#1a1a2e,stroke:#0f3460,color:#fff
```

### Pomiar Jakości

Skąd wiesz, czy twój skwantowany model jest wciąż dobry?

**Perplexity.** Najpopularniejsza metryka. Niższe jest lepsze. Oblicz perplexity na wstrzymanym zbiorze danych (WikiText-2 jest standardem) zarówno dla oryginalnego, jak i skwantowanego modelu. Delta mówi ci, ile informacji zniszczyła kwantyzacja. Zasady kciuka: delta < 0,5 jest doskonała, 0,5-1,0 jest dobra, 1,0-2,0 jest akceptowalna dla większości zadań, > 2,0 oznacza, że coś poszło nie tak.

**Benchmarki specyficzne dla zadania.** Uruchom skwantowany model na MMLU, HumanEval, GSM8K lub swoim niestandardowym zestawie ewaluacyjnym. Porównaj z oryginałem. Kwantyzacja wpływa na różne zdolności nierównomiernie. Zadania matematyczne i kodowe są bardziej wrażliwe na utratę precyzji niż ogólna wiedza.

**Porównanie wyników.** Wygeneruj odpowiedzi z obu modeli na te same prompty i porównaj. LLM-jako-sędzia (Lekcja 10) działa tu dobrze. Oblicz wskaźnik wygranych: jaki ułamek promptów skwantowany model dorównuje lub bije oryginał?

**Opóźnienie i przepustowość.** Kwantyzacja istnieje, aby modele były szybsze i tańsze. Zmierz tokeny na sekundę, czas do pierwszego tokena i użycie pamięci. Skwantowany model, który jest wolniejszy niż oryginał, jest gorszy niż bezużyteczny.

| Model | Format | Rozmiar | Perplexity (WikiText-2) | MMLU | Tokeny/s (A100) |
|-------|--------|------|------------------------|------|-------------------|
| Llama 3 70B | FP16 | 140GB | 3.12 | 79.5% | 38 |
| Llama 3 70B | FP8 | 70GB | 3.14 | 79.3% | 55 |
| Llama 3 70B | GPTQ INT4 | 35GB | 4.32 | 77.8% | 72 |
| Llama 3 70B | AWQ INT4 | 35GB | 4.18 | 78.1% | 75 |
| Llama 3 70B | GGUF Q4_K_M | 40GB | 4.25 | 77.9% | 28 (CPU) |

Wzór: FP8 jest prawie za darmo. INT4 kosztuje 1-2 punkty MMLU, ale podwaja przepustowość i ćwiartkuje pamięć. Kompromis jest wart dla prawie każdego wdrożenia.

### Rzeczywiste Liczby

FP16 do FP8 na H100: 30-50% przyspieszenia inferencji, < 0,1% utraty jakości. To jest kwantyzacja bez zastanowienia. Każde wdrożenie na H100 powinno jej używać.

FP16 do INT8 (LLM.int8()): 2x redukcja pamięci, < 0,5% utraty jakości. Podejście mieszanej precyzji utrzymuje odstające cechy w FP16, kwantyzując wszystko inne do INT8.

FP16 do INT4 (GPTQ/AWQ): 4x redukcja pamięci, 1-3% utraty jakości w zależności od modelu i metody. Umożliwia modele 70B na pojedynczej karcie GPU 48GB.

FP16 do INT4 (GGUF Q4_K_M): 3,5x redukcja pamięci, 1-2% utraty jakości. Zoptymalizowany do inferencji na CPU. Model 70B w Q4_K_M to około 40GB i działa z prędkością 10-15 tokenów/sekundę na M3 Max z 64GB.

FP16 do INT2: 8x redukcja pamięci, 5-15% utraty jakości. Realistyczny tylko dla konkretnych wąskich zadań, gdzie możesz tolerować degradację. Granica badań, nie gotowy do produkcji do ogólnego użytku.

```figure
quantization
```

## Build It

### Krok 1: Reprezentacje Formatu Liczbowego

Zbuduj reprezentację na poziomie bitów każdego formatu, aby zobaczyć dokładnie, co robią znak, wykładnik i mantysa.

```python
import numpy as np


def float_to_fp32_bits(value):
    bits = np.float32(value).view(np.uint32)
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '08b'),
            "mantissa_bits": format(int(mantissa), '023b'),
            "value": float(value),
            "actual_exponent": int(exponent) - 127}


def float_to_fp16_bits(value):
    fp16 = np.float16(value)
    bits = fp16.view(np.uint16)
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '05b'),
            "mantissa_bits": format(int(mantissa), '010b'),
            "value": float(fp16),
            "actual_exponent": int(exponent) - 15}


def float_to_bf16_bits(value):
    fp32_bits = np.float32(value).view(np.uint32)
    bf16_bits = (fp32_bits >> 16).astype(np.uint16)
    sign = (bf16_bits >> 15) & 1
    exponent = (bf16_bits >> 7) & 0xFF
    mantissa = bf16_bits & 0x7F
    reconstructed = np.uint32(bf16_bits.astype(np.uint32) << 16).view(np.float32)
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '08b'),
            "mantissa_bits": format(int(mantissa), '07b'),
            "value": float(reconstructed),
            "actual_exponent": int(exponent) - 127}


def simulate_fp8_e4m3(value):
    sign = 1 if value < 0 else 0
    abs_val = abs(value)
    max_val = 448.0
    abs_val = min(abs_val, max_val)
    if abs_val == 0:
        return {"sign": sign, "exponent": 0, "mantissa": 0, "value": 0.0,
                "exponent_bits": "0000", "mantissa_bits": "000"}
    exp = int(np.floor(np.log2(abs_val)))
    exp = max(-6, min(8, exp))
    mantissa_val = abs_val / (2.0 ** exp) - 1.0
    mantissa_quant = round(mantissa_val * 8) / 8
    mantissa_quant = max(0, min(0.875, mantissa_quant))
    reconstructed = (1.0 + mantissa_quant) * (2.0 ** exp)
    if sign:
        reconstructed = -reconstructed
    mantissa_int = int(round(mantissa_quant * 8))
    return {"sign": sign, "exponent": exp + 7, "mantissa": mantissa_int,
            "exponent_bits": format(exp + 7, '04b'),
            "mantissa_bits": format(mantissa_int, '03b'),
            "value": float(reconstructed),
            "actual_exponent": exp}


def display_format_comparison(value):
    fp32 = float_to_fp32_bits(value)
    fp16 = float_to_fp16_bits(value)
    bf16 = float_to_bf16_bits(value)
    fp8 = simulate_fp8_e4m3(value)

    print(f"\n  Value: {value}")
    print(f"  {'Format':<8} {'Stored Value':>14} {'Error':>12} {'Sign':>5} {'Exp Bits':>10} {'Man Bits':>25}")
    print(f"  {'-'*76}")
    print(f"  {'FP32':<8} {fp32['value']:>14.6f} {abs(fp32['value'] - value):>12.8f} {fp32['sign']:>5} {fp32['exponent_bits']:>10} {fp32['mantissa_bits']:>25}")
    print(f"  {'FP16':<8} {fp16['value']:>14.6f} {abs(fp16['value'] - value):>12.8f} {fp16['sign']:>5} {fp16['exponent_bits']:>10} {fp16['mantissa_bits']:>25}")
    print(f"  {'BF16':<8} {bf16['value']:>14.6f} {abs(bf16['value'] - value):>12.8f} {bf16['sign']:>5} {bf16['exponent_bits']:>10} {bf16['mantissa_bits']:>25}")
    print(f"  {'FP8e4m3':<8} {fp8['value']:>14.6f} {abs(fp8['value'] - value):>12.8f} {fp8['sign']:>5} {fp8['exponent_bits']:>10} {fp8['mantissa_bits']:>25}")
```

### Krok 2: Kwantyzacja Symetryczna (Na Tensor i Na Kanał)

Podstawowe operacje kwantyzacji. Na tensor używa jednej skali dla całej macierzy. Na kanał używa jednej skali na wiersz lub kolumnę.

```python
def quantize_symmetric(tensor, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = np.max(np.abs(tensor))
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int32), 1.0
    scale = abs_max / qmax
    quantized = np.clip(np.round(tensor / scale), qmin, qmax).astype(np.int32)
    return quantized, float(scale)


def dequantize_symmetric(quantized, scale):
    return quantized.astype(np.float64) * scale


def quantize_per_channel(tensor, num_bits=8, axis=0):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    if axis == 0:
        abs_max = np.max(np.abs(tensor), axis=1, keepdims=True)
    else:
        abs_max = np.max(np.abs(tensor), axis=0, keepdims=True)

    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scales = abs_max / qmax
    quantized = np.clip(np.round(tensor / scales), qmin, qmax).astype(np.int32)
    return quantized, scales.squeeze()


def dequantize_per_channel(quantized, scales, axis=0):
    if axis == 0:
        return quantized.astype(np.float64) * scales.reshape(-1, 1)
    else:
        return quantized.astype(np.float64) * scales.reshape(1, -1)


def quantize_asymmetric(tensor, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1
    t_min = np.min(tensor)
    t_max = np.max(tensor)
    if t_max == t_min:
        return np.zeros_like(tensor, dtype=np.int32), 1.0, 0
    scale = (t_max - t_min) / (qmax - qmin)
    zero_point = int(np.round(qmin - t_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    quantized = np.clip(np.round(tensor / scale + zero_point), qmin, qmax).astype(np.int32)
    return quantized, float(scale), int(zero_point)


def dequantize_asymmetric(quantized, scale, zero_point):
    return (quantized.astype(np.float64) - zero_point) * scale
```

### Krok 3: Pomiar Jakości

Zmierz, ile informacji niszczy kwantyzacja. Średni błąd kwadratowy, stosunek sygnału do szumu i podobieństwo cosinusowe między oryginalnym a zrekonstruowanym tensorem.

```python
def quantization_error(original, reconstructed):
    diff = original - reconstructed
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(np.abs(diff)))
    signal_power = float(np.mean(original ** 2))
    snr_db = 10 * np.log10(signal_power / max(mse, 1e-20))

    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    norm_orig = np.linalg.norm(orig_flat)
    norm_recon = np.linalg.norm(recon_flat)
    if norm_orig == 0 or norm_recon == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(orig_flat, recon_flat) / (norm_orig * norm_recon))

    return {"mse": mse, "rmse": rmse, "max_error": max_error,
            "snr_db": float(snr_db), "cosine_similarity": cosine_sim}


def compare_quantization_methods(tensor, num_bits=8):
    q_pt, s_pt = quantize_symmetric(tensor, num_bits)
    recon_pt = dequantize_symmetric(q_pt, s_pt)
    err_pt = quantization_error(tensor, recon_pt)

    q_pc, s_pc = quantize_per_channel(tensor, num_bits, axis=0)
    recon_pc = dequantize_per_channel(q_pc, s_pc, axis=0)
    err_pc = quantization_error(tensor, recon_pc)

    q_asym, s_asym, zp = quantize_asymmetric(tensor, num_bits)
    recon_asym = dequantize_asymmetric(q_asym, s_asym, zp)
    err_asym = quantization_error(tensor, recon_asym)

    print(f"\n  Quantization Comparison ({num_bits}-bit, tensor shape {tensor.shape}):")
    print(f"  {'Method':<20} {'MSE':>12} {'SNR (dB)':>10} {'Cosine Sim':>12} {'Max Error':>12}")
    print(f"  {'-'*68}")
    print(f"  {'Per-tensor sym':<20} {err_pt['mse']:>12.8f} {err_pt['snr_db']:>10.2f} {err_pt['cosine_similarity']:>12.8f} {err_pt['max_error']:>12.8f}")
    print(f"  {'Per-channel sym':<20} {err_pc['mse']:>12.8f} {err_pc['snr_db']:>10.2f} {err_pc['cosine_similarity']:>12.8f} {err_pc['max_error']:>12.8f}")
    print(f"  {'Asymmetric':<20} {err_asym['mse']:>12.8f} {err_asym['snr_db']:>10.2f} {err_asym['cosine_similarity']:>12.8f} {err_asym['max_error']:>12.8f}")

    return {"per_tensor": err_pt, "per_channel": err_pc, "asymmetric": err_asym}
```

### Krok 4: Przeszukiwanie Szerokości Bitowej

Kwantyzuj ten sam tensor przy różnych szerokościach bitowych (2, 3, 4, 8, 16) i mierz jakość na każdym poziomie. To pokazuje dokładnie, gdzie jest klif jakości.

```python
def bit_width_sweep(tensor):
    print(f"\n  Bit-Width Sweep (tensor shape {tensor.shape}):")
    print(f"  {'Bits':>6} {'Levels':>8} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12} {'Compression':>12}")
    print(f"  {'-'*64}")

    results = []
    for bits in [2, 3, 4, 8, 16]:
        q, s = quantize_per_channel(tensor, bits, axis=0)
        recon = dequantize_per_channel(q, s, axis=0)
        err = quantization_error(tensor, recon)
        levels = 2 ** bits
        compression = 32.0 / bits

        print(f"  {bits:>6} {levels:>8} {err['mse']:>14.8f} {err['snr_db']:>10.2f} {err['cosine_similarity']:>12.8f} {compression:>11.1f}x")
        results.append({"bits": bits, "levels": levels, "error": err, "compression": compression})

    return results
```

### Krok 5: Eksperyment Wrażliwości

Symuluj kwantyzację różnych części transformera i mierz, które komponenty są najbardziej wrażliwe. To demonstruje hierarchię wrażliwości: wagi < aktywacje < pamięć KV < uwaga.

```python
def simulate_transformer_layer(input_data, weights, kv_scale=1.0):
    hidden = input_data @ weights["qkv"]
    seq_len = hidden.shape[1]
    d_model = weights["qkv"].shape[1] // 3
    q, k, v = hidden[:, :, :d_model], hidden[:, :, d_model:2*d_model], hidden[:, :, 2*d_model:]

    attn_scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_model) * kv_scale
    attn_max = np.max(attn_scores, axis=-1, keepdims=True)
    attn_exp = np.exp(attn_scores - attn_max)
    attn_weights = attn_exp / np.sum(attn_exp, axis=-1, keepdims=True)

    attn_output = attn_weights @ v
    output = attn_output @ weights["out"]
    return output, {"q": q, "k": k, "v": v, "attn_scores": attn_scores,
                    "attn_weights": attn_weights, "attn_output": attn_output}


def sensitivity_experiment(batch_size=2, seq_len=16, d_model=64, num_bits=8):
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, d_model) * 0.1

    weights = {
        "qkv": np.random.randn(d_model, 3 * d_model) * (2.0 / d_model) ** 0.5,
        "out": np.random.randn(d_model, d_model) * (2.0 / d_model) ** 0.5,
    }

    baseline_output, baseline_internals = simulate_transformer_layer(input_data, weights)

    experiments = {}

    q_qkv, s_qkv = quantize_per_channel(weights["qkv"], num_bits, axis=0)
    q_out, s_out = quantize_per_channel(weights["out"], num_bits, axis=0)
    quantized_weights = {
        "qkv": dequantize_per_channel(q_qkv, s_qkv, axis=0),
        "out": dequantize_per_channel(q_out, s_out, axis=0),
    }
    weight_quant_output, _ = simulate_transformer_layer(input_data, quantized_weights)
    experiments["Weights only"] = quantization_error(baseline_output, weight_quant_output)

    _, fresh_internals = simulate_transformer_layer(input_data, weights)
    q_act, s_act = quantize_per_channel(
        fresh_internals["attn_output"].reshape(-1, d_model), num_bits, axis=0
    )
    quant_attn_out = dequantize_per_channel(q_act, s_act, axis=0).reshape(batch_size, seq_len, d_model)
    act_quant_output = quant_attn_out @ weights["out"]
    experiments["Activations only"] = quantization_error(baseline_output, act_quant_output)

    q_k, s_k = quantize_per_channel(fresh_internals["k"].reshape(-1, d_model), num_bits, axis=0)
    q_v, s_v = quantize_per_channel(fresh_internals["v"].reshape(-1, d_model), num_bits, axis=0)
    quant_k = dequantize_per_channel(q_k, s_k, axis=0).reshape(batch_size, seq_len, d_model)
    quant_v = dequantize_per_channel(q_v, s_v, axis=0).reshape(batch_size, seq_len, d_model)
    attn_scores_kv = (fresh_internals["q"] @ quant_k.transpose(0, 2, 1)) / np.sqrt(d_model)
    attn_max_kv = np.max(attn_scores_kv, axis=-1, keepdims=True)
    attn_exp_kv = np.exp(attn_scores_kv - attn_max_kv)
    attn_weights_kv = attn_exp_kv / np.sum(attn_exp_kv, axis=-1, keepdims=True)
    kv_quant_output = (attn_weights_kv @ quant_v) @ weights["out"]
    experiments["KV cache only"] = quantization_error(baseline_output, kv_quant_output)

    noise_scale = np.std(fresh_internals["attn_scores"]) * 0.05
    noisy_scores = fresh_internals["attn_scores"] + np.random.randn(*fresh_internals["attn_scores"].shape) * noise_scale
    noisy_max = np.max(noisy_scores, axis=-1, keepdims=True)
    noisy_exp = np.exp(noisy_scores - noisy_max)
    noisy_weights = noisy_exp / np.sum(noisy_exp, axis=-1, keepdims=True)
    attn_quant_output = (noisy_weights @ fresh_internals["v"]) @ weights["out"]
    experiments["Attention logits (5% noise)"] = quantization_error(baseline_output, attn_quant_output)

    print(f"\n  Sensitivity Experiment ({num_bits}-bit quantization):")
    print(f"  {'Component':<30} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12}")
    print(f"  {'-'*68}")
    for name, err in sorted(experiments.items(), key=lambda x: x[1]["mse"]):
        print(f"  {name:<30} {err['mse']:>14.8f} {err['snr_db']:>10.2f} {err['cosine_similarity']:>12.8f}")

    return experiments
```

### Krok 6: Symulowany GPTQ

GPTQ kwantyzuje jedną kolumnę na raz, używając Hesjanu do podjęcia decyzji, jak rozłożyć błąd zaokrąglenia. To jest uproszczona wersja, która oddaje główną ideę: użyj danych kalibracyjnych do pomiaru ważności wag, a następnie kwantyzuj najmniej ważne wagi bardziej agresywnie.

```python
def simulated_gptq(weight_matrix, calibration_inputs, num_bits=4):
    n_in, n_out = weight_matrix.shape
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    H = np.zeros((n_in, n_in))
    for x in calibration_inputs:
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        for row in range(x.shape[0]):
            xi = x[row].reshape(-1, 1)
            H += xi @ xi.T
    H /= len(calibration_inputs)
    H += np.eye(n_in) * 1e-4

    weight_importance = np.diag(H)

    quantized = np.zeros_like(weight_matrix, dtype=np.int32)
    scales = np.zeros(n_out)
    errors = np.zeros(n_out)

    W = weight_matrix.copy()

    for col in range(n_out):
        w_col = W[:, col]
        abs_max = np.max(np.abs(w_col))
        if abs_max == 0:
            scales[col] = 1.0
            continue
        scale = abs_max / qmax
        scales[col] = scale

        q_col = np.clip(np.round(w_col / scale), qmin, qmax).astype(np.int32)
        quantized[:, col] = q_col

        quant_error = w_col - q_col * scale
        errors[col] = np.sqrt(np.mean(quant_error ** 2))

        if col < n_out - 1:
            importance_weights = weight_importance / (np.max(weight_importance) + 1e-10)
            for next_col in range(col + 1, min(col + 4, n_out)):
                compensation = quant_error * importance_weights * 0.1
                W[:, next_col] += compensation

    return quantized, scales, {"column_errors": errors,
                               "mean_error": float(np.mean(errors)),
                               "max_error": float(np.max(errors))}


def dequantize_gptq(quantized, scales):
    result = np.zeros_like(quantized, dtype=np.float64)
    for col in range(quantized.shape[1]):
        result[:, col] = quantized[:, col] * scales[col]
    return result
```

### Krok 7: Symulacja AWQ

AWQ identyfikuje istotne wagi (te, które mnożą się z dużymi aktywacjami) i chroni je przez skalowanie przed kwantyzacją.

```python
def simulated_awq(weight_matrix, calibration_inputs, num_bits=4, salient_fraction=0.01):
    n_in, n_out = weight_matrix.shape
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    activation_magnitudes = np.zeros(n_in)
    for x in calibration_inputs:
        if x.ndim == 1:
            activation_magnitudes += np.abs(x)
        else:
            activation_magnitudes += np.mean(np.abs(x), axis=0)
    activation_magnitudes /= len(calibration_inputs)

    n_salient = max(1, int(n_in * salient_fraction))
    salient_indices = np.argsort(activation_magnitudes)[-n_salient:]

    scale_factors = np.ones(n_in)
    for idx in salient_indices:
        col_max = np.max(np.abs(weight_matrix[idx, :]))
        if col_max > 0:
            scale_factors[idx] = min(4.0, 1.0 / (col_max + 1e-8) * np.mean(np.abs(weight_matrix)))

    scaled_weights = weight_matrix * scale_factors.reshape(-1, 1)

    quantized, scales = quantize_per_channel(scaled_weights, num_bits, axis=0)
    dequantized = dequantize_per_channel(quantized, scales, axis=0)

    result = dequantized / scale_factors.reshape(-1, 1)

    err = quantization_error(weight_matrix, result)

    return result, {"salient_indices": salient_indices,
                    "scale_factors": scale_factors[salient_indices],
                    "error": err,
                    "n_salient": n_salient}
```

### Krok 8: Pełny Potok

Połącz wszystko. Porównaj naiwną kwantyzację, na kanał, GPTQ i AWQ na tej samej macierzy wag.

```python
def full_quantization_comparison(d_in=256, d_out=512, num_bits=4, n_calibration=32):
    np.random.seed(42)

    weight = np.random.randn(d_in, d_out) * 0.02
    outlier_rows = np.random.choice(d_in, size=5, replace=False)
    weight[outlier_rows] *= 10

    calibration = [np.random.randn(8, d_in) * 0.1 for _ in range(n_calibration)]

    q_naive, s_naive = quantize_symmetric(weight, num_bits)
    recon_naive = dequantize_symmetric(q_naive, s_naive)
    err_naive = quantization_error(weight, recon_naive)

    q_pc, s_pc = quantize_per_channel(weight, num_bits, axis=0)
    recon_pc = dequantize_per_channel(q_pc, s_pc, axis=0)
    err_pc = quantization_error(weight, recon_pc)

    q_gptq, s_gptq, gptq_info = simulated_gptq(weight, calibration, num_bits)
    recon_gptq = dequantize_gptq(q_gptq, s_gptq)
    err_gptq = quantization_error(weight, recon_gptq)

    recon_awq, awq_info = simulated_awq(weight, calibration, num_bits)
    err_awq = awq_info["error"]

    print(f"\n  Full Quantization Comparison ({num_bits}-bit, {d_in}x{d_out} matrix)")
    print(f"  Matrix has {len(outlier_rows)} outlier rows (10x scale)")
    print()
    print(f"  {'Method':<20} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12}")
    print(f"  {'-'*58}")
    print(f"  {'Naive per-tensor':<20} {err_naive['mse']:>14.8f} {err_naive['snr_db']:>10.2f} {err_naive['cosine_similarity']:>12.8f}")
    print(f"  {'Per-channel':<20} {err_pc['mse']:>14.8f} {err_pc['snr_db']:>10.2f} {err_pc['cosine_similarity']:>12.8f}")
    print(f"  {'Simulated GPTQ':<20} {err_gptq['mse']:>14.8f} {err_gptq['snr_db']:>10.2f} {err_gptq['cosine_similarity']:>12.8f}")
    print(f"  {'Simulated AWQ':<20} {err_awq['mse']:>14.8f} {err_awq['snr_db']:>10.2f} {err_awq['cosine_similarity']:>12.8f}")

    test_input = np.random.randn(4, d_in) * 0.1
    baseline = test_input @ weight
    output_naive = test_input @ recon_naive
    output_pc = test_input @ recon_pc
    output_gptq = test_input @ recon_gptq
    output_awq = test_input @ recon_awq

    print(f"\n  End-to-End Output Error (matmul with test input):")
    print(f"  {'Method':<20} {'Output MSE':>14} {'Output Cosine':>14}")
    print(f"  {'-'*50}")
    for name, output in [("Naive", output_naive), ("Per-channel", output_pc),
                          ("GPTQ", output_gptq), ("AWQ", output_awq)]:
        out_err = quantization_error(baseline, output)
        print(f"  {name:<20} {out_err['mse']:>14.8f} {out_err['cosine_similarity']:>14.8f}")

    return {"naive": err_naive, "per_channel": err_pc, "gptq": err_gptq, "awq": err_awq}


def memory_calculator(num_params_billions, bits_per_param):
    bytes_per_param = bits_per_param / 8
    total_bytes = num_params_billions * 1e9 * bytes_per_param
    total_gb = total_bytes / (1024 ** 3)
    return total_gb


def print_memory_table():
    print("\n  Memory Requirements by Model and Precision:")
    print(f"  {'Model':<15} {'FP32':>8} {'FP16':>8} {'FP8':>8} {'INT8':>8} {'INT4':>8} {'INT2':>8}")
    print(f"  {'-'*64}")
    for name, params in [("7B", 7), ("13B", 13), ("34B", 34), ("70B", 70), ("405B", 405)]:
        fp32 = memory_calculator(params, 32)
        fp16 = memory_calculator(params, 16)
        fp8 = memory_calculator(params, 8)
        int8 = memory_calculator(params, 8)
        int4 = memory_calculator(params, 4)
        int2 = memory_calculator(params, 2)
        print(f"  {name:<15} {fp32:>7.1f}G {fp16:>7.1f}G {fp8:>7.1f}G {int8:>7.1f}G {int4:>7.1f}G {int2:>7.1f}G")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("QUANTIZATION: MAKING MODELS FIT")
    print("=" * 70)

    print("\nSTEP 1: Number Format Comparison")
    print("-" * 50)
    for val in [0.1, 3.14159, -0.00073, 42.5, 0.0000012]:
        display_format_comparison(val)

    print("\n\nSTEP 2: Memory Requirements")
    print("-" * 50)
    print_memory_table()

    print("\n\nSTEP 3: Quantization Methods Comparison")
    print("-" * 50)
    weight_matrix = np.random.randn(128, 256) * 0.02
    weight_matrix[0] *= 15
    weight_matrix[42] *= 8
    compare_quantization_methods(weight_matrix, num_bits=8)
    compare_quantization_methods(weight_matrix, num_bits=4)

    print("\n\nSTEP 4: Bit-Width Sweep")
    print("-" * 50)
    sweep_tensor = np.random.randn(64, 128) * 0.05
    bit_width_sweep(sweep_tensor)

    print("\n\nSTEP 5: Sensitivity Experiment")
    print("-" * 50)
    print("\n  INT8:")
    sensitivity_experiment(num_bits=8)
    print("\n  INT4:")
    sensitivity_experiment(num_bits=4)

    print("\n\nSTEP 6: GPTQ vs AWQ vs Naive (INT4)")
    print("-" * 50)
    full_quantization_comparison(d_in=256, d_out=512, num_bits=4)

    print("\n\nSTEP 7: Distribution Analysis")
    print("-" * 50)
    np.random.seed(0)
    simulated_weights = np.random.randn(1000) * 0.02
    abs_vals = np.abs(simulated_weights)
    pct_in_range = np.mean(abs_vals < 0.1) * 100
    print(f"\n  Simulated weight distribution (1000 params, std=0.02):")
    print(f"  Weights in [-0.1, 0.1]: {pct_in_range:.1f}%")
    print(f"  Weights in [-0.05, 0.05]: {np.mean(abs_vals < 0.05) * 100:.1f}%")
    print(f"  Weights in [-0.01, 0.01]: {np.mean(abs_vals < 0.01) * 100:.1f}%")
    print(f"  Max absolute value: {np.max(abs_vals):.6f}")
    print(f"  Mean absolute value: {np.mean(abs_vals):.6f}")

    histogram = np.histogram(simulated_weights, bins=20)
    print(f"\n  Weight histogram:")
    max_count = max(histogram[0])
    for i in range(len(histogram[0])):
        bar_len = int(histogram[0][i] / max_count * 40)
        lo = histogram[1][i]
        hi = histogram[1][i + 1]
        print(f"  [{lo:>7.4f}, {hi:>7.4f}] {'#' * bar_len} ({histogram[0][i]})")

    print("\n\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
```

## Use It

### Kwantyzacja z AutoGPTQ

```python
# pip install auto-gptq transformers
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# from transformers import AutoTokenizer
#
# model_id = "meta-llama/Llama-3.1-8B"
# quantize_config = BaseQuantizeConfig(
#     bits=4,
#     group_size=128,
#     desc_act=False,
# )
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
#
# calibration = [tokenizer(t, return_tensors="pt") for t in calibration_texts[:128]]
# model.quantize(calibration)
# model.save_quantized("llama-8b-gptq-int4")
```

### Kwantyzacja z AutoAWQ

```python
# pip install autoawq
# from awq import AutoAWQForCausalLM
# from transformers import AutoTokenizer
#
# model_id = "meta-llama/Llama-3.1-8B"
# model = AutoAWQForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
#
# model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
# model.save_quantized("llama-8b-awq-int4")
```

### Konwersja do GGUF

```bash
# pip install llama-cpp-python
# python convert_hf_to_gguf.py meta-llama/Llama-3.1-8B --outtype q4_k_m --outfile llama-8b-q4km.gguf
# llama-server -m llama-8b-q4km.gguf -c 4096 -ngl 99
```

### Serwowanie z vLLM

```python
# pip install vllm
# vllm serve model-awq --quantization awq --dtype half --max-model-len 8192
```

vLLM natywnie obsługuje modele AWQ i GPTQ. Obsługuje dekwantyzację podczas mnożenia macierzy i używa stronicowanej uwagi dla pamięci KV. Dla FP8 na H100 dodaj `--dtype float8_e4m3fn`.

## Ship It

Ta lekcja produkuje `outputs/skill-quantization.md`, ramy decyzyjne do wyboru odpowiedniej strategii kwantyzacji. Biorąc pod uwagę rozmiar modelu, docelowy sprzęt i wymagania jakościowe, mówi ci, którego formatu, metody i kroków walidacji użyć. Zawiera obliczenia budżetu pamięci, rekomendacje precyzji na komponent i przepisy wdrożeniowe dla vLLM, llama.cpp i TensorRT-LLM.

## Exercises

1. Zaimplementuj kwantyzację grupową. Zamiast jednej skali na kanał, użyj jednej skali na grupę 128 wag w kanale. To jest to, czego GPTQ i AWQ faktycznie używają. Porównaj rozmiary grup 32, 64, 128 i 256 na tej samej macierzy wag. Mniejsze grupy dają lepszą jakość, ale większy narzut przechowywania współczynników skali.

2. Zbuduj kwantyzator mieszanej precyzji. Kwantyzuj pierwszą i ostatnią warstwę wielowarstwowej sieci w INT8, kwantyzując środkowe warstwy w INT4. Porównaj jakość wyjścia od końca do końca z jednolitym INT4 i jednolitym INT8. Zmierz oszczędności pamięci w porównaniu do wszystkiego-INT8.

3. Zaimplementuj estymator prostego przejścia (STE) dla treningu świadomego kwantyzacji. Wstaw fikcyjne operacje kwantyzacji/dekwantyzacji w przejściu w przód prostej dwuwarstwowej sieci trenowanej na zadaniu regresji. Porównaj końcową stratę między modelem trenowanym normalnie (potem PTQ do INT4) a modelem trenowanym z QAT od początku.

4. Zbuduj kwantyzator świadomy wartości odstających, zainspirowany LLM.int8(). Wykryj kanały, gdzie wielkość aktywacji przekracza 6x średnią. Utrzymaj te kanały w FP16 i skwantyzuj wszystko inne do INT8. Zmierz jakość od końca do końca na warstwie transformera z Kroku 5 z różnymi progami wartości odstających (3x, 6x, 10x).

5. Zaimplementuj pulpit jakości kwantyzacji. Mając macierz wag, oblicz i wyświetl: histogram dystrybucji wag, dystrybucję błędu kwantyzacji, współczynniki skali na kanał, najgorzej skwantowane kanały (najwyższy błąd rekonstrukcji) i podobieństwo cosinusowe między oryginalnymi a skwantowanymi wynikami dla 100 losowych wejść. Zidentyfikuj, które kanały powinny być utrzymane w wyższej precyzji.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|----------------|----------------------|
| FP16 | "Połówkowa precyzja" | 16-bitowa liczba zmiennoprzecinkowa z 5 bitami wykładnika i 10 bitami mantysy, maksymalna wartość 65 504, standardowy format inferencji |
| BF16 | "Brain float" | 16-bitowa liczba zmiennoprzecinkowa z 8 bitami wykładnika (taki sam zakres jak FP32) i 7 bitami mantysy, zaprojektowana przez Google do treningu |
| FP8 | "Ośmiobitowa liczba zmiennoprzecinkowa" | Dwa warianty: E4M3 (inferencja, więcej precyzji) i E5M2 (trening, większy zakres), natywny na H100 |
| INT8 | "Ośmiobitowa liczba całkowita" | 256 równomiernie rozmieszczonych wartości od -128 do 127, potrzebuje współczynnika skali do odwzorowania z liczb zmiennoprzecinkowych |
| INT4 | "Czterobitowa liczba całkowita" | 16 poziomów, wymaga wyrafinowanych metod (GPTQ, AWQ) do utrzymania jakości |
| Kwantyzacja na kanał | "Jedna skala na wiersz" | Używa osobnego współczynnika skali dla każdego kanału wyjściowego zamiast jednego dla całego tensora, dramatycznie zmniejsza błąd |
| GPTQ | "Metoda Hesjanu" | Kwantyzacja potreningowa używająca informacji drugiego rzędu do minimalizacji błędu wyjścia, jedna warstwa na raz |
| AWQ | "Świadoma aktywacji" | Skaluje istotne wagi (mnożone przez duże aktywacje) przed kwantyzacją, aby je chronić |
| GGUF | "Format llama.cpp" | Samodzielny plik modelu z warstwami mieszanej precyzji, zoptymalizowany do inferencji na CPU i Apple Silicon |
| PTQ | "Kwantyzacja po treningu" | Konwersja wag wytrenowanego modelu do niższej precyzji bez ponownego treningu, szybka, ale ograniczona przy ekstremalnej kompresji |
| QAT | "Kwantyzacja podczas treningu" | Wstawianie fikcyjnej kwantyzacji do przejścia w przód, aby model nauczył się tolerować zaokrąglenia, lepszy przy INT4/INT2 |
| Dane kalibracyjne | "Te 128 przykładów" | Mały zbiór danych przepuszczonych przez model do obliczenia statystyk aktywacji dla ustawienia współczynników skali |
| Współczynnik skali | "Mnożnik" | Konwertuje między zakresem zmiennoprzecinkowym a całkowitoliczbowym: `float_val = int_val * scale` |
| Delta perplexity | "O ile gorzej" | Różnica w perplexity między oryginalnym a skwantowanym modelem, < 0,5 jest doskonała, > 2,0 to problem |

## Further Reading

- [Frantar et al., 2022 -- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323) -- the paper that made INT4 quantization practical for LLMs using Hessian-guided weight rounding
- [Lin et al., 2023 -- "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978) -- protecting salient weights by scaling before quantization, matching or beating GPTQ
- [Dettmers et al., 2022 -- "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"](https://arxiv.org/abs/2208.07339) -- mixed-precision INT8 that keeps outlier features in FP16, enabling INT8 inference without quality loss
- [Xiao et al., 2023 -- "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"](https://arxiv.org/abs/2211.10438) -- migrating quantization difficulty from activations to weights for W8A8 deployment
- [Micikevicius et al., 2022 -- "FP8 Formats for Deep Learning"](https://arxiv.org/abs/2209.05433) -- the NVIDIA/ARM/Intel paper defining E4M3 and E5M2 formats now native on H100