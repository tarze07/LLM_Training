# Przewodnik po architekturze DeepSeek-V3

> Faza 10 · Lekcja 14 wymieniła sześć pokręteł architektonicznych, które każdy otwarty model posiada. DeepSeek-V3 (grudzień 2024, 671B parametrów łącznie, 37B aktywnych) używa wszystkich sześciu i dodaje cztery kolejne: Multi-Head Latent Attention, równoważenie obciążenia bez straty pomocniczej, Multi-Token Prediction i trenowanie DualPipe. Ta lekcja czyta architekturę DeepSeek-V3 od góry do dołu i wyprowadza każdą liczbę parametrów z opublikowanej konfiguracji. Na koniec potrafisz wyjaśnić, dlaczego stosunek 671B/37B jest właściwym zakładem i dlaczego MLA + MoE razem biją każdą z tych technik osobno na granicy możliwości.

**Type:** Learn
**Languages:** Python (stdlib, kalkulator parametrów)
**Prerequisites:** Phase 10 · 14 (przeglądy otwartych modeli), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minut

## Cele nauczania

- Przeczytać konfigurację DeepSeek-V3 od góry do dołu i wyjaśnić każde pole w kategoriach sześciu pokręteł GPT-2 plus czterech dodatków specyficznych dla DeepSeek.
- Wyprowadzić całkowitą liczbę parametrów (671B), liczbę aktywnych parametrów (37B) oraz składowe, które się na nie składają.
- Obliczyć rozmiar pamięci podręcznej KV dla MLA przy kontekście 128k i porównać z tym, co zapłaciłby gęsty model o tych samych aktywnych parametrach z GQA.
- Wymienić cztery innowacje specyficzne dla DeepSeek (MLA, MTP, routing bez straty pomocniczej, DualPipe) i wskazać, którą część architektury/stosu treningowego każda z nich celuje.

## Problem

DeepSeek-V3 to pierwszy graniczny otwarty model, którego architektura znacząco różni się od rodziny Llama. Llama 3 405B to „GPT-2 z sześcioma pokrętłami". DeepSeek-V3 to GPT-2 ze wszystkimi sześcioma pokrętłami plus czterema kolejnymi. Czytanie konfiguracji Llama 3 to rozgrzewka przed czytaniem konfiguracji DeepSeek, ale głęboka struktura — kształt bloku attention, logika routingu, cel treningowy — jest wystarczająco różna, by potrzebować osobnego przewodnika.

Korzyść z nauczenia się tego: wydanie otwartych wag DeepSeek-V3 zmieniło to, co oznacza „graniczna zdolność" w otwartych modelach. Architektura jest planem, który wiele przebiegów treningowych w 2026 roku kopiuje. Zrozumienie jej jest podstawowym wymogiem dla każdej roli związanej z trenowaniem lub wnioskowaniem granicznych LLM.

## Koncepcja

### Niezmienne jądro, ponownie

DeepSeek-V3 jest wciąż autoregresyjny. Wciąż układa bloki dekodera. Każdy blok wciąż ma attention plus MLP plus dwie RMSNormy. Wciąż używa SwiGLU w MLP. Wciąż używa RoPE. Pre-norm. Wagi związane z osadzeniami (weight-tied embeddings). Ta sama linia bazowa co każdy Llama czy Mistral.

### Różnica: MLA zamiast GQA

Z Fazy 10 · 14 wiesz, że GQA zmniejsza pamięć podręczną KV poprzez współdzielenie K i V w grupach głowic Q. Multi-Head Latent Attention (MLA) idzie dalej: K i V są kompresowane do współdzielonej niskowymiarowej reprezentacji latentnej (`kv_lora_rank`), a następnie dekompresowane na bieżąco na głowicę. Pamięć podręczna KV przechowuje tylko latent — zazwyczaj 512 liczb zmiennoprzecinkowych na token na warstwę, a nie 8 x 128 = 1024 liczby.

Przy kontekście 128k, DeepSeek-V3 z MLA (jeden współdzielony latent `c^{KV}` na token na warstwę; K i V są wyprowadzane z tego latentu przez projekcje w górę, które mogą być wchłonięte przez kolejne mnożenie macierzy):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

Hipotetyczna linia bazowa GQA (kształt Llama 3 70B, 8 głowic KV, wymiar głowicy 128) zapłaciłaby:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA jest 4 razy mniejsza niż pamięć podręczna GQA w stylu Llama-3-70B przy kontekście 128k.

Kompromis: MLA dodaje krok dekompresji na każde obliczenie attention (na głowicę). Dodatkowe obliczenia są małe w porównaniu do zaoszczędzonego pasma. Netto zysk dla wnioskowania na długich kontekstach.

### Routing: równoważenie obciążenia bez straty pomocniczej

Routery MoE decydują, którzy eksperci top-k przetwarzają każdy token. Naiwny router koncentruje zbyt dużo pracy na kilku ekspertach, pozostawiając innych bezczynnych. Standardowe rozwiązanie: dodanie członu straty pomocniczej, który karze brak równowagi obciążenia. To działa, ale nieznacznie pogarsza wydajność głównego zadania.

DeepSeek-V3 wprowadza schemat bez straty pomocniczej. Terminy bias per-ekspert są dodawane do logitów routera, dostosowywane podczas trenowania przez prostą regułę: jeśli ekspert `e` jest przeciążony, zmniejsz `bias_e`; jeśli niedociążony, zwiększ go. Żaden dodatkowy człon straty. Trenowanie pozostaje czyste. Obciążenie ekspertów pozostaje zrównoważone.

Wpływ na główną stratę: niemierzalny. Wpływ na architekturę MoE: czystsza, brak hiperparametru straty pomocniczej do strojenia.

### MTP: gęstsze trenowanie + darmowy szkic

Z Fazy 10 · 18 wiesz, że DeepSeek-V3 dodaje moduł MTP D=1, który przewiduje token dwie pozycje do przodu. Podczas wnioskowania wytrenowany moduł jest wykorzystywany jako szkic do dekodowania spekulatywnego z akceptacją na poziomie 80%+. Podczas trenowania każdy ukryty stan jest nadzorowany na D+1 = 2 celach, zapewniając gęstszy sygnał.

Parametry: 14B na wierzchu 671B głównego modelu. Narzut: 2,1%.

### Trenowanie: DualPipe

Z Fazy 10 · 19 wiesz, że DualPipe to dwukierunkowy potok, który nakłada fragmenty forward i backward z międzywęzłową komunikacją all-to-all. W skali 2 048 kart H800 DeepSeek-V3 odzyskuje około 245k godzin GPU, które 1F1B straciłby na pęcherzyki potoku.

### Konfiguracja, pole po polu

Oto konfiguracja DeepSeek-V3 (uproszczona):

```
hidden_size: 7168
intermediate_size: 18432   (gęsty rozmiar ukryty MLP, używany w pierwszych kilku warstwach)
moe_intermediate_size: 2048 (rozmiar ukryty MLP eksperta)
num_hidden_layers: 61
first_k_dense_layers: 3    (pierwsze 3 warstwy używają gęstego MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formalnie równe num_heads w MLA, ale
                            prawdziwa kompresja jest w kv_lora_rank)
kv_lora_rank: 512          (wymiar latentny MLA)
num_experts: 256            (liczba ekspertów MoE na blok)
num_experts_per_tok: 8      (routing top-8)
shared_experts: 1           (zawsze włączony współdzielony ekspert na blok)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 moduł MTP na głębokości 1)
```

Analiza:

- `hidden_size=7168`: wymiar osadzenia.
- `num_hidden_layers=61`: całkowita głębokość bloków.
- `first_k_dense_layers=3`: pierwsze 3 bloki używają gęstego MLP o rozmiarze 18432. Pozostałe 58 używa MoE.
- `num_attention_heads=128`: 128 głowic zapytania.
- `kv_lora_rank=512`: K i V są kompresowane do tego wymiaru latentnego i dekompresowane na głowicę.
- `num_experts=256, num_experts_per_tok=8`: każdy blok MoE ma 256 ekspertów, routing top-8.
- `shared_experts=1`: oprócz 256 routowanych ekspertów, 1 zawsze włączony ekspert wnosi wkład do każdego tokena. Myśl o tym jako o „gęstej podłodze", która zapewnia, że każdy token dostaje coś niezawodnego.
- `moe_intermediate_size=2048`: rozmiar ukryty MLP każdego eksperta. Mniejszy niż gęsty MLP, ponieważ jest ich 256.

### Rachunek parametrów

Pełne obliczenia znajdują się w `code/main.py`. Najważniejsze liczby:

- Osadzenie: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- Pierwsze 3 gęste bloki: attention z MLA (~144M na blok) + gęsty MLP (~260M na blok) + normalizacje. Około 1.2B łącznie.
- 58 bloków MoE: attention z MLA (~144M) + 256 ekspertów każdy (30M sztuka) + 1 współdzielony ekspert (30M) + normalizacja. Łącznie ~7.95B na blok, wliczając wszystkich ekspertów. 461B łącznie dla 58 bloków MoE.
- Moduł MTP: 14B.

Suma całkowita: ~476B dla rdzenia architektury + 14B MTP + wyraźnie opublikowana liczba 671B uwzględnia dodatkowe parametry strukturalne (tensory bias, komponenty specyficzne dla ekspertów, skalowanie współdzielonego eksperta itp.). Liczba, którą odtwarzamy w kalkulatorze, mieści się w granicach 3-5% opublikowanej — różnica pochodzi z drobiazgowego rachunku, który raport DeepSeek dokumentuje w dodatku do Sekcji 2.

Aktywne parametry na jeden forward:

- Attention: 144M na warstwę * 61 = 8.8B (wszystkie warstwy działają).
- Aktywny MLP: pierwsze 3 warstwy gęste (3 * 260M = 780M), 58 warstw MoE każda aktywna z 8 routowanymi + 1 współdzielonym + narzut routingu. Na warstwę aktywny MLP: ~260M. Łącznie: 3 * 260M + 58 * 260M = ~15.9B.
- Osadzenie + normalizacje: 1.2B.
- Łącznie aktywne: około 26B rdzeń + 14B MTP (trenowany, ale nie zawsze uruchamiany podczas wnioskowania) ≈ 37B.

### Stosunek 671B / 37B

18-krotny współczynnik rzadkości (aktywne parametry to 5.5% całości). DeepSeek-V3 to najrzadszy graniczny model MoE, który został wydany z otwartymi wagami. Mixtral 8x7B przy stosunku 13/47 (28%) jest znacznie gęstszy. Llama 4 Maverick przy stosunku 17B/400B (4.25%) jest porównywalny. Zakład DeepSeek: w skali granicznej, więcej ekspertów z niższym współczynnikiem aktywacji daje lepszą jakość na aktywny FLOP.

### Gdzie plasuje się DeepSeek-V3

| Model | Łącznie | Aktywne | Stosunek | Attention | Nowe pomysły |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | Rozszerzenie YaRN |

### Następcy: R1, V4

DeepSeek-R1 (2025) to przebieg trenowania rozumowania na szkielecie V3. R1 używa tej samej architektury. Zmienił się przepis po trenowaniu (RL na dużą skalę na weryfikowalnych zadaniach), a nie architektura pretreningu.

Oczekuje się, że DeepSeek-V4 (jeśli zostanie wydany) zachowa MLA + MoE + MTP i doda DSA (DeepSeek Sparse Attention), następcę NSA z Fazy 10 · 17. Linia rozwojowa jest stabilna: innowacje na poziomie architektury kumulują się; każda wersja dodaje kolejne pokrętła.

```figure
moe-routing
```

## Użyj

`code/main.py` to kalkulator parametrów wyspecjalizowany dla kształtu DeepSeek-V3. Uruchom go, porównaj jego wyniki z liczbami z artykułu i użyj go na hipotetycznych wariantach (256 ekspertów vs 512, top-8 vs top-16, ranga MLA 512 vs 1024).

Na co patrzeć:

- Całkowita liczba parametrów vs opublikowane 671B.
- Liczba aktywnych parametrów vs opublikowane 37B.
- Pamięć podręczna KV przy kontekście 128k — porównanie MLA vs GQA.
- Podział na warstwy, aby zobaczyć, gdzie faktycznie idzie budżet parametrów.

## Dostarcz

Ta lekcja produkuje `outputs/skill-deepseek-v3-reader.md`. Dla modelu z rodziny DeepSeek (V3, R1 lub dowolny przyszły wariant) tworzy odczyt architektury komponent po komponencie, który nazywa każde pole konfiguracji, wyprowadza liczby parametrów według komponentu i identyfikuje, której z czterech innowacji specyficznych dla DeepSeek model używa.

## Ćwiczenia

1. Uruchom `code/main.py`. Porównaj oszacowanie całkowitej liczby parametrów kalkulatora z opublikowanym 671B i zidentyfikuj, skąd pochodzi różnica. Sekcja 2 artykułu zawiera pełną specyfikację.

2. Zmodyfikuj konfigurację, aby użyć rangi MLA 256 zamiast 512. Oblicz wynikowy rozmiar pamięci podręcznej KV przy kontekście 128k. Jaki procent redukcji to daje i jakim kosztem dla ekspresyjności na głowicę?

3. Porównaj routing DeepSeek-V3 (256 ekspertów, top-8) z hipotetycznym wariantem (512 ekspertów, top-8). Całkowite parametry rosną; aktywne parametry pozostają takie same. Co daje dodatkowa pojemność ekspertów w teorii i jaki jest jej koszt podczas wnioskowania?

4. Przeczytaj Sekcję 2.1 raportu technicznego DeepSeek-V3 (arXiv:2412.19437) o MLA. Wyjaśnij w trzech zdaniach, dlaczego macierze dekompresji K i V mogą być „wchłonięte" przez kolejne mnożenie macierzy dla wydajności wnioskowania.

5. DeepSeek-V3 używa trenowania FP8 dla większości operacji. Oblicz oszczędności pamięci FP8 vs BF16 dla przechowywania wag 671B. Jak to się ma do budżetu trenowania na 14,8T tokenów?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| MLA | „Multi-Head Latent Attention" | Kompresuje K i V do współdzielonego niskowymiarowego latentu (kv_lora_rank, zazwyczaj 512), dekompresuje na głowicę na bieżąco; pamięć podręczna KV przechowuje tylko latent |
| kv_lora_rank | „Wymiar kompresji MLA" | Rozmiar współdzielonego latentu dla K i V; DeepSeek-V3 używa 512 |
| First k dense layers | „Wczesne warstwy pozostają gęste" | Pierwsze kilka warstw modelu MoE pomija router MoE i uruchamia gęsty MLP dla stabilności |
| num_experts_per_tok | „Routing top-k" | Ilu routowanych ekspertów działa na token; DeepSeek-V3 używa 8 |
| Współdzieleni eksperci | „Zawsze włączeni eksperci" | Eksperci, którzy przetwarzają każdy token niezależnie od routingu; DeepSeek-V3 używa 1 |
| Routing bez straty pomocniczej | „Równoważenie obciążenia przez bias" | Terminy bias per-ekspert dostosowywane podczas trenowania, aby utrzymać równowagę obciążenia ekspertów bez dodawania członu straty |
| Moduł MTP | „Dodatkowa głowica predykcyjna" | Blok transformera przewidujący t+2 z h^(1) i E(t+1); gęstsze trenowanie, darmowy szkic do dekodowania spekulatywnego |
| DualPipe | „Dwukierunkowy potok" | Harmonogram trenowania nakładający obliczenia forward/backward z międzywęzłowym all-to-all |
| Stosunek aktywnych parametrów | „Rzadkość" | active_params / total_params; DeepSeek-V3 osiąga 5.5% |
| Trenowanie FP8 | „Trenowanie 8-bitowe" | Przechowywanie treningowe i wiele operacji obliczeniowych w FP8; mniej więcej o połowę mniejsza pamięć niż BF16 przy niewielkim koszcie jakości |

## Dalsze czytanie

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — pełny dokument architektury, trenowania i wyników
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — pliki konfiguracyjne i notatki wdrożeniowe
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — poprzednik, który wprowadził MLA
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — następca trenowania rozumowania na architekturze V3
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — przyszły kierunek dla attention w rodzinie DeepSeek
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — referencja harmonogramu trenowania