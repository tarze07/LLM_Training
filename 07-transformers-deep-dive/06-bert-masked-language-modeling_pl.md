# BERT — Maskowane Modelowanie Języka (Masked Language Modeling)

> GPT przewiduje następne słowo. BERT przewiduje brakujące słowo. Jedno zdanie różnicy — i pół dekady wszystkiego, co ma kształt osadzenia (embedding).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## Problem

W 2018 roku każde zadanie NLP — analiza sentymentu, NER, QA, implikacja — trenowało własny model od zera na własnych oznaczonych danych. Nie było wstępnie wytrenowanego punktu kontrolnego "rozumiem angielski", który można by dostroić. ELMo (2018) pokazało, że można wstępnie trenować osadzenia kontekstowe za pomocą dwukierunkowego LSTM; pomogło, ale nie uogólniło.

BERT (Devlin i in. 2018) zapytał: co jeśli weźmiemy enkoder transformera, wytrenujemy go na każdym zdaniu w internecie i zmusimy do przewidywania brakujących słów na podstawie kontekstu z obu stron? Następnie dostrajasz jedną głowę do swojego zadania. Efektywność parametryczna była rewelacją.

Wynik: w ciągu 18 miesięcy BERT i jego warianty (RoBERTa, ALBERT, ELECTRA) zdominowały każdą istniejącą tablicę wyników NLP. Do 2020 roku każda wyszukiwarka, potok moderacji treści i system wyszukiwania semantycznego na Ziemi miały w środku BERTa.

W 2026 roku modele tylko z enkoderem są wciąż właściwym narzędziem do klasyfikacji, wyszukiwania i strukturalnej ekstrakcji — działają 5–10 razy szybciej na token niż dekodery, a ich osadzenia są kręgosłupem każdego nowoczesnego systemu wyszukiwania. ModernBERT (grudzień 2024) rozszerzył architekturę do kontekstu 8K z Flash Attention + RoPE + GeGLU.

## Koncepcja

![Maskowane modelowanie języka: wybierz tokeny, zamaskuj je, przewidź oryginały](../assets/bert-mlm.svg)

### Sygnał treningowy

Weź zdanie: `the quick brown fox jumps over the lazy dog`.

Zamaskuj losowo 15% tokenów:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Trenuj model, aby przewidywał oryginalne tokeny w zamaskowanych pozycjach. Ponieważ enkoder jest dwukierunkowy, przewidywanie `[MASK]` na pozycji 1 może wykorzystać `brown fox jumps` na pozycjach 2+. To jest rzecz, której GPT nie może zrobić.

### Zasady maskowania BERT

Z 15% tokenów wybranych do przewidywania:

- 80% jest zastępowanych `[MASK]`.
- 10% jest zastępowanych losowym tokenem.
- 10% pozostaje bez zmian.

Dlaczego nie zawsze `[MASK]`? Ponieważ `[MASK]` nigdy nie występuje podczas wnioskowania. Trenowanie modelu, aby oczekiwał `[MASK]` w 100% maskowanych pozycji, stworzyłoby przesunięcie dystrybucji między pretrenowaniem a dostrajaniem. 10% losowych + 10% bez zmian utrzymuje model w ryzach.

### Przewidywanie Następnego Zdania (NSP) — i dlaczego zostało porzucone

Oryginalny BERT był również trenowany na NSP: mając dwa zdania A i B, przewidź, czy B następuje po A. RoBERTa (2019) przeprowadziła ablację i pokazała, że NSP szkodzi, a nie pomaga. Nowoczesne enkodery pomijają to.

### Co zmieniło się w 2026: ModernBERT

Artykuł ModernBERT z 2024 roku przebudował blok z prymitywami z 2026 roku:

| Komponent | Oryginalny BERT (2018) | ModernBERT (2024) |
|-----------|------------------------|-------------------|
| Pozycyjny | Uczone absolutne | RoPE |
| Aktywacja | GELU | GeGLU |
| Normalizacja | LayerNorm | Pre-norm RMSNorm |
| Attention | W pełni gęste | Naprzemienne lokalne (128) + globalne |
| Długość kontekstu | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

I w przeciwieństwie do stosu z 2018 roku, jest natywny dla Flash Attention. Wnioskowanie jest 2–3 razy szybsze przy długości sekwencji 8K niż DeBERTa-v3 z lepszymi wynikami GLUE.

### Przypadki użycia, które w 2026 roku wciąż wybierają enkoder

| Zadanie | Dlaczego enkoder bije dekoder |
|---------|--------------------------------|
| Wyszukiwanie / osadzenia do wyszukiwania semantycznego | Kontekst dwukierunkowy = lepsza jakość osadzenia na token |
| Klasyfikacja (sentyment, intencja, toksyczność) | Jedno przejście w przód; żadnego narzutu generowania |
| NER / etykietowanie tokenów | Wyjście na pozycję, naturalnie dwukierunkowe |
| Implikacja zero-shot (NLI) | Głowa klasyfikatora na górze enkodera |
| Reranker dla RAG | Ocenianie przez cross-encoder, 10× szybciej niż rerankery LLM |

```figure
transformer-residual
```

## Zbuduj to

### Krok 1: logika maskowania

Zobacz `code/main.py`. Funkcja `create_mlm_batch` przyjmuje listę ID tokenów, rozmiar słownika i prawdopodobieństwo maskowania. Zwraca ID wejściowe (z zastosowanymi maskami) i etykiety (tylko w maskowanych pozycjach, -100 gdzie indziej — konwencja indeksu ignorowania PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Krok 2: wykonaj predykcję MLM na małym korpusie

Wytrenuj 2-warstwowy enkoder + głowę MLM na słowniku 20 słów, 200 zdań. Bez gradientu — robimy sprawdzenie poprawności przejścia w przód. Pełne trenowanie wymaga PyTorch.

### Krok 3: porównaj typy maskowania

Pokaż, jak reguła trzech sposobów utrzymuje model użytecznym bez `[MASK]`. Przewiduj na zdaniu bez maski i na zdaniu z maską. Oba powinny dawać rozsądne rozkłady tokenów, ponieważ model widział oba wzorce podczas trenowania.

### Krok 4: dostrój głowę

Zastąp głowę MLM głową klasyfikacyjną na zabawkowym zbiorze danych z sentymentem. Trenuje się tylko głowa; enkoder jest zamrożony. To jest wzór, za którym podąża każda aplikacja BERTa.

## Użyj tego

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Modele osadzeń to dostrojone BERTy.** Modele `sentence-transformers`, takie jak `all-MiniLM-L6-v2`, to BERTy trenowane ze stratą kontrastywną. Enkoder jest ten sam. Zmieniła się funkcja straty.

**Cross-encoder rerankery to również dostrojone BERTy.** Klasyfikacja par na `[CLS] query [SEP] doc [SEP]`. Dwukierunkowa attention między zapytaniem a dokumentem to właśnie to, co daje cross-encoderom przewagę jakościową nad biencoderami.

**Kiedy nie wybierać BERTa w 2026 roku.** Wszystko, co generatywne. Enkoder nie ma rozsądnego sposobu na autoregresyjne wytwarzanie tokenów. Również: wszystko poniżej 1B parametrów, gdzie mały dekoder może dorównać jakości przy większej elastyczności (Phi-3-Mini, Qwen2-1.5B).

## Dostarcz to

Zobacz `outputs/skill-bert-finetuner.md`. Umiejętność (skill) określa zakres dostrajania BERTa (wybór backbonu, specyfikacja głowy, dane, ewaluacja, zatrzymanie) dla nowego zadania klasyfikacji lub ekstrakcji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` i wypisz rozkład maskowania na 10 000 tokenów. Potwierdź, że ~15% jest wybranych, a z tych ~80% staje się `[MASK]`.
2. **Średnie.** Zaimplementuj maskowanie całych słów: jeśli słowo jest tokenizowane na podtokeny, zamaskuj wszystkie podtokeny razem lub żaden. Zmierz, czy poprawia to dokładność MLM na korpusie 500 zdań.
3. **Trudne.** Wytrenuj mały (2-warstwowy, d=64) BERT na 10 000 zdań z publicznego zbioru danych. Dostrój token `[CLS]` dla analizy sentymentu SST-2. Porównaj z bazowym modelem tylko z dekoderem przy dopasowanej liczbie parametrów — który wygrywa?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| MLM | "Maskowane modelowanie języka" | Sygnał treningowy: losowo zastąp 15% tokenów `[MASK]`, przewidź oryginały. |
| Dwukierunkowość | "Patrzy w obie strony" | Attention enkodera nie ma maski przyczynowej — każda pozycja widzi każdą inną pozycję. |
| `[CLS]` | "Token łącznikowy (pooler)" | Specjalny token dodawany na początku każdej sekwencji; jego końcowe osadzenie jest używane jako reprezentacja na poziomie zdania. |
| `[SEP]` | "Separator segmentów" | Oddziela sparowane sekwencje (np. zapytanie/dokument, zdanie A/B). |
| NSP | "Przewidywanie następnego zdania" | Drugie zadanie pretrenowania BERTa; okazało się bezużyteczne w RoBERTa, porzucone po 2019. |
| Dostrajanie (fine-tuning) | "Dostosuj do zadania" | Zachowaj enkoder w większości zamrożony; trenuj małą głowę na wierzchu dla zadania docelowego. |
| Cross-encoder | "Reranker" | BERT, który przyjmuje zarówno zapytanie, jak i dokument jako wejście i zwraca wynik trafności. |
| ModernBERT | "Odświeżenie z 2024" | Enkoder przebudowany z RoPE, RMSNorm, GeGLU, naprzemienną lokalną/globalną attention i kontekstem 8K. |

## Dalsze czytanie

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — oryginalny artykuł.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — jak prawidłowo trenować BERTa; zabija NSP.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — detekcja zastąpionych tokenów bije MLM przy dopasowanym nakładzie obliczeniowym.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — artykuł ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — kanoniczna referencja enkodera.