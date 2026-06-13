# Tokenizacja Podsłowowa — BPE, WordPiece, Unigram, SentencePiece

> Tokenizatory słów duszą się na nieznanych słowach. Tokenizatory znaków rozdymają długość sekwencji. Tokenizatory podsłowowe znajdują złoty środek. Każdy nowoczesny LLM działa na jednym z nich.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## Problem

Twój słownik zawiera 50 000 słów. Użytkownik wpisuje "nieznane_słowo". Twój tokenizator zwraca `[UNK]`. Model nie ma żadnego sygnału o tym słowie. Gorzej: 90. percentyl dokumentu w twoim korpusie zawiera 40 rzadkich słów, co oznacza 40 bitów utraconej informacji na dokument.

Tokenizacja podsłowowa rozwiązuje ten problem. Popularne słowa pozostają pojedynczymi tokenami. Rzadkie słowa rozkładają się na znaczące części: `nieznane_słowo` → `nie`, `znane`, `słowo`. Dane treningowe pokrywają wszystko, ponieważ każdy ciąg znaków to ostatecznie sekwencja bajtów.

Każdy czołowy LLM w 2026 roku działa na jednym z trzech algorytmów (BPE, Unigram, WordPiece), opakowanych w jedną z trzech bibliotek (tiktoken, SentencePiece, HF Tokenizers). Nie możesz wdrożyć modelu językowego bez wyboru jednego z nich.

## Koncepcja

![BPE vs Unigram vs WordPiece, krok po kroku](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Zacznij od słownika na poziomie znaków. Policz każdą sąsiednią parę. Połącz najczęstszą parę w nowy token. Powtarzaj, aż osiągniesz docelowy rozmiar słownika. Dominujący algorytm: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**BPE na poziomie bajtów.** Ten sam algorytm, ale operujący na surowych bajtach (256 bazowych tokenów) zamiast znaków Unicode. Gwarantuje zerową liczbę tokenów `[UNK]` — każda sekwencja bajtów jest kodowalna. GPT-2 używa 50 257 tokenów (256 bajtów + 50 000 merges + 1 specjalny).

**Unigram.** Zacznij od ogromnego słownika. Przypisz każdemu tokenowi prawdopodobieństwo unigramowe. Iteracyjnie usuwaj tokeny, których usunięcie najmniej zwiększa logarytm wiarygodności korpusu. Probabilistyczny w inferencji: może próbkować tokenizacje (przydatne do augmentacji danych poprzez regularyzację podsłowową). Używany przez T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Łączy pary, które maksymalizują wiarygodność korpusu treningowego, a nie surową częstotliwość. Używany przez BERT, DistilBERT, ELECTRA.

**SentencePiece vs tiktoken.** SentencePiece to biblioteka, która *trenuje* słowniki (BPE lub Unigram) bezpośrednio na surowym tekście Unicode, kodując białe znaki jako `▁`. tiktoken to szybki *enkoder* OpenAI działający na predefiniowanych słownikach; nie trenuje.

Zasada kciuka:

- **Trenowanie nowego słownika:** SentencePiece (wielojęzyczny, bez wstępnej tokenizacji) lub HF Tokenizers.
- **Szybka inferencja na słowniku GPT:** tiktoken (cl100k_base, o200k_base).
- **Oba:** HF Tokenizers — jedna biblioteka, trenowanie i serwowanie.

```figure
bpe-merge
```

## Zbuduj To

### Krok 1: BPE od zera

Zobacz `code/main.py`. Pętla:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Trzy fakty, które koduje ten algorytm. `</w>` oznacza koniec słowa, dzięki czemu "low" (przyrostek) i "lower" (prefiks) pozostają odrębne. Ważenie częstotliwością sprawia, że pary o wysokiej częstotliwości wygrywają wcześnie. Lista merges jest uporządkowana — inferencja stosuje merges w kolejności treningowej.

### Krok 2: kodowanie z nauczonymi merges

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Naiwna O(n·|merges|). Implementacje produkcyjne (tiktoken, HF Tokenizers) używają wyszukiwania rangi merges z kolejkami priorytetowymi i działają w czasie prawie liniowym.

### Krok 3: SentencePiece w praktyce

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Zauważ: brak wymaganej wstępnej tokenizacji, spacja zakodowana jako `▁`, `character_coverage` kontroluje, jak agresywnie rzadkie znaki są zachowywane vs mapowane na `<unk>`.

### Krok 4: tiktoken dla słowników zgodnych z OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Tylko kodowanie. Szybkie (backend w Ruście). Dokładne dopasowanie z tokenizacją GPT-4/5 do liczenia bajtów, szacowania kosztów i budżetowania okna kontekstowego.

## Pułapki, które wciąż występują w 2026

- **Dryf tokenizatora.** Trenowanie na słowniku A, wdrożenie na słowniku B. ID tokenów się różnią; model generuje śmieci. Sprawdzaj hash `tokenizer.json` w CI.
- **Niejednoznaczność białych znaków.** BPE "hello" vs " hello" daje różne tokeny. Zawsze określaj jawnie `add_special_tokens` i `add_prefix_space`.
- **Niedotrenowanie wielojęzyczne.** Korpusy zdominowane przez angielski produkują słowniki, które dzielą skrypty niełacińskie na 5-10x więcej tokenów. Ten sam prompt kosztuje 5-10x więcej w języku japońskim/arabskim na GPT-3.5. o200k_base częściowo naprawił ten problem.
- **Podział emoji.** Pojedyncze emoji może zająć 5 tokenów. Uwzględniaj obsługę emoji przy budżetowaniu kontekstu.

## Użyj Tego

Stos w 2026:

| Sytuacja | Wybór |
|-----------|------|
| Trenowanie modelu jednojęzycznego od zera | HF Tokenizers (BPE) |
| Trenowanie modelu wielojęzycznego | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serwowanie API zgodnego z OpenAI | tiktoken (`o200k_base` dla GPT-4+) |
| Słownik domenowy (kod, matematyka, białka) | Trenuj niestandardowe BPE na korpusie domenowym, scal z bazowym słownikiem |
| Inferencja na brzegu, mały model | Unigram (mniejsze słowniki działają lepiej) |

Rozmiar słownika to decyzja skalowania, nie stała. Zgrubna heurystyka: 32k dla <1B parametrów, 50-100k dla 1-10B, 200k+ dla wielojęzycznych/czołowych.

## Wdróż To

Zapisz jako `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Ćwiczenia

1. **Łatwe.** Wytrenuj BPE z 500 merges na małym korpusie z `code/main.py`. Zakoduj trzy wstrzymane słowa. Ile dało dokładnie 1 token vs >1 token?
2. **Średnie.** Porównaj liczbę tokenów na 100 zdaniach z angielskiej Wikipedii między `cl100k_base`, `o200k_base`, a SentencePiece BPE, który wytrenujesz z vocab=32k. Raportuj współczynnik kompresji każdego.
3. **Trudne.** Wytrenuj ten sam korpus z BPE, Unigram i WordPiece. Zmierz dokładność downstream przy użyciu każdego z nich na małym klasyfikatorze sentymentu. Czy wybór zmienia wynik o więcej niż 1 punkt F1?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Zachłanne łączenie najczęstszych par znaków aż do osiągnięcia docelowego rozmiaru słownika. |
| Byte-level BPE | Żadnych nieznanych tokenów | BPE na surowych 256 bajtach; GPT-2 / Llama tego używają. |
| Unigram | Probabilistyczny tokenizator | Przycina z dużego zbioru kandydatów używając log-wiarygodności; używany przez T5, Gemma. |
| SentencePiece | Ten od białych znaków | Biblioteka trenująca BPE/Unigram na surowym tekście; spacja zakodowana jako `▁`. |
| tiktoken | Ten szybki | Enkoder BPE OpenAI z backendem w Ruście dla predefiniowanych słowników. Bez trenowania. |
| Lista merges | Magiczne liczby | Uporządkowana lista merges `(a, b) → ab`; inferencja stosuje w kolejności. |
| Character coverage | Jak rzadki jest zbyt rzadki? | Frakcja znaków w korpusie treningowym, którą tokenizator musi pokryć; ~0.9995 typowo. |

## Dalsza Lektura

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — artykuł o BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — artykuł o Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — biblioteka.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — zwięzłe źródło.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + lista kodowań.