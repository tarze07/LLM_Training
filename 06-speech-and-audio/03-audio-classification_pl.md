# Klasyfikacja Audio — Od k-NN na MFCC do AST i BEATs

> Wszystko od "szczekanie psa vs syrena" do "jaki to język" to klasyfikacja audio. Cechy to mels. Architektura zmienia się co dekadę. Ewaluacja pozostaje AUC, F1 i recall na klasę.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## Problem

Dostajesz 10-sekundowy klip. Chcesz wiedzieć: "co to jest?" Dźwięk miejski (syrena, wiertarka, pies), komenda głosowa (tak/nie/stop), identyfikacja języka (en/es/ar), emocja mówcy (zły/neutralny), czy dźwięk otoczenia (wewnątrz/na zewnątrz, gwar). Wszystkie to *klasyfikacja audio*, a w 2026 podstawowa architektura jest dojrzała: log-mel → CNN lub Transformer → softmax.

Główna trudność to nie sieć. To dane. Zbiory danych audio mają brutalny brak równowagi klas, silne przesunięcie domeny (czyste vs hałaśliwe) i szum etykiet (kto zdecydował "miejski gwar" vs "hałas restauracji"?). 80% problemu to kuratorowanie, augmentacja i ewaluacja, nie zamiana CNN na Transformer.

## Koncepcja

![Drabina klasyfikacji audio: k-NN na MFCC do AST do BEATs](../assets/audio-classification.svg)

**k-NN na MFCC (baseline z lat 90.).** Spłaszcz MFCC na klip, oblicz podobieństwo cosinusowe z oznakowaną bazą, zwróć większościowy głos z top K. Zaskakująco silny na czystych, małych zbiorach (Speech Commands, ESC-50). Działa bez GPU.

**2D CNN na log-melach (2015-2019).** Traktuj log-mel `(T, n_mels)` jako obraz. Zastosuj ResNet-18 lub VGG-style. Globalna średnia pula w osi czasu. Softmax na klasach. Wciąż baseline w większości konkursów Kaggle w 2026.

**Audio Spectrogram Transformer, AST (2021-2024).** Podziel log-mel na łatki (np. 16×16), dodaj osadzenia pozycji, podaj do ViT. Stan sztuki na AudioSet (mAP 0.485) dla uczenia nadzorowanego.

**BEATs i WavLM-base (2024-2026).** Samonadzorowane pretrenowanie na milionach godzin. Dostrój do swojego zadania z 1-10% danych nadzorowanych, których byś potrzebował. W 2026 to domyślny punkt startowy dla dźwięków innych niż mowa. BEATs-iter3 bije AST o 1-2 mAP na AudioSet przy 1/4 mocy obliczeniowej.

**Encoder Whispera jako zamrożony backbone (2024).** Weź encoder Whispera, odrzuć dekoder, dołącz liniowy klasyfikator. Blisko SOTA w identyfikacji języka i prostej klasyfikacji zdarzeń bez augmentacji audio. Darmowy baseline "lunch".

### Brak równowagi klas to prawdziwe wyzwanie

ESC-50: 50 klas, 40 klipów każda — zbalansowane, łatwe. UrbanSound8K: 10 klas, niezbalansowane 10:1. AudioSet: 632 klas z długim ogonem 100 000:1. Techniki, które działają:

- Zbalansowane próbkowanie podczas treningu (nie w ewaluacji).
- Mixup: liniowa interpolacja dwóch klipów (i ich etykiet) jako augmentacja.
- SpecAugment: maskuj losowe pasma czasu i częstotliwości. Proste; krytyczne.

### Ewaluacja

- Multiclass wyłączny (Speech Commands): dokładność top-1, top-5.
- Multiclass multi-label (AudioSet, UrbanSound-style): średnia precyzja (mAP).
- Mocno niezbalansowane: recall na klasę + macro F1.

Liczby 2026, które powinieneś znać:

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

## Zbuduj To

### Krok 1: featuryzacja

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Krok 2: podsumowanie stałej długości

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Proste, ale mocne: średnia + wariancja w czasie daje 26-wymiarowe stałe osadzenie dla 13-współczynnikowego MFCC. Działa natychmiast. Biło SOTA bazujące na NN na ESC-50 jeszcze w 2017.

### Krok 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Krok 4: przejście na CNN na log-melach

W PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M parametrów. Trenuje w ~10 min na ESC-50 z pojedynczym RTX 4090. 80%+ dokładności.

### Krok 5: domyślne 2026 — dostrój BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Dla BEATs, użyj `microsoft/BEATs-base` przez bibliotekę `beats`; API transformers ma ten sam kształt.

## Użyj Tego

Stos 2026:

| Situation | Start with |
|-----------|-----------|
| Mały zbiór (<1000 klipów) | k-NN na średnich MFCC (twój baseline) + augmentacja audio |
| Średni zbiór (1K–100K) | BEATs lub AST fine-tune |
| Duży zbiór (>100K) | Trenuj od zera lub dostrój encoder Whispera |
| Rzeczywisty czas, edge | 40-MFCC CNN, skwantowany do int8 (styl KWS) |
| Multi-label (AudioSet) | BEATs-iter3 z BCE loss + mixup + SpecAugment |
| Identyfikacja języka | MMS-LID, SpeechBrain VoxLingua107 baseline |

Reguła decyzyjna: **zacznij od zamrożonego backbonu, nie świeżego modelu**. Dostrojenie głowy BEATs daje 95% SOTA w godzinach, nie tygodniach.

## Wyślij To

Zapisz jako `outputs/skill-classifier-designer.md`. Wybierz architekturę, augmentacje, strategię równoważenia klas i metrykę ewaluacji dla danego zadania klasyfikacji audio.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Trenuje baseline k-NN MFCC na 4-klasowym syntetycznym zbiorze (czyste tony o różnych wysokościach). Raportuj macierz pomyłek.
2. **Średnie.** Zastąp `summarize` przez [mean, var, skew, kurtosis]. Czy 4-momentowe pooling bije mean+var na tym samym syntetycznym zbiorze?
3. **Trudne.** Używając `torchaudio`, wytrenuj 2D CNN na ESC-50 fold 1. Raportuj dokładność 5-fold cross-validation. Dodaj SpecAugment (time mask = 20, freq mask = 10) i raportuj różnicę.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| AudioSet | ImageNet audio | Zbiór YouTube Google'a: 2M klipów, 632 klasy, słabo oznaczone. |
| ESC-50 | Mały benchmark klasyfikacji | 50 klas × 40 klipów dźwięków otoczenia. |
| AST | Audio Spectrogram Transformer | ViT na łatkach log-mel; SOTA 2021. |
| BEATs | Samonadzorowane audio | Model Microsoftu, iter3 lider AudioSet w 2026. |
| Mixup | Augmentacja par | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Augmentacja maskująca | Wyzeruj losowe pasma czasu i częstotliwości spektrogramu. |
| mAP | Główna metryka multi-label | Średnia precyzja dla wszystkich klas i progów. |

## Dalsza Lektura

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — architektura rekordu 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — domyślne 2024+.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — dominująca augmentacja audio.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 50-klasowy benchmark, który żyje dalej.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — taksonomia 632 klas YouTube; wciąż złoty standard.