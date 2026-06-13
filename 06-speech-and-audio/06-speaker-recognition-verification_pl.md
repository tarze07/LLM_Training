# Rozpoznawanie i Weryfikacja Mówcy

> ASR pyta "co powiedzieli?" Rozpoznawanie mówcy pyta "kto to powiedział?" Matematyka wygląda podobnie — osadzenia plus cosinus — ale każda produkcyjna decyzja opiera się na jednej liczbie EER.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## Problem

Użytkownik wypowiada hasło. Chcesz wiedzieć: czy to osoba, za którą się podaje (*weryfikacja*, 1:1), czy to pierwsza osoba w twojej bazie rejestracji (*identyfikacja*, 1:N)? Albo żadna — czy to nieznany mówca (*open-set*)?

Przed 2018: GMM-UBM + i-wektory. Przyzwoity EER, ale wrażliwy na zmianę kanału (telefon vs laptop) i emocje. 2018–2022: x-wektory (TDNN backbone trenowany z angular margin). 2022+: ECAPA-TDNN i WavLM-large embeddings. Do 2026 dziedzinę dominują trzy modele i jedna metryka.

Metryką jest **EER** — Equal Error Rate. Ustaw próg decyzyjny tak, aby False Accept Rate = False Reject Rate. Punkt przecięcia to EER. Używany w każdej publikacji, każdym rankingu, każdym zapytaniu ofertowym.

## Koncepcja

![Pipeline rejestracji + weryfikacji z osadzeniem + cosinusem + EER](../assets/speaker-verification.svg)

**Pipeline.** Rejestracja: nagraj 5–30 sekund docelowego mówcy; oblicz osadzenie o stałym wymiarze (192-d dla ECAPA-TDNN, 256-d dla WavLM-large). Weryfikacja: pobierz osadzenie wypowiedzi testowej; oblicz podobieństwo cosinusowe; porównaj z progiem.

**ECAPA-TDNN (2020, wciąż dominujący 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. Bloki 1D conv z squeeze-excitation, multi-head attention pooling, następnie warstwa liniowa do 192-d. Trenowany na VoxCeleb 1+2 (2700 mówców, 1.1M wypowiedzi) ze stratą Additive Angular Margin (AAM-softmax).

**WavLM-SV (2022+).** Dostrój pretrenowany backbone WavLM-large SSL z AAM loss. Wyższa jakość, ale wolniejszy — 300+ MB vs 15 MB.

**x-wektor (baseline).** TDNN + statistics pooling. Klasyczny; wciąż użyteczny na CPU / edge.

**AAM-softmax.** Standardowy softmax z dodanym marginesem `m` w przestrzeni kątowej: `cos(θ + m)` dla poprawnej klasy. Wymusza separację kątową między klasami. Typowe `m=0.2`, skala `s=30`.

### Scoring

- **Cosinus** między osadzeniem rejestracji i testowym. Decyzja oparta na progu.
- **PLDA (Probabilistic LDA).** Projekcja osadzeń do przestrzeni ukrytej, gdzie ten-sam-mówca vs inny-mówca ma iloraz wiarogodności w zamkniętej formie. Dodany na cosinus dla +10–20% redukcji EER. Standard przed 2020; teraz używany tylko w zamkniętych zestawach.
- **Normalizacja wyniku.** `S-norm` lub `AS-norm`: normalizuj każdy wynik względem kohorty średnich i odchyleń impostorów. Niezbędne do ewaluacji cross-domenowej.

### Liczby, które powinieneś znać (2026)

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-wektor (klasyczny) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentacja + osadzenie | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diaryzacja

"Kto mówił kiedy" w klipie z wieloma mówcami. Pipeline: VAD → segmentacja → osadź każdy segment → grupowanie (aglomeracyjne lub spektralne) → wygładź granice. Nowoczesny stos: `pyannote.audio` 3.1, który łączy segmentację mówców + osadzenie + grupowanie w jednym wywołaniu. SOTA DER na AMI w 2026 to ~15% (spadek z 23% w 2022).

## Zbuduj To

### Krok 1: zabawkowe osadzenie ze statystyk MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

To nie jest SOTA — tylko do nauczania. `code/main.py` używa tego jako proof-of-concept na syntetycznych danych mówców.

### Krok 2: podobieństwo cosinusowe + próg

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Krok 3: EER z par podobieństw

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Zwraca (eer, threshold_at_eer). Raportuj oba.

### Krok 4: produkcja z SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# rejestracja: uśrednij osadzenia 3-5 czystych próbek
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# weryfikacja
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # typowy próg ECAPA; dostosuj do swoich danych
```

### Krok 5: diaryzacja z pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Użyj Tego

Stos 2026:

| Situation | Pick |
|-----------|------|
| Zamknięta 1:1 weryfikacja, edge | ECAPA-TDNN + próg cosinusowy |
| Otwarta weryfikacja, chmura | WavLM-SV + AS-norm |
| Diaryzacja (spotkania, podcasty) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (wykrywanie replay/deepfake) | AASIST lub RawNet2 |
| Mały embedded (KWS + rejestracja) | Titanet-Small (NeMo) |

## Pułapki

- **Niedopasowanie kanału.** Model trenowany na VoxCeleb (wideo internetowe) ≠ audio z rozmowy telefonicznej. Zawsze ewaluuj na docelowym kanale.
- **Krótkie wypowiedzi.** EER gwałtownie spada poniżej 3 sekund testowego audio.
- **Rejestracja z szumem.** Jedna zaszumiona rejestracja zatruwa kotwicę. Użyj ≥3 czystych próbek i uśrednij.
- **Stały próg w różnych warunkach.** Zawsze dostrajaj próg na wstrzymanym zbiorze deweloperskim z docelowej domeny.
- **Cosinus na nieznormalizowanych osadzeniach.** Najpierw znormalizuj L2; w przeciwnym razie dominuje moduł.

## Wyślij To

Zapisz jako `outputs/skill-speaker-verifier.md`. Wybierz model, protokół rejestracji, plan dostrajania progu i zabezpieczenia przed oszustwami.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Buduje syntetycznych "mówców" (różne profile tonalne), rejestruje, oblicza EER na 100-parowej liście prób.
2. **Średnie.** Użyj SpeechBrain ECAPA na 30 wypowiedziach VoxCeleb1 (5 mówców × 6 każdy). Oblicz EER z cosinusem vs PLDA.
3. **Trudne.** Zbuduj pełny pipeline rejestracja → diaryzacja → weryfikacja z `pyannote.audio`. Oceń DER na AMI dev set.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| EER | Metryka nagłówkowa | Próg, gdzie False Accept = False Reject. |
| Verification | 1:1 | "Czy to Alicja?" |
| Identification | 1:N | "Kto mówi?" |
| Open-set | Możliwe nieznane | Zbiór testowy może zawierać niezarejestrowanych mówców. |
| Enrollment | Rejestracja | Obliczanie referencyjnego osadzenia mówcy. |
| AAM-softmax | Strata | Softmax z addytywnym marginesem kątowym; wymusza separację klastrów. |
| PLDA | Klasyczny scoring | Probabilistyczne LDA; scoring ilorazem wiarogodności na osadzeniach. |
| DER | Metryka diaryzacji | Diarization Error Rate — miss + false alarm + confusion. |

## Dalsza Lektura

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — klasyczny artykuł o głębokich osadzeniach.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — dominująca architektura 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — backbone SSL dla SV i diaryzacji.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — produkcyjny stos diaryzacji + osadzeń.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — aktualne rankingi EER między modelami.