# Spektrogramy, Skala Mel i Cechy Audio

> Sieci neuronowe nie przetwarzają dobrze surowych przebiegów. Przetwarzają spektrogramy. Spektrogramy mel przetwarzają jeszcze lepiej. Każdy ASR, TTS i klasyfikator audio w 2026 żyje lub umiera przez ten pojedynczy wybór przetwarzania wstępnego.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## Problem

Weź 10-sekundowy klip 16 kHz. To 160 000 floatów, wszystkie w `[-1, 1]`, prawie idealnie nieskorelowane z etykietą "szczekanie psa" lub "słowo kot". Surowy przebieg ma informację, ale w formie, której model nie może łatwo wydobyć. Dwa identyczne fonemy wypowiedziane w odstępie 100 ms mają zupełnie różne surowe próbki.

Spektrogram rozwiązuje ten problem. Zawija szczegóły czasowe tam, gdzie ludzka percepcja je ignoruje (mikrosekundowe drgania) i zachowuje strukturę tam, gdzie percepcja jest skupiona (które częstotliwości są energetyczne, w oknach czasowych ~10–25 ms).

Spektrogramy mel idą dalej. Ludzie postrzegają wysokość dźwięku logarytmicznie: 100 Hz vs 200 Hz brzmi "tak samo odległe" jak 1000 Hz vs 2000 Hz. Skala mel zawija oś częstotliwości, aby to dopasować. Spektrogram w skali mel to najważniejsza cecha w uczeniu maszynowym mowy od 2010 do 2026.

## Koncepcja

![Drabina: przebieg → STFT → spektrogram mel → MFCC](../assets/mel-features.svg)

**STFT (Krótkookresowa Transformata Fouriera).** Pokrój przebieg na zachodzące na siebie ramki (typowe: okno 25 ms, krok 10 ms = 400 próbek / 160 próbek przy 16 kHz). Pomnóż każdą ramkę przez funkcję okna (Hann to domyślny; Hamming ma nieco inny kompromis). Wykonaj FFT każdej ramki. Połącz widma modułów w macierz o kształcie `(n_frames, n_freq_bins)`. To jest twój spektrogram.

**Logarytm modułu.** Surowe moduły obejmują 5-6 rzędów wielkości. Weź `log(|X| + 1e-6)` lub `20 * log10(|X|)`, aby skompresować zakres dynamiczny. Każdy produkcyjny pipeline używa logarytmu modułu, nie surowego modułu.

**Skala mel.** Częstotliwość `f` w Hz mapuje się na mel `m` przez `m = 2595 * log10(1 + f / 700)`. Mapowanie jest w przybliżeniu liniowe poniżej 1 kHz i w przybliżeniu logarytmiczne powyżej. 80 pasm mel pokrywających 0–8 kHz to standardowe wejście ASR.

**Bank filtrów mel.** Zbiór trójkątnych filtrów rozmieszczonych równomiernie w skali mel. Każdy filtr jest ważoną sumą sąsiednich binów FFT. Pomnożenie modułu STFT przez macierz banku filtrów daje spektrogram mel w jednym mnożeniu macierzy.

**Log-mel spektrogram.** `log(mel_spec + 1e-10)`. Wejście Whispera. Wejście Parakeet. Wejście SeamlessM4T. Uniwersalny frontend audio w 2026.

**MFCC.** Weź log-mel spektrogram, zastosuj DCT (typ II), zachowaj pierwsze 13 współczynników. Dekoreluje cechy i dodatkowo kompresuje. Dominująca cecha do około 2015, gdy CNN/Transformery na surowych log-melach dogoniły. Wciąż używane w rozpoznawaniu mówcy (x-vectors, ECAPA).

**Kompromis rozdzielczości.** Większy FFT = lepsza rozdzielczość częstotliwościowa, ale gorsza rozdzielczość czasowa. 25 ms / 10 ms to domyślne dla audio-ML; 50 ms / 12,5 ms dla muzyki; 5 ms / 2 ms dla detekcji stanów przejściowych (uderzenia perkusji, plozje).

```figure
spectrogram-window
```

## Zbuduj To

### Krok 1: ramkuj przebieg

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

10-sekundowy klip 16 kHz z `frame_len=400, hop=160` daje 998 ramek.

### Krok 2: okno Hanny

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Mnożenie element po elemencie przed FFT. Usuwa wyciek widmowy spowodowany obcięciem na niezerowych końcach.

### Krok 3: moduł STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Produkcja używa `torch.stft` lub `librosa.stft` (FFT-owane, wektoryzowane). Pętla tutaj jest pedagogiczna; działa na krótkich klipach w `code/main.py`.

### Krok 4: bank filtrów mel

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mels pokrywających 0–8 kHz z `n_fft=400` daje macierz `(80, 201)`. Pomnóż moduł STFT `(n_frames, 201)` przez transpozycję, aby otrzymać spektrogram mel `(n_frames, 80)`.

### Krok 5: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Typowe alternatywy: `librosa.power_to_db` (dB znormalizowane względem referencji), `10 * log10(power + eps)`. Whisper używa bardziej złożonej procedury clip + normalize (patrz `log_mel_spectrogram` Whispera).

### Krok 6: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Zastosuj DCT do każdej ramki log-mel, zachowaj pierwsze 13 współczynników. To jest twoja macierz MFCC. Pierwszy współczynnik jest zwykle pomijany (koduje ogólną energię).

## Użyj Tego

Stos 2026:

| Task | Features |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, 10 ms hop, 25 ms window |
| TTS acoustic model (VITS, F5-TTS, Kokoro) | 80 mels, 5–12 ms hop dla precyzyjnej kontroli czasowej |
| Audio classification (AST, PANNs, BEATs) | 128 log-mels, 10 ms hop |
| Speaker embedding (ECAPA-TDNN, WavLM) | 80 log-mels lub raw-waveform SSL |
| Music (MusicGen, Stable Audio 2) | EnCodec discrete tokens (nie mels) |
| Keyword spotting | 40 MFCC dla małych urządzeń |

Reguła kciuka: **jeśli nie pracujesz z muzyką, zacznij od 80 log-mels.** Ciężar dowodu spoczywa na każdym odstępstwie.

## Pułapki, które wciąż występują w 2026

- **Niezgodność liczby melów.** Trening z 80 mels, wnioskowanie z 128 mels. Cicha porażka. Loguj kształt cech na obu końcach.
- **Niezgodność częstotliwości próbkowania powyżej.** Mels obliczone przy 22.05 kHz wyglądają inaczej niż przy 16 kHz. Napraw SR *przed* featuryzacją.
- **dB vs log.** Whisper oczekuje log-mel, nie dB-mel. Niektóre pipeline'y HF wykrywają automatycznie; twój niestandardowy kod nie będzie.
- **Dryf normalizacji.** Normalizacja per-utterance podczas treningu, globalna normalizacja podczas wnioskowania. Błąd produkcyjny, który podwaja WER.
- **Wyciek z paddingu.** Padding zerami na końcu klipu produkuje płaskie widmo w końcowych ramkach. Paduj symetrycznie lub replikuj.

## Wyślij To

Zapisz jako `outputs/skill-feature-extractor.md`. Umiejętność wybiera typ cech, liczbę melów, ramkę/krok i normalizację dla danego modelu docelowego.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Syntetyzuje ćwierkanie (częstotliwość przemiatana 200 → 4000 Hz) i wypisuje argmax pasma mel na ramkę. Opcjonalnie wykreśl i potwierdź, że pasuje do przemiatania.
2. **Średnie.** Uruchom ponownie z `n_mels` w `{40, 80, 128}` i `frame_len` w `{200, 400, 800}`. Zmierz szerokość pasma ostrego piku w osi czasu. Która kombinacja najlepiej rozdziela ćwierkanie?
3. **Trudne.** Zaimplementuj `power_to_db` i porównaj dokładność ASR małego klasyfikatora CNN na AudioMNIST używając (a) surowego log-mel, (b) dB-mel z `ref=max`, (c) MFCC-13 + delta + delta-delta. Raportuj dokładność top-1.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Frame | Wycinek | 25 ms kawałek przebiegu podawany do jednej FFT. |
| Hop | Krok | Próbki między kolejnymi ramkami; 10 ms to domyślne ASR. |
| Window | Hann/Hamming | Mnożnik punktowy, który ścina krawędzie ramki do zera. |
| STFT | Generator spektrogramu | Ramkowana + okienkowana FFT; daje macierz czas × częstotliwość. |
| Mel | Zawinięta częstotliwość | Skala percepcji logarytmicznej; `m = 2595·log10(1 + f/700)`. |
| Filterbank | Macierz | Trójkątne filtry rzutujące STFT na pasma mel. |
| Log-mel | Wejście Whispera | `log(mel_spec + eps)`; standaryzowane w 2026. |
| MFCC | Staroszkolna cecha | DCT log-mel; 13 współczynników, nieskorelowane. |

## Dalsza Lektura

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — artykuł o MFCC.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — oryginalna skala mel.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — przeczytaj referencyjną implementację.
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — odniesienie dla `mfcc`, `melspectrogram` i hop/window.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — produkcyjny pipeline dla modeli Parakeet + Canary.