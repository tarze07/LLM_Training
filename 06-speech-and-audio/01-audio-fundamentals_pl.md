# Podstawy Audio — Przebiegi, Próbkowanie, Transformata Fouriera

> Przebiegi to surowy sygnał. Spektrogramy to reprezentacja. Cechy mel to forma przyjazna dla ML. Każdy nowoczesny pipeline ASR i TTS pokonuje tę drabinę, a pierwszym szczeblem jest zrozumienie próbkowania i transformaty Fouriera.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## Problem

Mikrofon wytwarza sygnał ciśnienia w funkcji czasu. Twoja sieć neuronowa konsumuje tensory. Pomiędzy nimi znajduje się stos konwencji, które naruszone powodują ciche błędy: model trenuje się dobrze, ale WER podwaja się, TTS wysyła syk, a system klonowania głosu zapamiętuje mikrofon zamiast mówcy.

Każdy błąd w systemach mowy sprowadza się do jednego z trzech pytań:

1. Z jaką częstotliwością próbkowania nagrano dane i czego oczekuje model?
2. Czy sygnał jest aliasingowany?
3. Czy operujesz na surowych próbkach czy na reprezentacji częstotliwościowej?

Zrozum je dobrze, a reszta Fazy 6 będzie do opanowania. Popełnij je źle, a nawet Whisper-Large-v4 będzie produkować śmieci.

## Koncepcja

![Przebieg, próbkowanie, DFT i pasma częstotliwości wizualizowane](../assets/audio-fundamentals.svg)

**Przebieg (waveform).** Jednowymiarowa tablica floatów w zakresie `[-1.0, 1.0]`. Indeksowana numerem próbki. Aby przeliczyć na sekundy, podziel przez częstotliwość próbkowania: `t = n / sr`. 10-sekundowy klip przy 16 kHz to tablica 160 000 floatów.

**Częstotliwość próbkowania (sr).** Liczba próbek na sekundę. Typowe wartości w 2026:

| Rate | Use |
|------|-----|
| 8 kHz | Telefonia, legacy VOIP. Nyquist przy 4 kHz zabija spółgłoski. Unikaj dla ASR. |
| 16 kHz | Standard ASR. Whisper, Parakeet, SeamlessM4T v2 konsumują 16 kHz. |
| 22.05 kHz | Trening vocoderów TTS dla starszych modeli. |
| 24 kHz | Nowoczesne TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | Audio CD, muzyka. |
| 48 kHz | Film, profesjonalne audio, wysokiej wierności TTS (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** Częstotliwość próbkowania `sr` może jednoznacznie reprezentować częstotliwości do `sr/2`. Granica `sr/2` to *częstotliwość Nyquista*. Energia powyżej Nyquista ulega *aliasingowi* — zawijaniu w dół do niższych częstotliwości — i zniekształca sygnał. Zawsze stosuj filtr dolnoprzepustowy przed zmniejszaniem próbkowania.

**Głębia bitowa.** 16-bitowy PCM (signed int16, zakres ±32 767) to uniwersalny format wymiany. 24-bit dla muzyki, 32-bit float dla wewnętrznego DSP. Biblioteki takie jak `soundfile` czytają int16, ale udostępniają tablice float32 w zakresie `[-1, 1]`.

**Transformata Fouriera.** Każdy skończony sygnał jest sumą sinusoid o różnych częstotliwościach. Dyskretna Transformata Fouriera (DFT) oblicza dla `N` próbek `N` zespolonych współczynników — jeden na pasmo częstotliwości. `bin k` odpowiada częstotliwości `k · sr / N` Hz. Moduł to amplituda na danej częstotliwości, kąt to faza.

**FFT.** Szybka Transformata Fouriera: algorytm `O(N log N)` dla DFT, gdy `N` jest potęgą 2. Każda biblioteka audio używa FFT pod maską. 1024-próbkowa FFT przy 16 kHz daje 512 użytecznych pasm częstotliwości obejmujących 0–8 kHz z rozdzielczością 15,6 Hz.

**Ramkowanie + okno.** Nie poddajemy całego klipu FFT. Dzielimy go na zachodzące na siebie *ramki* (zazwyczaj 25 ms z krokiem 10 ms), mnożymy każdą ramkę przez funkcję okna (Hann, Hamming), aby wyeliminować nieciągłości na krawędziach, a następnie poddajemy FFT każdej ramki. To jest Krótkookresowa Transformata Fouriera (STFT). Lekcja 02 zaczyna się od tego miejsca.

```figure
mel-scale
```

## Zbuduj To

### Krok 1: odczytaj klip i wykreśl przebieg

`code/main.py` używa tylko modułu `wave` ze standardowej biblioteki, aby zachować demo bez zależności. W produkcji użyjesz `soundfile` lub `torchaudio.load` (oba zwracają krotki `(waveform, sr)`):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Krok 2: syntetyzuj sinusoidę od podstaw

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz sinus (A koncertowe) przy 16 kHz przez 1 sekundę to 16 000 floatów. Zapisz używając `wave.open(..., "wb")` z kodowaniem 16-bit PCM.

### Krok 3: oblicz DFT ręcznie

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — w porządku dla `N=256`, aby potwierdzić poprawność, bezużyteczne dla prawdziwego audio. Prawdziwy kod woła `numpy.fft.rfft` lub `torch.fft.rfft`.

### Krok 4: znajdź dominującą częstotliwość

Indeks piku modułu `k_star` odpowiada częstotliwości `k_star * sr / N`. Uruchomienie tego na 440 Hz sinusie powinno zwrócić pik w binie `440 * N / sr`.

### Krok 5: zademonstruj aliasing

Spróbkuj 7 kHz sinus z częstotliwością 10 kHz (Nyquist = 5 kHz). Ton 7 kHz jest powyżej Nyquista i składa się do `10 − 7 = 3 kHz`. Pik FFT pojawia się przy 3 kHz. To klasyczna demonstracja aliasingu i powód, dla którego każdy DAC/ADC ma wbudowany filtr dolnoprzepustowy.

## Użyj Tego

Stos, który faktycznie wdrożysz w 2026:

| Task | Library | Why |
|------|---------|-----|
| Odczyt/zapis WAV/FLAC/OGG | `soundfile` (wrapper libsndfile) | Najszybszy, stabilny, zwraca float32. |
| Resample | `torchaudio.transforms.Resample` lub `librosa.resample` | Wbudowane poprawne anty-aliasingowanie. |
| STFT / Mel | `torchaudio` lub `librosa` | GPU-friendly; ekosystem PyTorch. |
| Streaming w czasie rzeczywistym | `sounddevice` lub `pyaudio` | Wieloplatformowe bindingi PortAudio. |
| Inspekcja pliku | `ffprobe` lub `soxi` | CLI, szybki, raportuje sr/kanały/kodek. |

Reguła decyzyjna: **dopasuj częstotliwość próbkowania zanim dopasujesz cokolwiek innego**. Whisper oczekuje 16 kHz mono float32. Podaj mu 44,1 kHz stereo, a dostaniesz śmieci wyglądające jak błąd modelu.

## Wyślij To

Zapisz jako `outputs/skill-audio-loader.md`. Umiejętność pomaga sprawdzić, czy wejście audio pasuje do oczekiwań modelu docelowego i poprawnie resampluje, gdy nie pasuje.

## Ćwiczenia

1. **Łatwe.** Syntetyzuj 1-sekundową mieszankę 220 Hz + 440 Hz + 880 Hz przy 16 kHz. Uruchom DFT. Potwierdź trzy piki w oczekiwanych binach.
2. **Średnie.** Nagraj 3-sekundowy WAV swojego głosu przy 48 kHz. Zmniejsz próbkowanie do 16 kHz używając `torchaudio.transforms.Resample` (z anty-aliasingiem), a następnie do 16 kHz używając naiwnej decymacji (co trzecia próbka). Wykonaj FFT na obu. Gdzie pojawia się aliasing?
3. **Trudne.** Zbuduj STFT od zera używając tylko `math` i DFT z Kroku 3. Rozmiar ramki 400, krok 160, okno Hanna. Wykreśl moduły za pomocą `matplotlib.pyplot.imshow`. To jest spektrogram z Lekcji 02.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sample rate | Liczba próbek na sekundę | Częstotliwość w Hz, z jaką ADC mierzy sygnał. |
| Nyquist | Maksymalna częstotliwość do reprezentacji | `sr/2`; energia powyżej aliasinguje się w dół. |
| Bit depth | Rozdzielczość każdej próbki | `int16` = 65 536 poziomów; `float32` = 24-bit precyzja w `[-1, 1]`. |
| DFT | Transformata Fouriera dla sekwencji | `N` próbek → `N` zespolonych współczynników częstotliwości. |
| FFT | Szybka DFT | Algorytm `O(N log N)` wymagający `N` = potęgi 2. |
| Bin | Kolumna częstotliwości | `k · sr / N` Hz; rozdzielczość = `sr / N`. |
| STFT | Spektrogram pod maską | Ramkowana + okienkowana FFT w czasie. |
| Aliasing | Duchy częstotliwości | Energia powyżej Nyquista odbijająca się do niższych binów. |

## Dalsza Lektura

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — artykuł stojący za twierdzeniem o próbkowaniu.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — darmowy, kanoniczny podręcznik DSP.
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — praktyczny przewodnik z kodem.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — odniesienie, dlaczego rzeczywiste audio nie jest czystą sinusoidą.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — intuicja pasm częstotliwości wyjaśniona w 10 minut.
