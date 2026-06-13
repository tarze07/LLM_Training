# Tekst-na-Mowę (TTS) — Od Tacotron do F5 i Kokoro

> ASR odwraca mowę na tekst; TTS odwraca tekst na mowę. Stos 2026 składa się z trzech części: tekst → tokeny, tokeny → mel, mel → przebieg. Każda część ma domyślny model mieszczący się w laptopie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Problem

Masz string: "Please remind me to water the plants at 6 pm." Potrzebujesz 3-sekundowego klipu audio, który brzmi naturalnie, ma poprawną prozodię (pauzy, akcent), wymawia "plants" z właściwą samogłoską i działa w czasie poniżej 300 ms na CPU dla asystenta głosowego na żywo. Musisz także zmieniać głosy, obsługiwać wejście z przełączaniem kodu ("remind me at 6 pm, daijoubu?") i nie kompromitować się na nazwach własnych.

Nowoczesne pipeline'y TTS wyglądają tak:

1. **Frontend tekstowy.** Normalizuj tekst (daty, liczby, emaile), konwertuj na fonemy lub tokeny pod-słowne, przewiduj cechy prozodii.
2. **Model akustyczny.** Tekst → spektrogram mel. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder.** Mel → przebieg. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), neural codec vocoders w 2024+.

W 2026 podział na akustyczny + vocoder zaciera się dzięki end-to-end diffusion i flow-matching modelom. Jednak model mentalny trzech części wciąż obowiązuje do debugowania.

## Koncepcja

![Tacotron, FastSpeech, VITS, F5/Kokoro obok siebie](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: char-embedding → BiLSTM encoder → location-sensitive attention → autoregresyjny dekoder LSTM emituje ramki mel. Wolny (AR), chwiejny na długim tekście. Wciąż cytowany jako baseline.

**FastSpeech 2 (2020).** Nieautoregresyjny. Predyktor czasu trwania określa, ile ramek mel przypada na każdy fonem. 1-przebieg, 10× szybszy niż Tacotron. Traci nieco naturalności (monotoniczne dopasowanie), ale działa wszędzie.

**VITS (2021).** Wspólnie trenuje enkoder + flow-based duration + HiFi-GAN vocoder end-to-end z wnioskowaniem wariacyjnym. Wysoka jakość, pojedynczy model. Dominujący open-source TTS 2022–2024. Warianty: YourTTS (multi-speaker zero-shot), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Dyfuzyjny transformer na flow matchingu. Naturalna prozodia, zerokrotne klonowanie głosu z 5 sekundami referencyjnego audio. Na szczycie open-source'owych rankingów TTS w 2026. 335M parametrów.

**Kokoro (2024).** Mały (82M), działa na CPU, najlepszy angielski TTS do użycia w czasie rzeczywistym. Zamknięte słownictwo tylko angielski, Apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Komercyjny stan sztuki. ElevenLabs v2.5 z tagami emocji ("[whispered]", "[laughing]") i głosami postaci dominuje w produkcji audiobooków w 2026.

### Ewolucja vocodera

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA w momencie wydania |
| 2018 | WaveRNN | ~czas rzeczywisty | dobry |
| 2020 | HiFi-GAN | 100× czas rzeczywisty | blisko ludzkiego |
| 2022 | BigVGAN | 50× czas rzeczywisty | generalizuje przez mówców/języki |
| 2024 | SNAC, DAC (neural codecs) | zintegrowane z modelami AR | dyskretne tokeny, wydajne bitowo |

Do 2026 większość modeli "TTS" jest end-to-end od tekstu do przebiegu; spektrogram mel jest reprezentacją wewnętrzną.

### Ewaluacja

- **MOS (Mean Opinion Score).** Skala 1–5, crowdsourcing. Wciąż złoty standard; boleśnie wolny.
- **CMOS (Comparative MOS).** Preferencja A-vs-B. Węższe przedziały ufności na adnotację.
- **UTMOS, DNSMOS.** Predyktory MOS bez referencji. Używane w rankingach.
- **CER (Character Error Rate) przez ASR.** Przepuść wyjście TTS przez Whisper, oblicz CER względem tekstu wejściowego. Zastępcza miara zrozumiałości.
- **SECS (Speaker Embedding Cosine Similarity).** Jakość klonowania głosu.

Liczby 2026 na LibriTTS test-clean:

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## Zbuduj To

### Krok 1: fonemizacja wejścia

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Fonemy są uniwersalnym mostem. Unikaj podawania surowego tekstu do czegokolwiek poniżej jakości VITS.

### Krok 2: uruchom Kokoro (domyślne CPU 2026)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = amerykański angielski
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Działa offline, pojedynczy plik, 82M parametrów.

### Krok 3: uruchom F5-TTS z klonowaniem głosu

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Podaj 5-sekundowy referencyjny klip + jego transkrypcję; F5 klonuje prozodię i barwę.

### Krok 4: vocoder HiFi-GAN od zera

Zbyt duży, aby zmieścić się w skrypcie tutoriala, ale kształt jest:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 bloki upsample, łącznie 256x, aby przejść z tempa mel do tempa audio
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Trening: adversarial (dyskryminator na krótkich oknach) + strata rekonstrukcji spektrogramu mel + strata feature-matching. Skomodytyzowane — używaj pretrenowanych checkpointów z repozytorium `hifi-gan` lub nvidia-NeMo.

### Krok 5: pełny pipeline (pseudokod)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Użyj Tego

Stos 2026:

| Situation | Pick |
|-----------|------|
| Angielski asystent głosowy w czasie rzeczywistym | Kokoro (CPU) lub XTTS v2 (GPU) |
| Klonowanie głosu z 5 s referencji | F5-TTS |
| Komercyjne głosy postaci | ElevenLabs v2.5 |
| Narracja audiobooków | ElevenLabs v2.5 lub XTTS v2 + dostrajanie |
| Język niskozasobowy | Trenuj VITS na 5–20 h danych w języku docelowym |
| Ekspresyjne / tagi emocji | ElevenLabs v2.5 lub StyleTTS 2 fine-tune |

Lider open-source w 2026: **F5-TTS dla jakości, Kokoro dla wydajności**. Nie sięgaj po Tacotron, chyba że jesteś historykiem.

## Pułapki

- **Brak normalizatora tekstu.** "Dr. Smith" czyta się jako "Doctor" lub "Drive"? "2026" jako "twenty twenty six" czy "two zero two six"? Normalizuj PRZED fonemizatorem.
- **OOV nazwy własne.** "Ghumare" → "ghyu-mair"? Dostarcz zastępczy model grapheme-to-phoneme dla nieznanych tokenów.
- **Przycinanie (clipping).** Wyjście vocodera rzadko przycina, ale niedopasowanie skali mel przy wnioskowaniu może przekroczyć ±1.0. Zawsze `np.clip(wav, -1, 1)`.
- **Niedopasowanie częstotliwości próbkowania.** Kokoro produkuje 24 kHz; twój downstream pipeline oczekuje 16 kHz → resampluj lub dostaniesz aliasing.

## Wyślij To

Zapisz jako `outputs/skill-tts-designer.md`. Zaprojektuj pipeline TTS dla danego głosu, opóźnienia i języka docelowego.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Buduje słownik fonemów z zabawkowego słownictwa, szacuje czas trwania na fonem i wypisuje fałszywy harmonogram "mel".
2. **Średnie.** Zainstaluj Kokoro, syntetyzuj to samo zdanie głosem `af_bella` i `am_adam`. Porównaj czasy trwania audio i subiektywną jakość.
3. **Trudne.** Nagraj 5-sekundowy referencyjny klip siebie. Użyj F5-TTS, aby go sklonować. Raportuj SECS między referencją a sklonowanym wyjściem.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Phoneme | Jednostka dźwięku | Abstrakcyjna klasa dźwięku; 39 w angielskim (ARPABet). |
| Duration predictor | Jak długo trwa każdy fonem | Wyjście modelu nie-AR; liczba całkowita ramek na fonem. |
| Vocoder | Mel → przebieg | Sieć neuronowa mapująca mel-spec na surowe próbki. |
| HiFi-GAN | Standardowy vocoder | Oparty na GAN; dominujący 2020–2024. |
| MOS | Jakość subiektywna | 1–5 średnia ocena od ludzkich oceniających. |
| SECS | Metryka klonowania głosu | Podobieństwo cosinusowe między osadzeniem mówcy docelowego a wyjściem. |
| F5-TTS | Open-source SOTA 2024 | Flow-matching diffusion; zerokrotne klonowanie. |
| Kokoro | Lider CPU angielskiego | Model 82M parametrów, Apache 2.0. |

## Dalsza Lektura

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — baseline seq2seq.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end flow-based.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — obecny open-source SOTA.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — vocoder wciąż w użyciu w 2026.
- [Kokoro-82M na HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — przyjazny dla CPU angielski TTS 2024.