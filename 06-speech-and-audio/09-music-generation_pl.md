# Generowanie Muzyki — MusicGen, Stable Audio, Suno i Trzęsienie Ziemi Licencyjne

> Generowanie muzyki w 2026: Suno v5 i Udio v4 dominują komercyjnie; MusicGen, Stable Audio Open i ACE-Step prowadzą w open-source. Problem techniczny jest w większości rozwiązany. Problem prawny (ugoda Warner Music za $500M, ugoda UMG) przekształcił dziedzinę w latach 2025-2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## Problem

Tekst → 30-sekundowy do 4-minutowy klip muzyczny, z tekstem, wokalem i strukturą. Trzy podproblemy:

1. **Generowanie instrumentalne.** Tekst typu "lo-fi hip-hop drums with warm keys" → audio. MusicGen, Stable Audio, AudioLDM.
2. **Generowanie piosenek (z wokalem + tekstem).** "Country song about rainy Texas nights" → pełna piosenka. Suno, Udio, YuE, ACE-Step.
3. **Warunkowe / sterowane.** Rozszerz istniejący klip, wygeneruj ponownie bridge, zamień gatunek, separuj ścieżki lub inpaint. Inpainting Udio + separacja ścieżek to funkcja do dorównania w 2026.

## Koncepcja

![Generowanie muzyki: token-LM vs dyfuzja, mapa modeli 2026](../assets/music-generation.svg)

### Token LM nad tokenami kodeka neuralnego

**MusicGen** Meta (2023, MIT) i wiele pochodnych: warunkuj na osadzeniach tekstu/melodii, autoregresyjnie przewiduj tokeny EnCodec (32 kHz, 4 codebooki), dekoduj z EnCodec. 300M - 3.3B parametrów. Silny baseline; ma problemy powyżej 30 sekund.

**ACE-Step** (open-source, 4B XL wydany kwiecień 2026) rozszerza to na generowanie pełnych piosenek warunkowanych tekstem. Najbliższe Suno w otwartej społeczności.

### Dyfuzja nad mels lub latentami

**Stable Audio (2023)** i **Stable Audio Open (2024)**: latentna dyfuzja na skompresowanym audio. Doskonałe w pętlach, projektowaniu dźwięku, teksturach ambientowych. Nie świetne w ustrukturyzowanych pełnych piosenkach.

**AudioLDM / AudioLDM2**: tekst-do-audio przez dyfuzję latentną w stylu T2I, uogólnione do muzyki, efektów dźwiękowych, mowy.

### Hybrydowe (produkcyjne) — Suno, Udio, Lyria

Zamknięte wagi. Prawdopodobnie AR codec LM + vocoder oparty na dyfuzji z wyspecjalizowanymi głowami wokalu/perkusji/melodii. Suno v5 (2026) jest liderem jakości ELO 1293. Udio v4 dodaje inpainting + separację ścieżek (bas, perkusja, wokal jako osobne pobieranie).

### Ewaluacja

- **FAD (Fréchet Audio Distance).** Odległość na poziomie osadzeń między rozkładem wygenerowanego a prawdziwego audio przy użyciu cech VGGish lub PANNs. Niższe znaczy lepiej. MusicGen small: 4.5 FAD na MusicCaps; SOTA ~3.0.
- **Muzykalność (subiektywna).** Preferencja ludzka. Suno v5 ELO 1293 prowadzi.
- **Dopasowanie tekst-audio.** Wynik CLAP między promptem a wyjściem.
- **Artefakty muzykalności.** Przejścia off-beat, dryf frazy wokalnej, utrata struktury powyżej 30 s.

## Mapa modeli 2026

| Model | Params | Długość | Wokal | Licencja |
|-------|--------|---------|-------|----------|
| MusicGen-large | 3.3B | 30 s | nie | MIT |
| Stable Audio Open | 1.2B | 47 s | nie | Stability non-commercial |
| ACE-Step XL (kwi 2026) | 4B | &gt; 2 min | tak | Apache-2.0 |
| YuE | 7B | &gt; 2 min | tak, wielojęzyczny | Apache-2.0 |
| Suno v5 (zamknięty) | ? | 4 min | tak, ELO 1293 | komercyjna |
| Udio v4 (zamknięty) | ? | 4 min | tak + ścieżki | komercyjna |
| Google Lyria 3 (zamknięty) | ? | czas rzeczywisty | tak | komercyjna |
| MiniMax Music 2.5 | ? | 4 min | tak | API komercyjne |

## Krajobraz prawny (2025-2026)

- **Ugoda Warner Music vs Suno.** $500M. WMG ma teraz nadzór nad podobieństwem AI, prawami muzycznymi i utworami generowanymi przez użytkowników na Suno. Podobna ugoda UMG na Udio.
- **EU AI Act** + **California SB 942**: muzyka generowana przez AI musi być oznaczona.
- **Riffusion / MusicGen** na MIT nie mają bagażu zgodności, ale też nie mają komercyjnego wokalu.

Wzory bezpieczne do wdrożenia:

1. Generuj tylko instrumentalne (MusicGen, Stable Audio Open, wyjścia MIT/CC0).
2. Używaj komercyjnych API (Suno, Udio, ElevenLabs Music) z licencją na generację.
3. Trenuj na własnym lub licencjonowanym katalogu (większość firm kończy tutaj).
4. Oznaczaj generacje watermarkami + metadanymi.

## Zbuduj To

### Krok 1: generuj z MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Trzy rozmiary: `small` (300M, szybki), `medium` (1.5B), `large` (3.3B). Small wystarcza, aby sprawdzić "czy pomysł działa."

### Krok 2: warunkowanie melodią

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody przyjmuje chromatogram i zachowuje melodię, zmieniając barwę. Przydatne do "daj mi tę melodię jako kwartet smyczkowy."

### Krok 3: ewaluacja FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Oblicza odległość osadzeń VGGish. Przydatne do testów regresji na poziomie gatunku; nie zastępuje ludzkich słuchaczy.

### Krok 4: dodanie do przepływu pracy LLM-muzyka

Połącz z pomysłami z Lekcji 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Użyj Tego

| Cel | Stos |
|-----|------|
| Projektowanie dźwięku instrumentalnego | Stable Audio Open |
| Muzyka do gier / adaptacyjna | Google Lyria RealTime (zamknięty) |
| Pełne piosenki z wokalem (komercyjne) | Suno v5 lub Udio v4 z jawną licencją |
| Pełne piosenki z wokalem (otwarte) | ACE-Step XL lub YuE |
| Krótki dżingiel reklamowy | MusicGen z melodią warunkowaną hummingiem |
| Tło do widea muzycznego | MusicGen + Stable Video Diffusion |

## Pułapki, które wciąż występują w 2026

- **Prompty piorące prawa autorskie.** "Song in the style of Taylor Swift" — komercyjne Suno/Udio filtrują to teraz, otwarte modele nie. Dodaj własną listę filtrów.
- **Powtarzanie / dryf powyżej 30 s.** Modele AR zapętlają się. Crossfaduj wiele generacji lub użyj ACE-Step dla spójności strukturalnej.
- **Dryf tempa.** Modele odchodzą od BPM. Używaj tagów BPM w prompcie i post-filtruj z `beat_track` librosy.
- **Zrozumiałość wokalu.** Suno jest doskonałe; otwarte modele są często niewyraźne na słowach. Jeśli tekst ma znaczenie, użyj komercyjnego API lub dostrój.
- **Wyjście mono.** Otwarte modele generują mono lub pseudo-stereo. Ulepsz z właściwą rekonstrukcją stereo (ezst, stereo diffusion Cartesia).

## Wyślij To

Zapisz jako `outputs/skill-music-designer.md`. Wybierz model, strategię licencyjną, plan długości/struktury i metadane ujawnienia dla wdrożenia generowania muzyki.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Produkuje "generatywną" progresję akordów + wzór perkusyjny jako symbole ASCII — kreskówkę generowania muzyki. Odtwórz przez dowolny renderer MIDI.
2. **Średnie.** Zainstaluj `audiocraft`, wygeneruj 10-sekundowe klipy dla 4 promptów gatunkowych z MusicGen-small, zmierz FAD względem referencyjnego zestawu gatunków.
3. **Trudne.** Używając ACE-Step (lub MusicGen-melody), wygeneruj trzy warianty tej samej melodii z różnymi promptami barwy. Oblicz podobieństwo CLAP do promptu, aby zweryfikować dopasowanie.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FAD | FID audio | Odległość Frécheta między rozkładami osadzeń rzeczywistego a wygenerowanego audio. |
| Chromagram | Melodia jako wysokości | 12-wymiarowy wektor na ramkę; wejście do warunkowania melodią. |
| Stems | Ścieżki instrumentów | Rozdzielone bas / perkusja / wokal / melodia jako WAV. |
| Inpainting | Regeneruj sekcję | Zamasuj okno czasowe; model regeneruje tylko to. |
| CLAP | CLIP tekst-audio | Osadzenie kontrastywne tekst-audio; ewaluuje dopasowanie tekst-audio. |
| EnCodec | Kodek muzyczny | Kodek neuralny Meta używany przez MusicGen; 32 kHz, 4 codebooki. |

## Dalsza Lektura

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — otwarty benchmark autoregresyjny.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — domyślne do projektowania dźwięku.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — otwarty 4B generator pełnych piosenek, kwiecień 2026.
- [Suno v5 platform docs](https://suno.com) — lider jakości komercyjnej.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — latentna dyfuzja dla muzyki + efektów dźwiękowych.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — precedens z listopada 2025.