# Zabezpieczenia Przed Podszywaniem Głosowym i Watermarkowanie Audio — ASVspoof 5, AudioSeal, WaveVerify

> Klonowanie głosu wdrożyło się szybciej niż obrona. Produkcyjne systemy głosowe 2026 potrzebują dwóch rzeczy: detektora (AASIST, RawNet2), który klasyfikuje prawdziwą vs fałszywą mowę, i watermarka (AudioSeal), który przetrwa kompresję i edycję. Wdróż oba lub nie wdrażaj klonowania głosu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## Problem

Trzy powiązane zabezpieczenia:

1. **Anti-spoofing / wykrywanie deepfake.** Mając klip audio, czy jest syntetyczny czy prawdziwy? Benchmarki ASVspoof (ASVspoof 2019 → 2021 → 5) to złoty standard.
2. **Watermarkowanie audio.** Osadź niedostrzegalny sygnał w wygenerowanym audio, który detektor może później wydobyć. AudioSeal (Meta) i WavMark to otwarte opcje.
3. **Potwierdzona proweniencja.** Kryptograficzne podpisywanie plików audio + metadane. C2PA / Content Authenticity Initiative.

Detekcja obsługuje przeciwników, którzy nie współpracują. Watermarkowanie obsługuje zgodność — wygenerowane przez AI audio powinno być identyfikowalne jako takie. Oba są wymagane w 2026.

## Koncepcja

![Anti-spoofing vs watermarking vs proweniencja — trzy warstwy obrony](../assets/spoofing-watermark.svg)

### ASVspoof 5 — benchmark 2024-2025

Największe zmiany względem poprzednich edycji:

- **Dane crowdsourcingowe** (nie studyjnie czyste) — realistyczne warunki.
- **~2000 mówców** (vs ~100 wcześniej).
- **32 algorytmy ataku.** TTS + konwersja głosu + perturbacje adwersarialne.
- **Dwa tory.** Countermeasure (CM) samodzielna detekcja; Spoofing-robust ASV (SASV) dla systemów biometrycznych.

Stan sztuki na ASVspoof 5: ~7.23% EER. Na starszym ASVspoof 2019 LA: 0.42% EER. Rzeczywiste wdrożenie: oczekuj 5-10% EER na dzikich klipach.

### AASIST i RawNet2 — rodziny modeli detekcyjnych

**AASIST** (2021, aktualizowane do 2026). Uwaga grafowa na cechach spektralnych. Obecny SOTA w zadaniu przeciwdziałania ASVspoof 5.

**RawNet2.** Konwolucyjny frontend na surowym przebiegu + backbone TDNN. Prostszy baseline; wciąż konkurencyjny po dostrojeniu.

**NeXt-TDNN + cechy SSL.** Wariant 2025: styl ECAPA + cechy WavLM + focal loss. Osiąga 0.42% EER na ASVspoof 2019 LA.

### AudioSeal — domyślny watermark 2024

**AudioSeal** Meta (sty 2024, v0.2 gru 2024). Kluczowy projekt:

- **Zlokalizowany.** Wykrywa watermark na ramkę przy rozdzielczości próbkowania 16 kHz (1/16000 s).
- **Generator + detektor trenowane razem.** Generator uczy się osadzać niesłyszalny sygnał; detektor uczy się go znajdować przez augmentacje.
- **Solidny.** Przetrwa kompresję MP3 / AAC, EQ, przesunięcie prędkości ±10%, mieszanie szumu +10 dB SNR.
- **Szybki.** Detektor działa przy 485× czasu rzeczywistego; 1000× szybciej niż WavMark.
- **Pojemność.** 16-bitowy ładunek (może kodować ID modelu, znacznik czasu generacji, ID użytkownika) możliwy do osadzenia w każdej wypowiedzi.

### WavMark

Otwarty baseline sprzed AudioSeal. Odwracalna sieć neuronowa, 32 bity/sek. Problemy:

- Synchronizacja brute-force jest wolna.
- Może być usunięty przez szum gaussowski lub kompresję MP3.
- Nie przyjazny dla czasu rzeczywistego.

### WaveVerify (lipiec 2025)

Adresuje słabości AudioSeal — szczególnie manipulacje czasowe (odwrócenie, prędkość). Używa generatora opartego na FiLM + detektora Mixture-of-Experts. Konkurencyjny z AudioSeal na standardowych atakach; obsługuje edycje czasowe.

### Luka wykorzystywana przez przeciwników

Z AudioMarkBench: "pod pitch shift, wszystkie watermarki wykazują Bit Recovery Accuracy poniżej 0.6, wskazując na prawie całkowite usunięcie." **Pitch-shift to uniwersalny atak.** Żaden watermark 2026 nie jest w pełni odporny na agresywną modyfikację wysokości. Dlatego potrzebujesz detekcji (AASIST) obok watermarkowania.

### C2PA / Content Authenticity Initiative

Nie technika ML — format manifestu. Pliki audio niosą kryptograficznie podpisane metadane o narzędziu tworzenia, autorze, dacie. Audobox / Seamless używają tego. Dobre dla proweniencji; nie robi nic, jeśli zły aktor ponownie koduje i usuwa metadane.

## Zbuduj To

### Krok 1: prosty detektor cech spektralnych (zabawkowy)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Syntetyczna mowa często ma niezwykle płaską energię wysokich częstotliwości. Produkcyjne detektory używają AASIST, nie tego. Ale intuicja się utrzymuje.

### Krok 2: AudioSeal osadź + wykryj

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float w [0, 1] — prawdopodobieństwo obecności watermarka
# decoded_payload: 16 bitów; dopasuj do osadzonego ładunku
```

### Krok 3: ewaluacja — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Krok 4: integracja produkcyjna

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Każda generacja wysyła: (1) watermark, (2) podpisany manifest, (3) dziennik audytu zgodny z polityką retencji.

## Użyj Tego

| Przypadek użycia | Zabezpieczenie |
|-----------------|---------------|
| Wdrażanie TTS / klonowania głosu | AudioSeal embed na każdym wyjściu (nie podlega negocjacji) |
| Biometryczne odblokowanie głosem | AASIST + ECAPA ensemble; wyzwanie żywotności |
| Wykrywanie oszustw w call-center | AASIST na 20% próbki przychodzących rozmów |
| Autentyczność podcastu | C2PA signing przy przesyłaniu, AudioSeal jeśli AI-generowane |
| Badania / trenowanie detektorów | ASVspoof 5 train/dev/eval sets |

## Pułapki

- **Watermark bez uruchomionego detektora.** Bez sensu. Wdróż detektor w swoim CI.
- **Detekcja bez kalibracji.** AASIST trenowany na ASVspoof LA przeucza się; dokładność w świecie rzeczywistym spada. Kalibruj na swojej domenie.
- **Luka pitch-shift.** Agresywny pitch shift usuwa większość watermarków. Miej zastępczy detektor.
- **Strip-and-rehost metadanych.** C2PA jest trywialnie do obejścia przez ponowne kodowanie. Zawsze dodawaj kryptograficzną + percepcyjną (watermark) obronę razem.
- **Żywotność jako detekcja.** Poproś użytkownika o wypowiedzenie losowej frazy. Zapobiega atakom replay, ale nie klonowaniu w czasie rzeczywistym.

## Wyślij To

Zapisz jako `outputs/skill-spoof-defender.md`. Wybierz model detekcji, watermark, manifest proweniencji i playbook operacyjny dla wdrożenia generowania głosu.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Zabawkowy detektor + zabawkowy watermark embed/detect na syntetycznym audio.
2. **Średnie.** Zainstaluj `audioseal`, osadź 16-bitowy ładunek w wyjściu TTS, ponownie zdekoduj. Zniekształć audio szumem i zmierz Bit Recovery Accuracy.
3. **Trudne.** Dostrój RawNet2 lub AASIST na ASVspoof 2019 LA. Zmierz EER. Przetestuj na wstrzymanym zbiorze klipów wygenerowanych przez F5-TTS — zobacz, jak degradacja OOD wpływa na detekcję.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ASVspoof | Benchmark | Coroczne wyzwanie; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detektor | Klasyfikator: prawdziwa mowa vs syntetyczna / konwertowana. |
| SASV | Weryfikacja mówcy + CM | Zintegrowana biometria + wykrywanie spoofingu. |
| AudioSeal | Watermark Meta | Zlokalizowany, 16-bitowy ładunek, 485× szybszy niż WavMark. |
| Bit Recovery Accuracy | Przetrwanie watermarka | Frakcja bitów ładunku odzyskanych po ataku. |
| C2PA | Manifest proweniencji | Kryptograficzne metadane o tworzeniu / autorstwie. |
| AASIST | Rodzina detektorów | SOTA anti-spoofing oparty na graph-attention. |

## Dalsza Lektura

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — obecny benchmark.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — domyślny watermark.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — detektor MoE dla ataków czasowych.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — backbone detekcyjny SOTA.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — ewaluacja solidności.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — format manifestu proweniencji.