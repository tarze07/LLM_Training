# Ewaluacja Audio — WER, MOS, UTMOS, MMAU, FAD i Otwarte Rankingi

> Nie możesz wdrożyć tego, czego nie możesz zmierzyć. Ta lekcja nazywa metryki 2026 dla każdego zadania audio: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), audio-język (MMAU, LongAudioBench), muzyka (FAD, CLAP) i mówca (EER). Plus rankingi, w których porównujesz.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## Problem

Każde zadanie audio ma wiele metryk, każda mierzy inną oś. Użycie złej metryki to sposób, w jaki wdrażasz model, który wygląda świetnie na dashboardzie i fatalnie w produkcji. Kanoniczna lista 2026:

| Zadanie | Podstawowa | Drugorzędna |
|---------|-----------|-------------|
| ASR | WER | CER · RTFx · opóźnienie pierwszego tokena |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Klonowanie głosu | SECS (cosinus ECAPA) | MOS · CER |
| Weryfikacja mówcy | EER | minDCF · FAR / FRR w punkcie operacyjnym |
| Diaryzacja | DER | JER · confusion mówcy |
| Klasyfikacja audio | top-1 · mAP | macro F1 · recall na klasę |
| Generowanie muzyki | FAD | CLAP · MOS panelu słuchaczy |
| Model językowy audio | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Strumieniowa S2S | opóźnienie P50/P95 | WER · MOS |

## Koncepcja

![Macierz ewaluacji audio — metryki vs zadania vs rankingi 2026](../assets/eval-landscape.svg)

### Metryki ASR

**WER (Word Error Rate).** `(S + D + I) / N`. Małe litery, usuń interpunkcję, normalizuj liczby przed oceną. Użyj `jiwer` lub `whisper_normalizer` OpenAI. &lt; 5% = poziom ludzki dla czytanej mowy.

**CER (Character Error Rate).** Ten sam wzór, na poziomie znaków. Używany dla języków tonalnych (mandaryński, kantoński), gdzie segmentacja słów jest niejednoznaczna.

**RTFx (odwrotny współczynnik czasu rzeczywistego).** Sekundy audio przetworzone na sekundę ścienną. Wyższe znaczy lepiej. Parakeet-TDT osiąga 3380×. Whisper-large-v3 to ~30×.

**Opóźnienie pierwszego tokena.** Czas ścienny od wejścia audio do pierwszego tokena transkryptu. Kluczowe dla strumieniowania. Deepgram Nova-3: ~150 ms.

### Metryki TTS

**MOS (Mean Opinion Score).** Ocena ludzka 1-5. Złoty standard, ale powolny. Zbierz 20+ słuchaczy na próbkę, 100+ próbek na model.

**UTMOS (2022-2026).** Uczony predyktor MOS. Koreluje ~0.9 z ludzkim MOS na standardowych benchmarkach. F5-TTS: UTMOS 3.95; ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** Dla klonowania głosu. Cosinus osadzenia ECAPA między referencją a sklonowanym wyjściem. &gt; 0.75 = rozpoznawalny klon.

**WER-on-ASR-round-trip.** Przepuść wyjście TTS przez Whisper, oblicz WER względem tekstu wejściowego. Wykrywa regresje zrozumiałości. SOTA 2026: &lt; 2% CER.

**TTFA (time-to-first-audio).** Opóźnienie ścienne. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

### Specyficzne dla klonowania głosu

**SECS + MOS + CER** jako trójka. Klonowanie z wysokim SECS ale niskim MOS oznacza barwa-dobra-ale-nienaturalna; odwrotnie oznacza naturalny głos, ale zły mówca.

### Weryfikacja mówcy

**EER (Equal Error Rate).** Próg, gdzie False Accept Rate równa się False Reject Rate. ECAPA na VoxCeleb1-O: 0.87%.

**minDCF (min Detection Cost).** Ważony koszt w wybranym punkcie operacyjnym (często FAR=0.01). Bardziej istotny produkcyjnie niż EER.

### Diaryzacja

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. Brakująca mowa + fałszywy alarm + confusion mówcy, każdy jako ułamek. Spotkania AMI: DER ~10-20% jest realistyczne. pyannote 3.1 + Precision-2 komercyjny: &lt;10% DER na dobrze nagranym audio.

**JER (Jaccard Error Rate).** Alternatywa dla DER, odporna na błąd krótkich segmentów.

### Klasyfikacja audio

Multi-label: **mAP (mean Average Precision)** dla wszystkich klas. AudioSet: 0.548 mAP dla BEATs-iter3.

Multi-class wyłączny: **top-1, top-5 accuracy**. Speech Commands v2: 99.0% top-1 (Audio-MAE).

Niezbalansowane: **macro F1** + **recall na klasę**. Raportuj na klasę — zagregowana dokładność ukrywa, które klasy zawodzą.

### Generowanie muzyki

**FAD (Fréchet Audio Distance).** Odległość między rozkładami osadzeń VGGish prawdziwego a wygenerowanego audio. MusicGen-small na MusicCaps: 4.5. MusicLM: 4.0. Niższe znaczy lepiej.

**CLAP Score.** Wynik dopasowania tekst-audio używając osadzeń CLAP. &gt; 0.3 = rozsądne dopasowanie.

**MOS panelu słuchaczy.** Wciąż ostatnie słowo dla muzyki konsumenckiej. Suno v5 ELO 1293 na TTS Arena (z sparowanych preferencji ludzkich).

### Benchmarki audio-językowe

**MMAU (Massive Multi-Audio Understanding).** 10k par audio-QA.

**MMAU-Pro.** 1800 trudnych elementów, cztery kategorie: mowa / dźwięk / muzyka / wieloaudio. Losowy przypadek 25% na 4-opcyjnym. Gemini 2.5 Pro ogólnie ~60%; wieloaudio ~22% dla wszystkich modeli.

**LongAudioBench.** Wielominutowe klipy z zapytaniami semantycznymi. Audio Flamingo Next bije Gemini 2.5 Pro.

**AudioCaps / Clotho.** Benchmarki captioningu. Metryki SPICE, CIDEr, FENSE.

### Strumieniowa mowa-na-mowę

**Opóźnienie P50 / P95 / P99.** Czas ścienny od końca mowy użytkownika do pierwszej słyszalnej odpowiedzi. Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**WER / MOS** na wyjściu.

**Responsywność na wtargnięcie.** Czas od przerwania przez użytkownika do wyciszenia asystenta. Cel &lt; 150 ms.

### Rankingi 2026

| Ranking | Śledzi | URL |
|---------|--------|-----|
| Open ASR Leaderboard (HF) | Angielski + wielojęzyczny + długie formy | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | Angielski TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO z głosów parowanych | `artificialanalysis.ai/speech` |
| MMAU-Pro | Rozumowanie LALM | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Rozpoznawanie mówcy | `voxsrc.github.io` |
| MMAU music subset | Muzyczny LALM | (w ramach MMAU) |
| HEAR benchmark | Samonadzorowane audio | `hearbenchmark.com` |

## Zbuduj To

### Krok 1: WER z normalizacją

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Krok 2: TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Krok 3: SECS dla klonowania głosu

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Krok 4: FAD dla generowania muzyki

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Krok 5: EER dla weryfikacji mówcy (ten sam kod co Lekcja 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Użyj Tego

Paruj każde wdrożenie ze stałym harnessem ewaluacyjnym uruchamianym przy każdej aktualizacji modelu. Trzy kardynalne zasady:

1. **Normalizuj przed oceną.** Małe litery, usuń interpunkcję, rozwiń liczby. Raportuj regułę normalizacji.
2. **Raportuj rozkłady, nie średnie.** P50/P95/P99 dla opóźnienia. Recall na klasę dla klasyfikacji. Na kategorię dla MMAU.
3. **Uruchom jeden kanoniczny publiczny benchmark.** Nawet jeśli twoje dane produkcyjne się różnią, raportowanie na Open ASR / TTS Arena / MMAU pozwala recenzentom porównywać.

## Pułapki

- **Ekstrapolacja UTMOS.** Trenowany na czystej mowie w stylu VCTK; słabo ocenia zaszumione / klonowane / emocjonalne audio.
- **Bias panelu MOS.** 20 pracowników Amazon Mechanical Turk ≠ 20 docelowych użytkowników. Zapłać za panel domenowy, jeśli stawka jest wysoka.
- **FAD zależy od zbioru referencyjnego.** Porównuj względem tego samego rozkładu referencyjnego między modelami.
- **Zagregowany WER.** 5% WER ogólnie może ukrywać 30% WER na mowie z akcentem. Raportuj według segmentu demograficznego.
- **Nasycenie publicznych benchmarków.** Większość modeli granicznych jest blisko sufitu na standardowych benchmarkach. Zbuduj wewnętrzny wstrzymany zestaw odzwierciedlający twój ruch.

## Wyślij To

Zapisz jako `outputs/skill-audio-evaluator.md`. Wybierz metryki, benchmarki i format raportowania dla dowolnego wydania modelu audio.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Oblicz WER / CER / EER / SECS / FAD-ish / MMAU-ish na zabawkowych wejściach.
2. **Średnie.** Zbuduj harness TTS round-trip WER. Przepuść wyjście Kokoro lub F5-TTS przez Whisper. Oblicz WER na 50 promptach. Oznacz prompty z WER &gt; 10%.
3. **Trudne.** Oceń swój wybór LALM z Lekcji 10 na podzbiorach mowy + wieloaudio MMAU-Pro (50 elementów każdy). Raportuj dokładność na kategorię i porównaj z opublikowaną liczbą.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| WER | Wynik ASR | `(S+D+I)/N` na poziomie słów po normalizacji. |
| CER | WER znaków | Dla języków tonalnych lub systemów na poziomie znaków. |
| MOS | Opinia ludzka | Ocena 1-5; 20+ słuchaczy × 100 próbek. |
| UTMOS | Predyktor MOS ML | Uczony model; koreluje ~0.9 z ludzkim MOS. |
| SECS | Podobieństwo klonu | Cosinus ECAPA między referencją a klonem. |
| EER | Wynik weryfikacji mówcy | Próg, gdzie FAR = FRR. |
| DER | Wynik diaryzacji | (FA + Miss + Confusion) / całkowity. |
| FAD | Jakość generowania muzyki | Odległość Frécheta na osadzeniach VGGish. |
| RTFx | Przepustowość | Sekundy audio na sekundę ścienną. |

## Dalsza Lektura

- [jiwer](https://github.com/jitsi/jiwer) — biblioteka WER/CER z narzędziami normalizacji.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — uczony predyktor MOS.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — standard generowania muzyki.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — żywe rankingi 2026.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — ranking TTS z głosami ludzkimi.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — ranking rozumowania LALM.
- [HEAR benchmark](https://hearbenchmark.com/) — benchmarki SSL audio.