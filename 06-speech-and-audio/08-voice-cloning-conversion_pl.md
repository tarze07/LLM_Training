# Klonowanie Głosu i Konwersja Głosu

> Klonowanie głosu czyta twój tekst głosem kogoś innego. Konwersja głosu przepisuje twój głos na kogoś innego, zachowując to, co powiedziałeś. Oba opierają się na tym samym rozkładzie: oddziel tożsamość mówcy od treści.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Problem

W 2026 5-sekundowy klip audio wystarczy, aby wyprodukować wysokiej jakości klon czyjegoś głosu na konsumenckim GPU. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox wszystkie dostarczają zerokrotne lub kilkukrotne klonowanie. Technologia jest błogosławieństwem (TTS dostępności, dubbing, głosy wspomagające) i bronią (oszukańcze rozmowy, polityczne deepfake'i, kradzież własności intelektualnej).

Dwa blisko powiązane zadania:

- **Klonowanie głosu (strona TTS):** tekst + 5-sekundowy referencyjny głos → audio w tym głosie.
- **Konwersja głosu (strona mowy):** źródłowe audio (osoba A mówiąca X) + referencyjny głos osoby B → audio osoby B mówiącej X.

Oba faktoryzują przebieg na (treść, mówca, prozodia) i łączą treść z jednego źródła z mówcą z innego.

Kluczowe ograniczenie, z którym dostarczasz w 2026: **watermarkowanie i bramy zgody są prawnie wymagane w UE (AI Act, obowiązujący od sierpnia 2026) i w Kalifornii (AB 2905, obowiązujący od 2025)**. Twój pipeline musi emitować niesłyszalny watermark i odrzucać klony bez zgody.

## Koncepcja

![Klonowanie vs konwersja głosu: faktoryzuj, zamień mówcę, połącz](../assets/voice-cloning.svg)

**Zerokrotne klonowanie.** Przekaż 5-sekundowy klip do modelu wytrenowanego na tysiącach mówców. Enkoder mówcy mapuje klip na osadzenie mówcy; dekoder TTS warunkuje na tym osadzeniu plus tekście.

Używane przez: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Kilkupróbkowe dostrajanie.** Nagraj 5-30 minut docelowego głosu. Dostrój LoRA na modelu bazowym przez godzinę. Jakość skacze z "okej" na "nieodróżnialny". Coqui i ElevenLabs obsługują ten wzór; społeczność używa go z F5-TTS.

**Konwersja głosu (VC).** Dwie rodziny:

- **Rozpoznanie-synteza.** Uruchom model podobny do ASR, aby wydobyć reprezentację treści (np. miękkie posterogramy fonemów, PPG), następnie resyntezuj z docelowym osadzeniem mówcy. Odporny na język i akcent. Używane przez KNN-VC (2023), Diff-HierVC (2023).
- **Rozplątanie.** Trenuj autoenkoder, który oddziela treść, mówcę i prozodię w przestrzeni ukrytej. Zamień osadzenie mówcy przy wnioskowaniu. Niższa jakość, ale szybsze. Używane przez AutoVC (2019), warianty VITS-VC.

**Klonowanie oparte na kodowaniu neuralnym (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — traktuj audio jako dyskretne tokeny z SoundStream / EnCodec, trenuj duży autoregresyjny lub flow-matching model nad tokenami kodeka. Jakość porównywalna z ElevenLabs na krótkich promptach.

### Etyka, nie dodatek

**Watermarkowanie.** PerTh (Perth) i SilentCipher (2024) osadzają ~16-32 bitowe ID niewidocznie w audio. Przetrwa ponowne kodowanie, strumieniowanie i typowe edycje. Gotowe do produkcji open source.

**Bramy zgody.** Muszą parować każde sklonowane wyjście z weryfikowalnym rekordem zgody. "Ja, Rohit, w dniu 2026-04-22, autoryzuję ten głos do celu X." Przechowuj w dzienniku zabezpieczonym przed manipulacją.

**Detekcja.** AASIST, RawNet2 i Wav2Vec2-AASIST działają jako detektory. Wyzwanie ASVspoof 2025 opublikowało EER 0,8–2,3% dla najnowocześniejszych detektorów przeciwko wyjściom ElevenLabs, VALL-E 2 i Bark.

### Liczby (2026)

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Tak | 0.72 | 2.1% | 335M |
| XTTS v2 | Tak | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Tak | 0.70 | 2.8% | 220M |
| VALL-E 2 | Tak | 0.77 | 2.4% | 370M |
| VoiceBox | Tak | 0.78 | 2.1% | 330M |

SECS > 0.70 jest ogólnie nieodróżnialny od celu dla większości słuchaczy.

## Zbuduj To

### Krok 1: rozkład przez rozpoznanie-syntezę (demo tylko kod w main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Koncepcyjnie proste; masa implementacji jest w `tts_model` i enkoderze mówcy.

### Krok 2: zerokrotne klonowanie z F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Transkrypcja referencyjna musi dokładnie pasować do audio; niedopasowanie psuje dopasowanie.

### Krok 3: konwersja głosu z KNN-VC

```python
import torch
from knnvc import KNNVC  # model 2023, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC uruchamia WavLM, aby wydobyć osadzenia na ramkę dla źródła i puli docelowej, a następnie zastępuje każdą ramkę źródłową jej najbliższym sąsiadem w puli. Nieparametryczny, działa z minutą docelowej mowy.

### Krok 4: osadź watermark

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # zwraca bajty ładunku
```

~32 bity ładunku, wykrywalne po ponownym kodowaniu MP3 i lekkim szumie.

### Krok 5: brama zgody

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Wymagana podpisana zgoda"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Użyj Tego

Stos 2026:

| Situation | Pick |
|-----------|------|
| 5-sekundowe zerokrotne klonowanie, open-source | F5-TTS lub OpenVoice v2 |
| Komercyjne klonowanie produkcyjne | ElevenLabs Instant Voice Clone v2.5 |
| Konwersja głosu (przepisywanie) | KNN-VC lub Diff-HierVC |
| Dostrajanie wielu mówców | StyleTTS 2 + adapter mówcy |
| Klonowanie międzyjęzykowe | XTTS v2 lub VALL-E X |
| Wykrywanie deepfake | Wav2Vec2-AASIST |

## Pułapki

- **Niedopasowana referencyjna transkrypcja.** F5-TTS i podobne wymagają, aby referencyjny tekst dokładnie pasował do referencyjnego audio, łącznie z interpunkcją.
- **Pogłosowa referencja.** Echo zabija klon. Nagrywaj sucho, z bliska.
- **Niedopasowanie emocjonalne.** Referencja treningowa "radosna" produkuje radosne klony wszystkiego. Dopasuj emocje referencji do docelowego użycia.
- **Wyciek języka.** Klonowanie anglojęzycznego mówcy, a następnie proszenie modelu o mówienie po francusku często przenosi akcent; używaj modeli międzyjęzykowych (XTTS, VALL-E X).
- **Brak watermarku.** Prawnie niewdrowalne w UE od sierpnia 2026.

## Wyślij To

Zapisz jako `outputs/skill-voice-cloner.md`. Zaprojektuj pipeline klonowania lub konwersji z bramą zgody + watermarkiem + celem jakości.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Demonstruje zamianę osadzenia mówcy, obliczając cosinus między dwoma "mówcami" przed i po zamianie.
2. **Średnie.** Użyj OpenVoice v2, aby sklonować własny głos. Zmierz SECS między referencją a klonem. Zmierz CER przez Whisper.
3. **Trudne.** Zastosuj watermark SilentCipher do 20 klonów, przepuść przez kodowanie+dekodowanie MP3 128 kbps, wykryj ładunek. Raportuj dokładność bitową.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 sekund wystarczy | Pretrenowany model + osadzenie mówcy; brak treningu. |
| PPG | Fonetyczny posterogram | Posterogramy ASR na ramkę używane jako reprezentacja treści niezależna od języka. |
| KNN-VC | Konwersja najbliższego sąsiada | Zastąp każdą ramkę źródłową najbliższą ramką z puli docelowej. |
| Neural codec TTS | Styl VALL-E | Model AR nad tokenami EnCodec/SoundStream. |
| Watermark | Niesłyszalny podpis | Bity osadzone w audio, przetrwają ponowne kodowanie. |
| SECS | Wierność klonowania | Cosinus między osadzeniami mówcy docelowego i klonu. |
| AASIST | Detektor deepfake | Model anty-spoofingowy; wykrywa syntetyczną mowę. |

## Dalsza Lektura

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — open-source SOTA zerokrotnego klonowania.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) i [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — konwersja głosu oparta na rozplątaniu.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — VC oparty na wyszukiwaniu.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — gotowy do produkcji 32-bitowy watermark audio.
- [ASVspoof 2025 results](https://www.asvspoof.org/) — wyścig zbrojeń detektor vs syntezator, aktualizowane 2026.