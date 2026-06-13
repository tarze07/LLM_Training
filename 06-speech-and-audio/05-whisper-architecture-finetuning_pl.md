# Whisper — Architektura i Dostrajanie

> Whisper to 30-sekundowy okienkowy transformer encoder-decoder, wytrenowany na 680k godzinach wielojęzycznych, słabo nadzorowanych par audio-tekst. Jedna architektura, wiele zadań, solidny w 99 językach. Referencyjny ASR 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Problem

Whisper, wydany przez OpenAI we wrześniu 2022, był pierwszym modelem ASR dostarczanym jako towar: wklej audio, otrzymaj tekst, 99 języków, odporny na szum, działa na laptopie. Do 2024 OpenAI wydało warianty Large-v3 i Turbo; do 2026 Whisper jest domyślnym baseline'em dla wszystkiego od transkrypcji podcastów po asystentów głosowych i napisy YouTube.

Ale Whisper nie jest pipeline'm, który możesz traktować jako czarną skrzynkę na zawsze. Przesunięcie domeny go zabija — żargon techniczny, akcenty mówców, nazwy własne, krótkie klipy, cisza. Musisz wiedzieć:

1. Czym właściwie jest w środku.
2. Jak podawać mu odpowiednio chunky, strumieniowo lub długie audio.
3. Kiedy dostrajać i jak.

## Koncepcja

![Encoder-dekoder Whispera, zadania, wnioskowanie z chunkami, dostrajanie](../assets/whisper.svg)

**Architektura.** Standardowy transformer encoder-decoder.

- Wejście: 30-sekundowy log-mel spektrogram, 80 mels, 10 ms hop → 3000 ramek. Krótsze klipy są paddowane zerami, dłuższe są chunkowane.
- Enkoder: conv-downsample (krok 2) + `N` bloków transformera. Dla Large-v3: 32 warstwy, 1280-wym, 20 głów.
- Dekoder: `N` bloków transformera z przyczynową self-attn + cross-attn do wyjścia enkodera. Taki sam rozmiar jak enkoder.
- Wyjście: tokeny BPE nad słownikiem 51 865 tokenów.

Large-v3 ma 1,55B parametrów. Turbo używa 4-warstwowego dekodera (z 32), tnąc opóźnienie 8× przy <1% stracie WER.

**Format prompta.** Whisper to model wielozadaniowy sterowany specjalnymi tokenami w prompcie dekodera:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — znacznik języka; wymusza zachowanie translacji-vs-transkrypcji.
- `<|transcribe|>` lub `<|translate|>` — tłumacz na angielski z dowolnego języka lub dosłownie.
- `<|notimestamps|>` — pomiń znaczniki czasu na poziomie słów (szybciej).

Prompt to to, co pozwala jednemu modelowi robić wiele zadań. Zmień `<|en|>` na `<|fr|>`, a transkrybuje francuski.

**30-sekundowe okno.** Wszystko jest przypięte do 30 sekund. Dłuższe klipy wymagają chunkowania; krótsze są paddowane. Okna nie są strumieniowane natywnie — dlatego istnieją WhisperX, Whisper-Streaming i faster-whisper.

**Normalizacja log-mel.** `(log_mel - mean) / std`, gdzie statystyki pochodzą z korpusu treningowego Whispera. *Musisz* użyć przetwarzania wstępnego Whispera (`whisper.audio.log_mel_spectrogram`), nie `librosa.feature.melspectrogram`.

### Warianty w 2026

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Dostrajanie

Kanoniczny przepływ pracy w 2026:

1. Zbierz 10–100 godzin docelowego audio z dopasowanymi transkrypcjami.
2. Uruchom `transformers.Seq2SeqTrainer` z callbackiem `generate_with_loss`.
3. Wydajne parametry: LoRA na `q_proj`, `k_proj`, `v_proj` warstw uwagi redukuje pamięć GPU 4× przy <0,3 koszcie WER.
4. Zamroź enkoder, jeśli masz <10 godzin. Dostrajaj tylko dekoder.
5. Użyj własnego tokenizatora i formatu prompta Whispera; nigdy nie zamieniaj tokenizatorów.

Wyniki społeczności: dostrojenie Medium na 20 godzinach dyktowania medycznego obniża WER z 12% do 4,5% na słownictwie medycznym. Dostrojenie Turbo na 4 godzinach islandzkiego obniża WER z 18% do 6%.

## Zbuduj To

### Krok 1: uruchom Whisper po wyjęciu z pudełka

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # zapobiega powtarzaniu się
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Kluczowe domyślne, które zawsze powinieneś nadpisywać: `temperature=0.0` (próbkowanie domyślnie 0.0 → 0.2 → 0.4 … łańcuch spadkowy), `condition_on_previous_text=False` (zapobiega kaskadowemu problemowi halucynacji) i `no_speech_threshold=0.6` (detekcja ciszy).

### Krok 2: chunkowanie długich form

```python
# whisperx to referencyjne narzędzie 2026 dla długich form ze znacznikami czasu na poziomie słów
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX dodaje (1) Silero VAD, (2) dopasowanie na poziomie słów przez wav2vec 2.0, (3) diaryzację przez `pyannote.audio`. Roboczy koń 2026 do produkcyjnej transkrypcji.

### Krok 3: dostrajanie z LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trenowalne / 809M całkowite
```

Następnie standardowa pętla Trainer. Checkpoint co 1000 kroków. Ewaluuj z WER na wstrzymanym zbiorze.

### Krok 4: sprawdź, czego uczy się każda warstwa

```python
# Zdobądź wagi cross-attention podczas dekodowania, aby zobaczyć, na co dekoder zwraca uwagę.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: warstwa × głowa × krok × długość_src
```

Wizualizuj za pomocą heatmapy — zobaczysz ukośne dopasowanie, gdy kroki dekodera skanują ramki enkodera. Ta przekątna to pojęcie znaczników czasu słów w Whisperze.

## Użyj Tego

Stos 2026:

| Situation | Pick |
|-----------|------|
| Ogólny angielski, offline | Large-v3-turbo przez `whisperx` |
| Mobile / edge | Whisper-Tiny skwantowany (int8) lub Moonshine |
| Wielojęzyczny długi format | Large-v3 przez `whisperx` + diaryzacja |
| Język niskozasobowy | Dostrój Medium lub Turbo z LoRA |
| Strumieniowanie (2 s opóźnienia) | Whisper-Streaming lub Parakeet-TDT |
| Znaczniki czasu na poziomie słów | WhisperX (wymuszone dopasowanie przez wav2vec 2.0) |

`faster-whisper` (backend CTranslate2) to najszybsze środowisko wykonawcze CPU+GPU w 2026 — 4× szybsze niż vanilla z identycznym wyjściem.

## Pułapki, które wciąż występują w 2026

- **Halucynacje tekstu na ciszy.** Whisper trenowany na napisach zawiera "Thanks for watching!", "Subscribe!", teksty piosenek. Zawsze stosuj bramę VAD przed wywołaniem.
- **Kaskada `condition_on_previous_text`.** Jedna halucynacja zanieczyszcza kolejne okna. Ustaw `False`, chyba że potrzebujesz płynności między chunkami.
- **Padding krótkich klipów.** 2-sekundowy klip paddowany do 30 sekund może halucynować w końcowej ciszy. Użyj `pad=False` lub bramy VAD.
- **Złe statystyki mel.** Użycie mels z librosy zamiast z Whispera daje prawie losowe wyjście. Użyj `whisper.audio.log_mel_spectrogram`.

## Wyślij To

Zapisz jako `outputs/skill-whisper-tuner.md`. Zaprojektuj pipeline dostrajania lub wnioskowania Whispera dla danej domeny.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Tokenizuje prompt w stylu Whispera, oblicza budżet kształtu dekodowanego i wypisuje harmonogram chunków dla 10-minutowego klipu.
2. **Średnie.** Zainstaluj `faster-whisper`, transkrybuj 10-minutowy podcast, porównaj WER z ludzkim transkryptem. Wypróbuj `language="auto"` vs wymuszone `language="en"`.
3. **Trudne.** Używając HF `datasets`, wybierz język, z którym Whisper ma problemy (np. urdu), dostrój Medium z LoRA na 2 epoki na 2 godzinach i raportuj różnicę WER.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 30-sec window | Limit Whispera | Twardy limit wejścia; chunkuj dłuższe audio. |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` rozpoczyna prompt dekodera. |
| Timestamps token | Dopasowanie czasowe | Każde 0.02 s przesunięcia to specjalny token w słowniku 51k. |
| Turbo | Szybki wariant | 4 warstwy dekodera, 8× szybszy, <1% regresji WER. |
| WhisperX | Owijka długich form | VAD + Whisper + dopasowanie wav2vec + diaryzacja. |
| LoRA fine-tune | Wydajne dostrajanie | Dodaj adaptery niskiego rzędu do uwagi; trenuj ~0.3% parametrów. |
| Hallucination | Cicha porażka | Whisper produkuje płynny angielski z szumu/ciszy. |

## Dalsza Lektura

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — oryginalna architektura i przepis treningowy.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4-warstwowy dekoder, 8× przyspieszenie.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — długie formy, dopasowanie słów, diaryzacja.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — backend CTranslate2, 4× szybszy.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — kanoniczny przewodnik LoRA / pełnego FT.