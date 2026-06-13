# Modele Językowe Audio — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> Modele językowe audio 2026 rozumują nad mową + dźwiękami otoczenia + muzyką. Qwen2.5-Omni-7B dorównuje GPT-4o Audio na MMAU-Pro. Audio Flamingo Next bije Gemini 2.5 Pro na LongAudioBench. Luka między otwartym a zamkniętym jest praktycznie zamknięta — z wyjątkiem zadań wieloaudio, gdzie wszyscy są blisko losowego.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## Problem

Masz 5 sekund audio: pies szczeka, ktoś krzyczy "stop!", potem cisza. Przydatne pytania obejmują wiele osi:

- **Transkrypcja.** "Co zostało powiedziane?" — terytorium ASR.
- **Rozumowanie semantyczne.** "Czy osoba jest w niebezpieczeństwie?" — wymaga wspólnego zrozumienia szczekania + krzyku + ciszy.
- **Rozumowanie muzyczne.** "Jakie instrumenty grają melodię?"
- **Wyszukiwanie w długim audio.** "Gdzie w tym 90-minutowym wykładzie instruktor wyjaśnił gradient descent?"

Pojedynczy model odpowiadający na wszystkie te pytania jednym promptem to **model językowy audio** (LALM / ALM). Oddzielny od czystego ASR: LALMy produkują swobodne odpowiedzi w języku naturalnym, nie tylko transkrypty.

## Koncepcja

![Model językowy audio: enkoder audio + projektor + dekoder LLM](../assets/alm-architecture.svg)

### Trzykomponentowy szablon

Każdy LALM 2026 ma ten sam szkielet:

1. **Enkoder audio.** Enkoder Whispera · BEATs · CLAP · WavLM · lub niestandardowy enkoder na model.
2. **Projektor.** Liniowy lub MLP łączący cechy enkodera audio z przestrzenią osadzeń tokenów LLM.
3. **LLM.** Dekoder oparty na Llamie / Qwen / Gemmie. Przyjmuje przeplatane tokeny tekst + audio; generuje tekst.

Trening:

- **Etap 1.** Zamroź enkoder + LLM; trenuj tylko projektor na danych ASR / captioning.
- **Etap 2.** Pełne / LoRA dostrajanie na zadaniach audio z instrukcjami (QA, rozumowanie, rozumienie muzyki).
- **Etap 3 (opcjonalny).** Głos-wejście / głos-wyjście dodaje dekoder mowy. Qwen2.5-Omni i AF3-Chat to robią.

### Mapa modeli 2026

| Model | Backbone | Enkoder audio | Modalność wyjścia | Dostęp |
|-------|----------|---------------|-------------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Niestandardowy + Whisper | tekst + mowa | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Niestandardowy | tekst + mowa | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | tekst | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | tekst | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | tekst | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | tekst | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | tekst | Apache-2.0 |
| Gemini 2.5 Flash/Pro (zamknięty) | Gemini | własnościowy | tekst + mowa | API |
| GPT-4o Audio (zamknięty) | GPT-4o | własnościowy | tekst + mowa | API |

### Sprawdzenie rzeczywistości benchmarków (2026)

**MMAU-Pro.** 1800 par QA obejmujących mowę / dźwięk / muzykę / mieszane. Podzbiór wieloaudio włączony.

| Model | Ogólny | Mowa | Dźwięk | Muzyka | Wieloaudio |
|-------|--------|------|--------|--------|------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA na LongAudioBench | — | — | — | — |

**Kolumna wieloaudio jest druzgocąca dla wszystkich.** Losowy przypadek w 4-opcyjnym wyborze wielokrotnym = 25%; większość modeli osiąga około tyle. LALMy wciąż mają problem z porównywaniem dwóch klipów.

### Gdzie LALMy są przydatne w 2026

- **Audyt zgodności nagrań call-center.** "Czy agent wspomniał o wymaganym ujawnieniu?"
- **Dostępność.** Opisz zdarzenia dźwiękowe dla niesłyszących użytkowników (nie tylko transkrypcja).
- **Moderacja treści.** Wykrywaj agresywny język + groźny ton + kontekst tła.
- **Rozdział podcastów / spotkań.** Semantyczne podsumowanie, nie tylko tury mówców.
- **Analiza katalogu muzycznego.** "Znajdź wszystkie utwory ze zmianą tonacji w sekcji B."

### Gdzie NIE są (jeszcze) przydatne

- Szczegółowa teoria muzyki (poniżej poziomu akordów).
- Rozumowanie z przypisaniem mówcy w długich rozmowach (pogarsza się po 10 minutach).
- Porównanie wieloaudio (22-26% to ledwie powyżej losowego).
- Rozumowanie w czasie rzeczywistym w strumieniu (większość to offline batch inference).

## Zbuduj To

### Krok 1: zapytaj Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Krok 2: wzór projektora

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

To wszystko. Projektor to zwykle 1-3 warstwy liniowe. Trenowanie go na parach ASR (audio → transkrypt) to zadanie pretekstowe Etapu 1.

### Krok 3: benchmarkowanie MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Raportuj per-kategorię (mowa / dźwięk / muzyka / wieloaudio) osobno. Zagregowane liczby ukrywają, gdzie model zawodzi.

## Użyj Tego

| Zadanie | Wybór 2026 |
|---------|------------|
| Swobodne QA audio (otwarte) | Qwen2.5-Omni-7B |
| Najlepsze otwarte na długim audio | Audio Flamingo Next |
| Najlepsze zamknięte | Gemini 2.5 Pro |
| Agent głos-wejście / głos-wyjście | Qwen2.5-Omni lub GPT-4o Audio |
| Rozumowanie muzyczne | Audio Flamingo 3 lub 2 (muzycznie wyspecjalizowany AF-CLAP) |
| Audyt call-center | Gemini 2.5 Pro przez API, z RAG nad dokumentami polityki |

## Pułapki

- **Zbytnie zaufanie do wieloaudio.** Jeśli twoje zadanie wymaga "który klip ma X," wydajność na poziomie losowego przypadku jest realna.
- **Degradacja długiego audio.** Po 10 minutach przypisywanie mówcy w większości modeli załamuje się. Najpierw zrób diaryzację (Lekcja 6), potem podsumuj.
- **Halucynacje na ciszy.** Ten sam problem w stylu Whispera dziedziczony przez LALMy używające enkodera Whispera. Zastosuj bramę VAD.
- **Cherry-pickowanie benchmarków.** Wpis na blogu dostawcy podkreśla najlepsze kategorie. Uruchom sam podzbiór wieloaudio MMAU-Pro.

## Wyślij To

Zapisz jako `outputs/skill-alm-picker.md`. Wybierz LALM + podzbiór benchmarku + modalność wyjścia (tekst vs mowa) dla danego zadania rozumienia audio.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`, aby zobaczyć zabawkowy wzór projektora + fałszywe routowanie LALM (osadzenie-audio, tokeny-tekstu) → tokeny wyjściowe.
2. **Średnie.** Oceń Qwen2.5-Omni-7B na 100 elementach mowy MMAU-Pro. Porównaj z opublikowaną liczbą.
3. **Trudne.** Zbuduj minimalny baseline audio-captioning: enkoder BEATs + 2-warstwowy projektor + zamrożony Llama-3.2-1B. Dostrój tylko projektor na AudioCaps. Porównaj z SALMONN na Clotho-AQA.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LALM | ChatGPT audio | Enkoder audio + projektor + dekoder LLM. |
| Projector | Adapter | Mały MLP mapujący cechy audio do przestrzeni osadzeń LLM. |
| MMAU | Benchmark | 10k par audio-QA przez mowę, dźwięk, muzykę. |
| MMAU-Pro | Trudniejsze MMAU | 1800 pytań wieloaudio / wymagających rozumowania. |
| LongAudioBench | Ewaluacja długich form | Wielominutowe klipy z zapytaniami semantycznymi. |
| Voice-in / voice-out | Mowa natywna | Model pobiera mowę i emituje mowę bez pośrednictwa tekstu. |

## Dalsza Lektura

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — referencyjna architektura.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — mowa-wejście-mowa-wyjście.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — otwarty lider długiego audio.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — SOTA na LongAudioBench.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — pionier podwójnego enkodera.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — żywe rankingi 2026.