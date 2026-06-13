# Strumieniowa Mowa-na-Mowę — Moshi, Hibiki i Dialog Full-Duplex

> Lata 2024-2026 przedefiniowały AI głosową. Moshi dostarcza pojedynczy model, który słucha i mówi jednocześnie przy 200 ms opóźnienia. Hibiki robi tłumaczenie mowy-na-mowę fragment po fragmencie. Oba porzucają pipeline ASR → LLM → TTS na rzecz ujednoliconej architektury full-duplex nad tokenami kodeka Mimi. To nowy projekt referencyjny.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Problem

Każdy agent głosowy zbudowany z Lekcji 11 + 12 ma fundamentalne minimalne opóźnienie około 300-500 ms: VAD się uruchamia, STT przetwarza, LLM rozumuje, TTS generuje. Każdy etap ma własne minimalne opóźnienie. Możesz dostrajać i parallelizować, ale kształt pipeline'u cię ogranicza.

Moshi (Kyutai, 2024-2026) zadaje inne pytanie: co jeśli nie ma pipeline'u? Co jeśli jeden model pobiera audio i emituje audio bezpośrednio, ciągle, z tekstem jako pośrednim "wewnętrznym monologiem" zamiast wymaganego etapu?

Odpowiedzią jest **full-duplex mowa-na-mowę**. Teoretyczne opóźnienie 160 ms (80 ms ramka Mimi + 80 ms opóźnienie akustyczne). Praktyczne opóźnienie 200 ms na pojedynczym GPU L4. To połowa tego, co osiąga najlepszy w swojej klasie agent głosowy w pipeline'ie.

## Koncepcja

![Architektura Moshi: dwa równoległe strumienie Mimi + tekst wewnętrznego monologu](../assets/moshi-hibiki.svg)

### Architektura Moshi

**Wejścia.** Dwa strumienie kodeka Mimi, oba przy 12.5 Hz × 8 codebooków:

- Strumień 1: audio użytkownika (Mimi-zakodowane, stale napływające)
- Strumień 2: własne audio Moshi (wygenerowane przez Moshi)

**Transformer.** 7B-parametrowy Temporal Transformer przetwarza oba strumienie i tekstowy strumień "wewnętrznego monologu". W każdym 80 ms kroku:

1. Konsumuje najnowsze tokeny Mimi użytkownika (8 codebooków).
2. Konsumuje najnowsze tokeny Mimi Moshi (8 codebooków, wyprodukowane).
3. Generuje następny token tekstu Moshi (wewnętrzny monolog).
4. Generuje następne tokeny Mimi Moshi (8 codebooków przez mały Depth Transformer).

Wszystkie trzy strumienie — audio użytkownika, audio Moshi, tekst Moshi — działają równolegle. Moshi może słyszeć użytkownika podczas mówienia; może przerwać siebie, gdy użytkownik przerywa; może dawać sygnały zwrotne ("mhm") bez przerywania głównej wypowiedzi.

**Depth transformer.** W ramach jednej ramki, 8 codebooków nie jest przewidywanych równolegle — mają między-sobowe zależności. Mały 2-warstwowy "depth transformer" przewiduje je sekwencyjnie w ciągu 80 ms. To standardowa faktoryzacja dla AR codec LM (używana również przez VALL-E, VibeVoice).

### Dlaczego tekst wewnętrznego monologu pomaga

Bez jawnego tekstu model musi niejawnie modelować język w swoim strumieniu akustycznym. Wgląd Moshi: wymuś emitowanie tokenów tekstu obok audio. Strumień tekstu to zasadniczo transkrypt tego, co Moshi mówi. Poprawia spójność semantyczną, ułatwia wymianę głowy modelu językowego i daje transkrypty za darmo.

### Hibiki: strumieniowe tłumaczenie mowa-na-mowę

Ta sama architektura, trenowana na parach tłumaczeniowych. Źródłowe audio na wejściu, docelowe audio na wyjściu, ciągle. Hibiki-Zero (luty 2026) eliminuje potrzebę danych treningowych z dopasowaniem na poziomie słów — używa danych na poziomie zdań + uczenia przez wzmacnianie GRPO do optymalizacji opóźnienia.

Cztery pary językowe początkowo obsługiwane; można dostosować do nowego języka z ≈1000 godzin.

### Szerszy stos Kyutai (2026)

- **Moshi** — dialog full-duplex (francuski pierwszy, angielski dobrze obsługiwany)
- **Hibiki / Hibiki-Zero** — symultaniczne tłumaczenie mowy
- **Kyutai STT** — strumieniowy ASR (500 ms lub 2.5 s wyprzedzenia)
- **Kyutai Pocket TTS** — 100M-parametrowy TTS działający na CPU (sty 2026)
- **Unmute** — pełny pipeline łączący je na publicznych serwerach

Przepustowość na GPU L40S: 64 równoczesne sesje przy 3× czasu rzeczywistego.

### Sesame CSM — kuzyn

Sesame CSM (2025) używa podobnego pomysłu — backbonu Llama-3 z głową kodeka Mimi. Ale CSM jest jednokierunkowy (pobiera kontekst + tekst, produkuje mowę), a nie full-duplex. To najlepszy TTS z "obecnością głosu" na rynku; nie do końca to samo, co możliwość full-duplex Moshi.

### Liczby wydajności 2026

| Model | Opóźnienie | Zastosowanie | Licencja |
|-------|-----------|-------------|----------|
| Moshi | 200 ms (L4) | full-duplex dialog angielski / francuski | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | Strumieniowe tłumaczenie francuski ↔ angielski | CC-BY 4.0 |
| Hibiki-Zero | to samo | 5 par językowych, brak dopasowanych danych | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | TTS warunkowany kontekstem | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | zamknięty, OpenAI API | komercyjna |
| Gemini 2.5 Live | ~350 ms | zamknięty, Google API | komercyjna |

## Zbuduj To

### Krok 1: interfejs

Moshi udostępnia serwer WebSocket, który pobiera 80 ms fragmenty Mimi-zakodowanego audio i zwraca 80 ms fragmenty Mimi-zakodowanego audio. W obie strony. Ciągle.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Krok 2: pętla full-duplex

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Obie kierunki działają jednocześnie. Python asyncio lub Rust futures to standardowy transport.

### Krok 3: cel treningowy (koncepcyjnie)

Dla każdej 80 ms ramki `t`:

- Wejście: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Przewidź: `moshi_text[t]`, następnie `moshi_mimi[t, codebook_0..7]`

Tekst jest przewidywany przed audio (wewnętrzny monolog); audio jest przewidywane sekwencyjnie po codebookach w depth transformerze.

### Krok 4: gdzie Moshi wygrywa, a gdzie nie

Moshi wygrywa:

- Poniżej 250 ms end-to-end na tanim sprzęcie.
- Naturalne sygnały zwrotne i przerywanie.
- Brak kodu klejącego pipeline'u.

Moshi nie wygrywa:

- Wywoływanie narzędzi (nie trenowane do tego; potrzebujesz osobnej ścieżki LLM).
- Długie rozumowanie (Moshi to model dialogowy ~8B, nie Claude/GPT-4).
- Dokładność faktograficzna w niszowych tematach.
- Większość przypadków użycia w przedsiębiorstwach (wciąż używają pipeline'ów w 2026).

## Użyj Tego

| Sytuacja | Wybór |
|----------|-------|
| Towarzysz głosowy o najniższym opóźnieniu | Moshi |
| Rozmowa z tłumaczeniem na żywo | Hibiki |
| Demo głosowe / badania | Moshi, CSM |
| Agent przedsiębiorstwa z narzędziami | Pipeline (Lekcja 12), nie Moshi |
| TTS z niestandardowym głosem w kontekście | Sesame CSM |
| Mowa-na-mowę, dowolne języki | GPT-4o Realtime lub Gemini 2.5 Live (komercyjne) |

## Pułapki

- **Ograniczone wywoływanie narzędzi.** Moshi to model dialogowy, nie framework agenta. Połącz z pipeline'm dla narzędzi.
- **Warunkowanie konkretnym głosem.** Moshi używa pojedynczej trenowanej persony; klonowanie to osobne uruchomienie treningu.
- **Pokrycie językowe.** Francuski + angielski jest doskonałe; inne ograniczone. Hibiki-Zero pomaga, ale wciąż potrzebujesz danych treningowych.
- **Koszt zasobów.** Pełna sesja Moshi trzyma slot GPU; to nie tani wzorzec wdrożenia współdzielonego.

## Wyślij To

Zapisz jako `outputs/skill-duplex-pipeline.md`. Wybierz architekturę pipeline vs full-duplex dla obciążenia agenta głosowego, z uzasadnieniem.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Symuluje architekturę dwustrumieniową + wewnętrzny monolog symbolicznie.
2. **Średnie.** Pobierz Moshi z HuggingFace, uruchom serwer, przetestuj jedną konwersację. Zmierz opóźnienie ścienne od końca mowy użytkownika do początku odpowiedzi Moshi.
3. **Trudne.** Weź swojego agenta pipeline'owego z Lekcji 12 i porównaj opóźnienie P50 vs Moshi na 20 dopasowanych wypowiedziach testowych. Opisz, kiedy pipeline wygrywa architektonicznie.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Full-duplex | Słysz-i-mów jednocześnie | Dwa strumienie audio aktywne jednocześnie w tym samym modelu. |
| Inner monologue | Strumień tekstu modelu | Moshi emituje tokeny tekstu obok swojego wyjścia audio. |
| Depth transformer | Predyktor między-codebookowy | Mały transformer przewidujący 8 codebooków w jednej 80 ms ramce. |
| Mimi | Kodek Kyutai | 12.5 Hz × 8 codebooków; semantyczny+akustyczny; napędza Moshi. |
| Streaming S2S | Audio → audio na żywo | Tłumaczenie/dialog fragment po fragmencie, bez etapów pipeline'u. |
| Back-channeling | Reakcje "Mhm" | Moshi może emitować małe potwierdzenia bez przerywania swojej tury. |

## Dalsza Lektura

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — artykuł.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — strumieniowe tłumaczenie bez dopasowanych danych.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — specyfikacja CSM.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — instalacja + serwer.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — zamknięty komercyjny odpowiednik.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — framework STT/TTS pod maską.