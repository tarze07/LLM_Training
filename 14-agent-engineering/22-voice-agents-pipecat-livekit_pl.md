# Agenci Głosowi: Pipecat i LiveKit

> Agenci głosowi to pierwszorzędna kategoria produkcyjna w 2026 roku. Pipecat daje ci ramowy potok w Pythonie (VAD → STT → LLM → TTS → transport). LiveKit Agents łączy modele AI z użytkownikami przez WebRTC. Docelowe opóźnienia produkcyjne lądują na 450–600ms end-to-end dla premium stosów.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 minutes

## Learning Objectives

- Opisz ramowy potok Pipecat: DOWNSTREAM (źródło→ujście) i UPSTREAM (sterowanie).
- Wymień kanoniczne etapy potoku głosowego i które transporty obsługuje Pipecat.
- Wyjaśnij dwie klasy agentów głosowych LiveKit Agents (MultimodalAgent, VoicePipelineAgent) i kiedy każda pasuje.
- Podsumuj oczekiwania dotyczące opóźnień produkcyjnych w 2026 i jak wpływają na wybory architektoniczne.

## Problem

Agenci głosowi to nie pętla tekstowa z doczepionym TTS. Budżety opóźnień są brutalne (~600ms), częściowe audio jest domyślne, wykrywanie tury to model, a transporty obejmują od telefonii SIP do WebRTC. Albo budujesz potok ramowy (Pipecat), albo opierasz się na platformie (LiveKit).

## Koncepcja

### Pipecat (pipecat-ai/pipecat)

- Ramowy framework potokowy w Pythonie.
- Łańcuch `Frame` → `FrameProcessor`.
- Dwa kierunki przepływu:
  - **DOWNSTREAM** — źródło → ujście (audio na wejściu, TTS na wyjściu).
  - **UPSTREAM** — sprzężenie zwrotne i sterowanie (anulowanie, metryki, wtrącenie).
- `PipelineTask` zarządza cyklem życia ze zdarzeniami (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) i obserwatorami dla metryk/śledzenia/RTVI.

Typowy potok:

```
VAD (Silero) → STT → LLM (kontekst zmienia się użytkownik/asystent) → TTS → transport
```

Transporty: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows dodaje strukturalne konwersacje (maszyny stanów). Pipecat Cloud to zarządzane środowisko uruchomieniowe.

### LiveKit Agents (livekit/agents)

- Łączy modele AI z użytkownikami przez WebRTC.
- Kluczowe koncepcje: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Dwie klasy agentów głosowych:
  - **MultimodalAgent** — bezpośrednie audio przez OpenAI Realtime lub odpowiednik.
  - **VoicePipelineAgent** — kaskada STT → LLM → TTS; daje kontrolę na poziomie tekstu.
- Semantyczne wykrywanie tury przez model transformer.
- Natywna integracja MCP.
- Telefonia przez SIP.
- 50+ modeli bez kluczy API przez LiveKit Inference; 200+ więcej przez wtyczki.

### Platformy komercyjne

Vapi (~450–600ms na zoptymalizowanym premium stosie) i Retell (~600ms end-to-end w 180 testowych połączeniach) budują na nich. Wybierz platformę, gdy chcesz zarządzany stos głosowy bez zespołu WebRTC.

### Gdzie ten wzorzec zawodzi

- **Brak obsługi wtrącenia.** Użytkownik przerywa; agent dalej mówi. Wymaga ramek anulowania UPSTREAM w Pipecat, odpowiednika w LiveKit.
- **Ignorowanie pewności STT.** Transkrypty o niskiej pewności podawane do LLM jak ewangelia. Bramkuj na pewności lub żądaj potwierdzenia.
- **Obcięcie TTS w środku zdania.** Gdy potok anuluje w trakcie wypowiedzi, TTS musi wiedzieć lub obciąć audio.
- **Ignorowanie budżetu opóźnienia.** Każdy komponent dodaje 50–200ms. Zsumuj swój łańcuch przed wysłaniem.

### Typowe opóźnienia w 2026

- VAD: 20–60ms
- STT częściowe: 100–250ms
- LLM pierwszy token: 150–400ms
- TTS pierwsze audio: 100–200ms
- Transport RTT: 30–80ms

End-to-end 450–600ms to poziom premium. 800–1200ms jest powszechne. Wszystko > 1500ms wydaje się zepsute.

## Build It

`code/main.py` to ramowy zabawkowy potok z:

- Typami `Frame` (audio, transkrypt, tekst, tts_audio, sterowanie).
- Interfejsem `Processor` z `process(frame)`.
- Pięcioetapowym potokiem (VAD → STT → LLM → TTS → transport) jako skryptowane procesory.
- Ramką anulowania UPSTREAM do demonstracji wtrącenia.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje normalny przepływ i wtrącenie anulujące, które zatrzymuje TTS w środku wypowiedzi.

## Use It

- **Pipecat** dla pełnej kontroli — niestandardowe procesory, Python-first, wymienni dostawcy.
- **LiveKit Agents** dla wdrożeń WebRTC-first i telefonii.
- **Vapi / Retell** dla hostowanych agentów głosowych bez zespołu WebRTC.
- **OpenAI Realtime / Gemini Live** dla bezpośredniego audio-wejście/audio-wyjście (MultimodalAgent).

## Ship It

`outputs/skill-voice-pipeline.md` buduje szkielet potoku głosowego w stylu Pipecat z VAD + STT + LLM + TTS + transport plus obsługą wtrącenia.

## Ćwiczenia

1. Dodaj obserwator metryk do swojego zabawkowego potoku: licz ramki na etap na sekundę. Gdzie kumuluje się opóźnienie?
2. Zaimplementuj STT bramkowane pewnością: poniżej progu, poproś „czy mógłbyś to powtórzyć?"
3. Dodaj semantyczne wykrywanie tury: prosta reguła — jeśli transkrypt kończy się na „?", koniec tury.
4. Przeczytaj dokumentację transportów Pipecat. Zamień transport stdlib na konfigurację SmallWebRTCTransport (atrapa).
5. Zmierz OpenAI Realtime vs kaskadę STT+LLM+TTS na tym samym zapytaniu. Jaki koszt opóźnienia niesie kontrola na poziomie tekstu?

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| Ramka | „Zdarzenie" | Typowana jednostka danych w potoku (audio, transkrypt, tekst, sterowanie) |
| Procesor | „Etap potoku" | Handler z process(frame) |
| DOWNSTREAM | „Przepływ do przodu" | Źródło do ujścia: audio na wejściu, mowa na wyjściu |
| UPSTREAM | „Przepływ zwrotny" | Sterowanie: anulowanie, metryki, wtrącenie |
| VAD | „Detekcja aktywności głosowej" | Wykrywa, kiedy użytkownik mówi |
| Semantyczne wykrywanie tury | „Inteligentny koniec tury" | Decyzja oparta na modelu, że użytkownik skończył |
| MultimodalAgent | „Agent bezpośredniego audio" | Audio na wejściu, audio na wyjściu; brak tekstu pośrodku |
| VoicePipelineAgent | „Agent kaskadowy" | STT + LLM + TTS; kontrola na poziomie tekstu |

## Dalsza lektura

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — potok ramowy, procesory, transporty
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + prymitywy głosowe
- [Vapi](https://vapi.ai/) — zarządzana platforma głosowa
- [Retell AI](https://www.retellai.com/) — zarządzany głos, benchmarkowane opóźnienia