# Capstone 03 — Asystent Głosowy w Czasie Rzeczywistym (ASR do LLM do TTS)

> Agent głosowy, który działa dobrze, ma opóźnienie poniżej 800ms, wie, kiedy przestałeś mówić, obsługuje wtargnięcie w mowę i może wywołać narzędzie bez zacięcia. Retell, Vapi, LiveKit Agents i Pipecat osiągają ten poziom w 2026 roku. Robią to w tym samym kształcie: strumieniowy ASR, detektor zakończenia wypowiedzi, strumieniowy LLM i strumieniowy TTS, wszystkie połączone przez WebRTC z agresywnymi budżetami opóźnień na każdym przeskoku. Zbuduj go, zmierz WER, MOS i wskaźnik fałszywych przerwań i uruchom go przy utracie pakietów.

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 hours

## Problem

Głos był najszybciej rozwijającą się kategorią UX AI w latach 2025-2026. Techniczny sufit spadał każdego kwartału. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0 i Pipecat 0.0.70 wszystkie umożliwiają sub-800ms pierwszy dźwięk. Granicą nie jest samo opóźnienie. To odczucie interakcji: nie przerywanie użytkownikowi, nie bycie przerwanym, odtwarzanie po przerwaniu w środku zdania, wywoływanie narzędzia w trakcie rozmowy bez zacięcia audio, przetrwanie zaszumionych sieci mobilnych.

Nie osiągniesz tego łącząc trzy wywołania REST. Architektura to strumieniowany potok od końca do końca. Zbuduj go, a tryby awarii staną się widoczne: VAD dostrojony do audio telefonicznego odpalający na telewizorze w tle, detektor zakończenia czekający na interpunkcję, która nigdy nie nadchodzi, TTS buforujący 400ms przed emisją. Capstone polega na naprawianiu ich jeden po drugim pod obciążeniem i opublikowaniu raportu opóźnień i jakości.

## Koncepcja

Potok ma pięć strumieniowych etapów: **audio in** (WebRTC z przeglądarki lub PSTN), **ASR** (strumieniowe częściowe transkrypty z Deepgram Nova-3 lub faster-whisper), **detekcja zakończenia wypowiedzi** (VAD plus mały model detektora, który czyta częściowe transkrypty w poszukiwaniu sygnałów ukończenia), **LLM** (strumieniowe tokeny gdy tylko wypowiedź zostanie uznana za zakończoną), **TTS** (strumieniowe audio na zewnątrz w ciągu ~200ms od pierwszego tokena LLM).

Trzy aspekty przekrojowe. **Wtargnięcie**: gdy użytkownik zaczyna mówić podczas gdy agent mówi, TTS natychmiast anuluje, a ASR przejmuje. **Użycie narzędzi**: wywołania funkcji w trakcie rozmowy (pogoda, kalendarz) muszą działać na bocznym kanale bez zacięcia audio; agent pre-wypełnia token potwierdzenia ("moment..."), jeśli opóźnienie przekracza 300ms. **Backpressure**: przy utracie pakietów, częściowe transkrypty są wstrzymywane, VAD podnosi próg bramki mowy, a agent unika mówienia ponad niepotwierdzoną wiadomością.

Pomiar jest ilościowy. WER poniżej 8% na benchmarku Hamming VAD przy 15 dB SNR. Pierwszy dźwięk p50 poniżej 800ms na 100 zmierzonych rozmowach. Wskaźnik fałszywych przerwań poniżej 3%. MOS powyżej 4.2 na TTS. 50 równoczesnych rozmów na pojedynczym g5.xlarge. Te liczby są rezultatem.

## Architektura

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (lub Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (pogoda,
 Whisper-v3)      per 20ms        on partials        kalendarz)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Stack

- Transport: LiveKit Agents 1.0 (WebRTC) plus Twilio PSTN gateway; Pipecat 0.0.70 jako alternatywny framework
- ASR: Deepgram Nova-3 (strumieniowy, sub-300ms pierwszy częściowy) lub faster-whisper Whisper-v3-turbo samodzielnie hostowany
- VAD: Silero VAD v5 plus detektor zakończenia LiveKit (mały transformer czytający częściowe transkrypty)
- LLM: OpenAI GPT-4o-realtime dla ścisłej integracji, Gemini 2.5 Flash Live, lub kaskadowy Claude Haiku 4.5 (strumieniowe odpowiedzi, osobna ścieżka audio)
- TTS: Cartesia Sonic-2 (najniższy pierwszy bajt), ElevenLabs Flash v3, lub open-source Orpheus do samodzielnego hostowania
- Narzędzia: FastMCP side-channel dla pogody/kalendarza/rezerwacji; agent pre-emituje wypełniacz jeśli narzędzie zajmuje >300ms
- Obserwowalność: OpenTelemetry spany głosowe, Langfuse ślady głosowe z odtwarzaniem audio
- Wdrożenie: pojedynczy g5.xlarge (24GB VRAM) dla samodzielnie hostowanego Whisper + Orpheus; hostowane API dla najniższego opóźnienia

## Build It

1. **Sesja WebRTC.** Uruchom pokój LiveKit i klienta webowego, który strumieniuje audio z mikrofonu. Na serwerze, dołącz agenta roboczego, który dołącza do pokoju.

2. **Strumieniowanie ASR.** Przekazuj 20ms ramki PCM do Deepgram Nova-3 (lub faster-whisper na GPU). Subskrybuj częściowe i końcowe transkrypty. Loguj opóźnienie na częściowy transkrypt.

3. **VAD i detektor zakończenia wypowiedzi.** Uruchom Silero VAD v5 na strumieniu ramek. Na zdarzeniu końca mowy, odpal detektor zakończenia LiveKit na najnowszym częściowym transkrypcie. Zatwierdź "wypowiedź zakończona" tylko gdy VAD wskazuje ciszę przez 500ms, a detektor punktuje zakończenie > 0.6.

4. **Strumień LLM.** Po zakończeniu wypowiedzi, rozpocznij wywołanie LLM z bieżącą rozmową plus końcowym transkryptem. Strumieniuj tokeny na zewnątrz. Przy pierwszym tokenie, przekaż do TTS.

5. **Strumień TTS.** Cartesia Sonic-2 strumieniuje fragmenty audio z powrotem. Pierwszy fragment musi opuścić serwer w ciągu 200ms od pierwszego tokena LLM. Emituj fragmenty do pokoju LiveKit; klient odtwarza przez bufor jitter WebRTC.

6. **Wtargnięcie w mowę.** Gdy VAD wykryje nową mowę użytkownika podczas odtwarzania TTS, natychmiast anuluj strumień TTS, porzuć pozostałe wyjście LLM i uzbrój ponownie ASR. Opublikuj span `tts_canceled`.

7. **Boczny kanał narzędzi.** Zarejestruj pogodę i kalendarz jako narzędzia do wywoływania funkcji. Gdy wywołane, odpal wywołanie współbieżnie; jeśli nie rozwiąże się w ciągu 300ms, każ LLM wyemitować "moment, sprawdzam" jako wypełniacz; wznów gdy narzędzie zwróci wynik.

8. **Harness ewaluacyjny.** Nagraj 100 rozmów. Oblicz WER (względem wstrzymanego transkryptu), wskaźnik fałszywych przerwań (TTS anulowany gdy użytkownik był w środku zdania), pierwszy dźwięk p50, MOS TTS (ludzki lub NISQA) i test strat pakietów (upuść 3% pakietów).

9. **Test obciążenia.** Przeprowadź 50 równoczesnych rozmów na pojedynczym g5.xlarge z syntetycznym rozmówcą. Zmierz utrzymany pierwszy dźwięk p95.

## Use It

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Ship It

`outputs/skill-voice-agent.md` jest rezultatem. Mając domenę (obsługa klienta, harmonogramowanie lub kiosk), uruchamia agenta LiveKit z potokiem ASR/VAD/LLM/TTS dostrojonym do parametrów pomiarowych. Rubryka:

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Opóźnienie end-to-end | p50 pierwszy dźwięk poniżej 800ms na 100 nagranych rozmowach |
| 20 | Jakość przejmowania wypowiedzi | Wskaźnik fałszywych przerwań poniżej 3% na benchmarku Hamming VAD |
| 20 | Poprawność użycia narzędzi | Wywołania narzędzi w trakcie rozmowy zwracające poprawne dane bez zacięcia audio |
| 20 | Niezawodność przy utracie pakietów | WER i stabilność przejmowania wypowiedzi z 3% wstrzykniętej utraty pakietów |
| 15 | Kompletność harnessu ewaluacyjnego | Powtarzalne pomiary z publiczną konfiguracją |
| **100** | | |

## Ćwiczenia

1. Zamień Deepgram Nova-3 na faster-whisper v3 turbo na g5.xlarge. Zmierz różnicę opóźnienia i WER. Zidentyfikuj, gdzie decyzje CPU-vs-GPU mają znaczenie.

2. Dodaj politykę arbitrażu przerwań: co robi agent, gdy użytkownik wtargnie w mowę podczas wywołania narzędzia? Porównaj trzy polityki (twarde anulowanie, dokończ-narzędzie-potem-zatrzymaj się, dodaj do kolejki następną wypowiedź).

3. Przeprowadź test odporności detektora zakończenia: daj użytkownikowi długie pauzy w środku zdania. Dostrój próg ciszy VAD i próg punktacji detektora zakończenia dla najniższego fałszywego przerwania bez przekraczania 900ms.

4. Wdróż tego samego agenta na PSTN przez Twilio. Porównaj pierwszy dźwięk PSTN do WebRTC. Wyjaśnij różnice bufora jitter i kodeka.

5. Dodaj detekcję aktywności głosowej dla języków innych niż angielski (japoński, hiszpański). Zmierz wskaźnik fałszywych wyzwalaczy Silero VAD v5 względem dostrojeń językowych.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Turn detection | "Koniec wypowiedzi" | Klasyfikator, który mając ciszę VAD i częściowy transkrypt, decyduje, że użytkownik skończył mówić |
| Barge-in | "Obsługa przerwań" | Anulowanie odtwarzania TTS, gdy VAD wykryje nową mowę użytkownika |
| First-audio-out | "Opóźnienie" | Czas od zaprzestania mówienia przez użytkownika do pierwszego pakietu audio opuszczającego serwer |
| VAD | "Bramka mowy" | Model klasyfikujący ramki audio jako mowę lub ciszę; Silero VAD v5 jest domyślnym w 2026 |
| Jitter buffer | "Wygładzanie audio" | Bufor po stronie klienta, który przechowuje pakiety przez krótki czas, aby absorbować zmienność sieci |
| Filler | "Token potwierdzenia" | Krótka fraza, którą agent emituje, aby uniknąć ciszy, gdy narzędzie jest wolne |
| MOS | "Mean opinion score" | Percepcyjna ocena jakości mowy; NISQA jest automatycznym proxy |

## Dalsza lektura

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — reference WebRTC agent framework
- [Pipecat](https://github.com/pipecat-ai/pipecat) — alternate Python-first streaming agent framework
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — reference for integrated speech models
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — streaming ASR reference
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD reference model
- [Cartesia Sonic-2](https://docs.cartesia.ai) — low-latency TTS reference
- [Retell AI architecture](https://docs.retellai.com) — production voice agent architecture
- [Vapi.ai production stack](https://docs.vapi.ai) — alternate production reference