# Generowanie dźwięku

> Dźwięk to sygnał 1-D o częstotliwości 16-48 kHz. Pięciosekundowy klip to 80-240k próbek. Żaden transformer nie obsługuje bezpośrednio takiej sekwencji. Rozwiązanie dla każdego produkcyjnego modelu audio w 2026 roku jest takie samo: neuronowy kodek (Encodec, SoundStream, DAC) kompresuje dźwięk do dyskretnych tokenów przy 50-75 Hz, a transformer lub model dyfuzyjny generuje tokeny.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## Problem

Trzy zadania generowania dźwięku:

1. **Zamiana tekstu na mowę.** Dany tekst, wygeneruj mowę. Czysta mowa jest wąskopasmowa i ma silną strukturę fonetyczną — dobrze rozwiązywana przez transformer-over-tokens. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Generowanie muzyki.** Dany prompt (tekst, melodia, progresja akordów, gatunek), wygeneruj muzykę. Znacznie szersza dystrybucja. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **Efekty dźwiękowe / projektowanie dźwięku.** Dany prompt, wygeneruj dźwięk otoczenia lub Foley. AudioGen, AudioLDM 2, Stable Audio Open.

Wszystkie trzy działają na tym samym podłożu: neuronowy kodek audio + token-AR lub generator dyfuzyjny.

## Koncepcja

![Generowanie dźwięku: tokeny kodeka + transformer lub dyfuzja](../assets/audio-generation.svg)

### Neuronowe kodeki audio

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). Splotowy enkoder kompresuje przebieg do wektora na krok czasowy; residualna kwantyzacja wektorowa (RVQ) konwertuje każdy wektor na kaskadę K indeksów z książki kodowej. Dekoder odwraca proces. Dźwięk 24 kHz przy 2 kbps z użyciem 8 książek kodowych RVQ przy 75 Hz = 600 tokenów/sek.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Dwa paradygmaty generatywne na wierzchu

**Token-autoregresywny.** Spłaszcz tokeny RVQ w sekwencję, uruchom transformer tylko z dekoderem. MusicGen używa "opóźnionego równoległości" (delayed parallel), aby emitować K strumieni książek kodowych równolegle z offsetami na strumień. VALL-E generuje tokeny mowy z promptu tekstowego + 3-sekundowej próbki głosu.

**Latentna dyfuzja.** Spakuj tokeny kodeka jako ciągłe latenty lub modeluj je za pomocą kategorycznej dyfuzji. Stable Audio 2.5 używa flow matchingu na ciągłych latentach audio. AudioLDM 2 używa dyfuzji text-to-mel-to-audio.

Trend 2024-2026: flow matching wygrywa dla muzyki (szybsza inferencja, czystsze próbki), podczas gdy token-AR wciąż dominuje w mowie, ponieważ jest naturalnie przyczynowy i dobrze strumieniuje.

## Krajobraz produkcyjny

| System | Zadanie | Szkielet | Opóźnienie |
|--------|---------|----------|------------|
| ElevenLabs V3 | TTS | Token-AR + neuronowy wokoder | ~300ms pierwszy token |
| OpenAI GPT-4o audio | Pełny dupleks mowy | End-to-end wielomodalny AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Niestrumieniowy |
| Stable Audio 2.5 | Muzyka / SFX | DiT + flow matching na latentach audio | ~10s na 1-minutowy klip |
| Suno v4 | Pełne piosenki | Nieujawnione; podejrzewany token-AR | ~30s na piosenkę |
| Udio v1.5 | Pełne piosenki | Nieujawnione | ~30s na piosenkę |
| MusicGen 3.3B | Muzyka | Token-AR na Encodec 32kHz | Czas rzeczywisty |
| AudioCraft 2 | Muzyka + SFX | Flow matching | ~5s na 5s klip |
| Riffusion v2 | Muzyka | Dyfuzja spektrogramu | ~10s |

## Zbuduj to

`code/main.py` symuluje główną ideę: trenuj mały transformer następnego tokena na syntetycznych sekwencjach "tokenów audio" wygenerowanych z dwóch odrębnych "stylów" (naprzemienne niskie i wysokie tokeny dla stylu A, monotoniczna rampa dla stylu B). Warunkuj na styl i próbkuj.

### Krok 1: syntetyczne tokeny audio

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Krok 2: trenuj mały predyktor tokenów

Predyktor w stylu bigramu warunkowany na styl. Chodzi o wzór: tokeny kodeka → trening z cross-entropią → autoregresyjne próbkowanie.

### Krok 3: próbkuj warunkowo

Mając token stylu i token początkowy, próbkuj następny token z przewidzianego rozkładu. Kontynuuj przez 20-40 tokenów.

## Pułapki

- **Jakość kodeka ogranicza jakość wyjścia.** Jeśli kodek nie jest w stanie wiernie odtworzyć dźwięku, żadna jakość generatora nie pomoże. DAC jest obecnie najlepszym otwartym rozwiązaniem.
- **Kumulacja błędów RVQ.** Każda warstwa RVQ modeluje residuum poprzedniej. Błędy na warstwie 1 propagują się. Próbkowanie z temperaturą 0 na wyższych warstwach pomaga.
- **Struktura muzyczna.** 30 sekund tokenów to 20k+ tokenów przy 75 Hz. Trudne dla transformerów. MusicGen używa przesuwanego okna + kontynuacji promptu; Stable Audio używa krótszych klipów + crossfade'owania.
- **Artefakty na granicach.** Crossfade'owanie między wygenerowanymi klipami wymaga ostrożnego overlap-add.
- **Apetyt na czyste dane.** Generatory muzyki potrzebują dziesiątek tysięcy godzin licencjonowanej muzyki. Pozew RIAA przeciw Suno / Udio (2024) ujawnił ten problem.
- **Etyka klonowania głosu.** 3-sekundowa próbka plus prompt tekstowy wystarcza VALL-E / XTTS / ElevenLabs do sklonowania głosu. Każdy model produkcyjny potrzebuje wykrywania nadużyć + list opt-out.

## Użyj tego

| Zadanie | Stos 2026 |
|---------|-----------|
| Komercyjny TTS | ElevenLabs, OpenAI TTS lub Azure Neural |
| Klonowanie głosu (z weryfikacją zgody) | XTTS v2 (otwarty) lub ElevenLabs Pro |
| Muzyka w tle, szybko | Stable Audio 2.5 API, Suno lub Udio |
| Muzyka z tekstem | Suno v4 lub Udio v1.5 |
| Efekty dźwiękowe / Foley | AudioCraft 2, ElevenLabs SFX lub Stable Audio Open |
| Agent głosowy w czasie rzeczywistym | GPT-4o realtime lub Gemini Live |
| Otwarte badania nad muzyką | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Dubbing / tłumaczenie | HeyGen, ElevenLabs Dubbing |

## Dostarcz to

Zapisz `outputs/skill-audio-brief.md`. Skill przyjmuje brief audio (zadanie, czas trwania, styl, głos, licencja) i zwraca: model + hosting, format promptu (tagi gatunku, deskryptory stylu, znaczniki strukturalne), łańcuch kodek + generator + wokoder, protokół seeda oraz plan ewaluacji (MOS / CLAP score / CER dla TTS / A/B użytkownika).

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` i jawnie ustaw styl. Zweryfikuj, że wygenerowane sekwencje pasują do wzorca stylu.
2. **Średnie.** Dodaj opóźnione równoległe dekodowanie: symuluj 2 strumienie tokenów, które muszą pozostać przesunięte o 1 krok. Trenuj wspólny predyktor.
3. **Trudne.** Użyj HuggingFace transformers, aby uruchomić MusicGen-small lokalnie. Wygeneruj 10-sekundowy klip z trzema różnymi promptami; A/B pod kątem zgodności stylu.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Kodek | "Neuronowa kompresja" | Enkoder/dekoder dla dźwięku; typowe wyjście to tokeny 50-75 Hz. |
| RVQ | "Residualna VQ" | Kaskada K kwantyzatorów; każdy modeluje residuum poprzedniego. |
| Token | "Jeden symbol kodeka" | Dyskretny indeks do książki kodowej; typowo 1024 lub 2048. |
| Opóźnione równoległe | "Offsetowane książki kodowe" | Emituj K strumieni tokenów z przesuniętymi offsetami, aby zmniejszyć długość sekwencji. |
| Flow matching | "Zwycięzca 2024 dla audio" | Alternatywa o prostszej ścieżce dla dyfuzji; szybsze próbkowanie. |
| Prompt głosowy | "3-sekundowa próbka" | Embedding mówcy lub prefiks tokenów sterujący sklonowanym głosem. |
| Spektrogram mel | "Wizualizacja" | Percepcyjny spektrogram logarytmicznej wielkości; używany przez wiele systemów TTS. |
| Wokoder | "Mel na przebieg" | Neuronowy komponent konwertujący spektrogramy mel z powrotem na dźwięk. |

## Uwaga produkcyjna: dźwięk to problem strumieniowania

Dźwięk to jedyna modalność wyjściowa, której użytkownicy oczekują *w trakcie generowania*, a nie w całości naraz. W kategoriach produkcyjnych oznacza to, że TPOT (Time Per Output Token) ma znaczenie, ponieważ prędkość słuchania użytkownika jest docelową przepustowością — nie prędkość czytania. Dla dźwięku 16kHz tokenizowanego przy ~75 tokenów/sekundę (Encodec), serwer musi generować ≥75 tokenów/sekundę na użytkownika, aby playback był płynny.

Dwie konsekwencje architektoniczne:

- **Modele audio z flow matchingiem nie mogą łatwo strumieniować.** Stable Audio 2.5 i AudioCraft 2 renderują stałą długość klipu w jednym przebiegu. Aby strumieniować, dzielisz klip na fragmenty i nakładasz granice — pomyśl o dyfuzji z przesuwanym oknem — dodając 100-300ms narzutu opóźnienia w porównaniu do modelu AR kodeka.

Jeśli produkt to "czat głosowy na żywo" lub "kontynuacja muzyki w czasie rzeczywistym", wybierz ścieżkę AR kodeka. Jeśli to "wyrenderuj 30-sekundowy klip po przesłaniu", flow matching wygrywa pod względem jakości i całkowitego opóźnienia.

## Dalsza lektura

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — standard kodeka.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — pierwszy szeroko stosowany neuronowy kodek audio.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — tekst-na-muzykę 2025 z flow matchingiem.