# Modele Audio-Językowe: Łuk od Whisper do Audio Flamingo 3

> Whisper (Radford i in., grudzień 2022) ustabilizował rozpoznawanie mowy — 680k godzin słabo nadzorowanej wielojęzycznej mowy, prosty transformer enkoder-dekoder, benchmark, który sprawił, że każda kolejna publikacja ASR go cytuje. Ale rozpoznawanie to nie rozumowanie. Zapytanie "jakie instrumenty są w tym nagraniu" lub "jaką emocję wyraża mówca" lub "co się stało w 3 minucie" wymaga zrozumienia audio, nie transkrypcji. Qwen-Audio, SALMONN, LTU i NVIDIA Audio Flamingo 3 (AF3, lipiec 2025) stopniowo budowały ten stos: zachowaj enkodery klasy Whisper, dołącz Q-formery, trenuj na danych instrukcyjnych audio-tekst, dodaj rozumowanie przez łańcuch myśli. Ta lekcja opisuje ten łuk.

**Type:** Build
**Languages:** Python (stdlib, spektrogram log-Mel + szkielet audio Q-formera)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## Learning Objectives

- Obliczyć spektrogram log-Mel z przebiegu: okienkowanie, FFT, banki filtrów, transformacja logarytmiczna.
- Porównać opcje enkoderów: enkoder Whisper, BEATs, hybryda AF-Whisper. Kiedy każdy wygrywa.
- Zbudować audio Q-former: N uczonych zapytań cross-attendujących do łat spektrogramu.
- Wyjaśnić trenowanie kaskadowe (Whisper-następnie-LLM) vs end-to-end audio-LLM: dlaczego end-to-end skaluje się lepiej dla rozumowania.

## Problem

Rozpoznawanie mowy zostało rozwiązane przez Whisper. OCR dla audio jest towarem. Ale "towar" kończy się na transkrypcji. Jeśli model nie może rozumować nad tym, co usłyszał — timing, mówcy, emocje, struktura muzyczna, dźwięki otoczenia — sama transkrypcja nie może napędzać funkcji produktu.

Trzy oczywiste ścieżki:

1. Kaskada: Whisper transkrybuje, LLM rozumuje nad transkrypcją. Działa dla scenariuszy czystej mowy. Zawodzi dla muzyki, audio otoczenia, nakładania się wielu mówców, emocji.

2. End-to-end audio-LLM: enkoder audio podaje tokeny audio bezpośrednio do LLM, pomijając transkrypcję. Zachowuje informację akustyczną (emocje, mówca, otoczenie). Wymaga nowych danych treningowych.

3. Hybryda: enkoder audio + dekoder tekstowy, który może zarówno transkrybować, jak i rozumować. Qwen-Audio i Audio Flamingo wybierają tę ścieżkę.

## Koncepcja

### Spektrogram log-Mel: cecha wejściowa

Każdy enkoder audio zaczyna od tej samej cechy: spektrogramu log-Mel.

1. Resampluj do 16 kHz.
2. Krótkoczasowa transformata Fouriera z oknami 25ms, skokiem 10ms.
3. Weź amplitudę wyniku FFT.
4. Zastosuj banki filtrów Mel (zazwyczaj 80 filtrów log-spaced 0-8000 Hz), aby przekształcić do częstotliwości percepcyjnej.
5. Kompresja logarytmiczna (log(1 + x)) dla zakresu dynamicznego.

Wynik: tablica 2D o kształcie (T, 80), gdzie T to liczba ramek czasowych. Dla 30-sekundowego klipu przy częstotliwości ramek 100 Hz: (3000, 80).

### Enkoder Whisper

Enkoder Whisper to 12-warstwowy transformer w stylu ViT przetwarzający spektrogram log-Mel jako sekwencję ramek czasowych. Wynik: jeden wektor stanu ukrytego na ramkę czasową.

Dla ASR, dekoder Whisper to transformer z cross-attention, który generuje tokeny tekstowe warunkowane wyjściem enkodera. Standardowy enkoder-dekoder.

Dla ALM (audio-LLM), chcesz użyć wyjścia enkodera jako wejścia do innego LLM. Wzorzec: enkoder Whisper zamrożony, Q-former trenowalny, LLM zamrożony lub dostrajany.

### BEATs i enkodery specyficzne dla audio

Whisper był trenowany na danych zdominowanych przez mowę. Jest słabszy dla muzyki i audio otoczenia.

BEATs (Chen i in., 2022) to samonadzorowany transformer trenowany na AudioSet. Lepiej uchwyca muzykę i dźwięki otoczenia niż Whisper przy tej samej liczbie parametrów.

AF-Whisper (hybryda Audio Flamingo 3): konkatenacja cech Whisper + BEATs jako wejście audio. Whisper niesie sygnał lingwistyczny, BEATs niesie sygnał akustyczny.

### Audio Q-former

Ten sam wzorzec co wizualny Q-former BLIP-2. Stała liczba uczonych zapytań (często 32 lub 64) cross-attenduje nad ramkami wyjściowymi enkodera audio. Zapytania stają się tokenami audio konsumowanymi przez LLM.

Etap alignment treningu: sam Q-former, straty kontrastywne + captioning na parach audio-tekst (AudioCaps, Clotho). Etap instrukcyjny: end-to-end, odmroź LLM, trenuj na danych instrukcyjnych.

### Łuk — SALMONN, Qwen-Audio, AF3

SALMONN (Tang i in., 2023): Whisper + BEATs + Q-former + LLaMA. Pierwszy otwarty audio-LLM z poważną zdolnością rozumowania. Benchmarki na MMAU pokazują ~0.55 composite.

Qwen-Audio (Chu i in., 2023): podobna architektura, trenowany na bogatszym zbiorze danych, dostrojony do wieloobrotowego dialogu. MMAU ~0.60.

LTU — Listen, Think, Understand (Gong i in., 2023): jawne dane rozumowania, skupienie na łańcuchu myśli nad klipami audio. Mniejszy, ale bardziej ukierunkowany.

Audio Flamingo 3 (Goel i in., lipiec 2025): obecny otwarty SOTA. Szkielet LLM 8B (Qwen2 7B), enkoder Whisper-large konkatenowany z BEATs, Q-former z 64 zapytaniami, trenowanie na 1M+ parach instrukcyjnych audio-tekst. MMAU 0.72, dorównuje własnościowym modelom granicznym w niektórych podzadaniach.

AF3 wprowadza również łańcuch myśli na żądanie dla audio: model może opcjonalnie emitować tokeny myślowe ("pozwól mi najpierw zidentyfikować instrumenty: ...") przed ostateczną odpowiedzią. Dokładność w złożonych zadaniach rozumowania wzrasta o 3-5 punktów, gdy myślenie jest włączone.

### Kaskada vs end-to-end

Potok kaskadowy:

1. Whisper transkrybuje audio → tekst.
2. LLM rozumuje nad tekstem.

Działa idealnie dla "podsumuj ten podcast." Zawodzi dla:
- "Jaki jest nastrój tej piosenki?" — nastrój jest w dźwięku, nie w słowach.
- "Kto mówi, Alicja czy Bob?" — wymaga identyfikacji mówcy.
- "W której sekundzie następuje eksplozja?" — ugruntowanie temporalne utracone w tekście.
- "Czy to jest prawdziwe czy generowane audio?" — detekcja deepfake wymaga cech akustycznych.

End-to-end zachowuje sygnał akustyczny. Qwen-Audio i AF3 obsługują muzykę, otoczenie i emocje natywnie.

### Przepis produkcyjny 2026

Dla nowego produktu rozumienia audio:

- Kaskada, jeśli: transkrypcja jest celem, brak muzyki, brak wnioskowania o emocjach.
- Rodzina AF3 / Qwen-Audio, jeśli: muzyka, emocje, wielu mówców lub złożone rozumowanie audio.

Kaskada jest tańsza i prostsza. End-to-end jest bardziej zdolny.

### MMAU — benchmark rozumowania audio

MMAU (Massive Multimodal Audio Understanding) to benchmark rozumowania audio z lat 2024-2025:

- 10 000 par pytań i odpowiedzi audio-tekst obejmujących mowę, muzykę, dźwięki otoczenia.
- Obejmuje klasyfikację, rozumowanie temporalne, rozumowanie przyczynowe, otwarte QA.
- Testuje to, co potoki kaskadowe systematycznie pomijają.

Otwarty SOTA (AF3) przy 0.72; model graniczny własnościowy ~0.78 (Gemini 2.5 Pro, Claude Opus 4.7). Luka jest mniejsza niż delta otwarty-vs-zamknięty VideoMME, co wskazuje, że audio-LLM dojrzewają.

## Use It

`code/main.py`:

- Implementuje obliczanie spektrogramu log-Mel w stdlib: okienkowanie, naiwne DFT, bank filtrów Mel.
- Szkielet audio Q-formera: dla danych ramek wyjściowych enkodera, oblicz Q, K, V, attention i emituj N tokenów.
- Porównanie kaskady vs end-to-end na zabawkowym zadaniu.

## Ship It

Ta lekcja produkuje `outputs/skill-audio-llm-pipeline-picker.md`. Na podstawie zadania audio (transkrypcja, tagowanie muzyki, wnioskowanie o emocjach, diarizacja wielu mówców, klasyfikacja otoczenia) wybiera kaskadę, end-to-end AF3 lub hybrydę.

## Exercises

1. Oblicz wymiar spektrogramu log-Mel dla 30-sekundowego klipu przy 16kHz, oknie 25ms, skoku 10ms, 80 pasmach Mel. Jak to się zmienia przy 48kHz?

2. Dlaczego Whisper osiąga gorsze wyniki na muzyce? Jakie cechy audio uchwyca BEATs, których Whisper nie uchwyca?

3. Audio Q-former z 64 zapytaniami vs 32: przy jakiej złożoności zadania opłaca się 64? 32 oszczędza obliczenia przy czym?

4. Przeczytaj sekcję 4 AF3 o myśleniu na żądanie. Zaproponuj trzy zadania audio, w których łańcuch myśli pomaga najbardziej.

5. Zaimplementuj minimalny potok diarizacji używając wyjścia AF3. Jak sygnalizujesz zmiany mówcy?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Log-Mel spectrogram | "Cechy Mel" | Tablica 2D (czas, częstotliwość) wartości logarytmicznych po bankach filtrów Mel |
| Audio Q-former | "Audio Perceiver" | Wąskie gardło cross-atencyjne od wyjścia enkodera audio do stałoliczbowych zapytań zasilających LLM |
| Cascaded | "ASR-następnie-LLM" | Potok, w którym Whisper transkrybuje, a tekstowy LLM rozumuje; traci informację akustyczną |
| End-to-end | "Audio-LLM" | Cechy audio wchodzą do LLM bezpośrednio przez Q-former; zachowuje sygnał akustyczny |
| BEATs | "Enkoder audio AudioSet" | SSL transformer trenowany na AudioSet; silny w muzyce i dźwiękach otoczenia |
| MMAU | "Benchmark rozumowania audio" | 10k par QA obejmujących mowę, muzykę, otoczenie; standard ewaluacji 2024 |
| On-demand thinking | "Audio CoT" | Model może opcjonalnie emitować tokeny rozumowania przed ostateczną odpowiedzią, podnosi dokładność o 3-5 pkt |

## Further Reading

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)