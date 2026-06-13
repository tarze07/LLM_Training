# Modele Omni: Qwen2.5-Omni i Podział Thinker-Talker

> Demo produktowe GPT-4o w maju 2024 było przełomowe nie ze względu na bazowy model, ale na kształt produktu — interfejs głosowy, w którym mówisz, model widzi to, co widzi kamera, i odpowiada w poniżej 250ms. Otwarty ekosystem spędził resztę 2024 i 2025 roku, ścigając się, by osiągnąć tę powierzchnię produktu. Qwen2.5-Omni (marzec 2025) to referencyjny otwarty projekt: Thinker (duży transformer generujący tekst) plus Talker (równoległy transformer generujący mowę), połączone strumieniowymi tokenami mowy. Mini-Omni uprościło go, Moshi dorównało opóźnieniu, GLM-4-Voice rozszerzyło go na chiński. Ta lekcja opisuje architekturę Thinker-Talker i budżet opóźnienia, który umożliwia strumieniowy dialog w czasie rzeczywistym.

**Type:** Build
**Languages:** Python (stdlib, symulator opóźnienia potoku strumieniowego + pętla VAD)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## Learning Objectives

- Podzielić potok inferencji na Thinker (rozumowanie tekstowe) i Talker (synteza mowy) i wyjaśnić, dlaczego równoległe strumieniowanie działa.
- Obliczyć budżet czasu do pierwszego bajtu audio (TTFAB) dla interakcji konwersacyjnej, komponent po komponencie.
- Opisać TMRoPE — kodowanie pozycyjne zgodne czasowo w obrębie wizji, audio i tekstu w Thinkerze.
- Wymienić trzy wzorce konwersacji w czasie rzeczywistym: half-duplex, turn-taking, full-duplex.

## Problem

Asystent głosowy czasu rzeczywistego musi robić dużo, szybko:

1. Słyszeć użytkownika. Tokenizacja mowy w czasie rzeczywistym, detekcja aktywności głosowej (VAD), aby wiedzieć, kiedy skończył mówić.
2. Opcjonalnie widzieć. Wejście kamery przy 2-4 FPS, strumieniowane do Thinkera obok audio.
3. Myśleć. Skomponować odpowiedź warunkowaną historią konwersacji.
4. Mówić. Syntetyzować tokeny audio, dekodować do przebiegu, strumieniować do głośników użytkownika.

Każdy krok dodaje opóźnienia. Odczucie konwersacyjne wymaga całkowitego czasu odpowiedzi < 500ms — poniżej tego użytkownik przestaje zauważać opóźnienie. GPT-4o deklaruje ~250ms. Moshi ~160ms. Qwen2.5-Omni ~350-500ms.

Każdy komponent musi strumieniować. Nic nie może być "przetwórz wszystko wsadowo, potem dekoduj."

## Koncepcja

### Thinker i Talker

Dekompozycja Qwen2.5-Omni:

- Thinker: transformer generujący tekst o wielkości 7B-80B. Konsumuje przeplatane tokeny tekstu + obrazu + audio. Wyprowadza tokeny tekstowe reprezentujące, co powiedzieć.
- Talker: mniejszy transformer generujący mowę (200M-1B). Konsumuje tokeny tekstowe wyjściowe Thinkera plus ostatnie tokeny kontekstu mowy. Wyprowadza dyskretne tokeny mowy (indeksy residual-VQ).
- Dekoder mowy: strumieniowy dekoder przebiegu (rodzina SNAC, MoVQGAN), który przetwarza tokeny mowy na próbki audio w czasie rzeczywistym.

Separacja ma znaczenie. Thinker musi być duży dla dobrego rozumowania. Talker może być mały, ponieważ jego zadanie jest lokalne — konwersja tekstu na tokeny mowy. Większy Talker nie jest bardziej ekspresyjny; jest wolniejszy.

Działanie obu równolegle:

1. Thinker emituje token tekstowy t_i.
2. Talker konsumuje t_i (przez strumieniowanie) i emituje tokeny mowy s_i, s_{i+1}, ..., s_{i+k}.
3. Dekoder mowy konsumuje tokeny mowy w miarę napływania i emituje próbki audio.
4. Zanim Thinker dotrze do tokenu tekstowego t_{i+3}, Talker już strumieniował audio dla t_0..t_{i+2}.

### TMRoPE — multimodalne pozycje zgodne czasowo

Thinker musi integrować klatki obrazu (docierające, powiedzmy, przy 4 FPS), ramki audio (docierające przy 50 ramkach/sekundę) i tekst z historii konwersacji. Naiwny porządek sekwencji (najpierw wszystkie obrazy, potem całe audio, potem tekst) traci zgodność czasową.

TMRoPE przypisuje bezwzględne znaczniki czasu każdemu tokenowi. Token wizyjny przy t=2.3s. Token audio przy t=2.32s. Token tekstowy od użytkownika "stop" przy t=2.35s. RoPE obraca uwagę według znacznika czasu; model widzi je jako czasowo współbieżne.

To jest infrastruktura dla "machnął ręką mówiąc cześć" — model widzi klatkę wideo i audio w tym samym koncepcyjnym momencie.

### Strumieniowa synteza mowy

Tokeny mowy muszą być strumieniowane. Mini-Omni (Xie & Wu, 2024) wprowadziło "language models can hear, talk while thinking in streaming": tokeny wyjściowe Thinkera i tokeny wyjściowe Talkera przeplatają się w tej samej sekwencji. Talker uruchamia się, gdy tylko Thinker zatwierdzi następny token tekstowy. Bez granic wsadowych.

Moshi (Défossez i in., październik 2024) to najszybsza otwarta implementacja. 160ms TTFAB na pojedynczym A100. Architektura: pojedynczy transformer 7B, który emituje tokeny tekstu i mowy na naprzemiennych pozycjach, z "inner monologue" oddzielającym strumień myślenia od strumienia mówienia. To efektywnie Thinker + Talker scalone w jeden model z ostrożnym trenowaniem.

### VAD i turn-taking

Detekcja aktywności głosowej działa po stronie wejścia. Dwa wzorce:

- Half-duplex: użytkownik mówi, model słucha. Model mówi, użytkownik słucha. Wyraźne przekazanie przez detekcję ciszy VAD (~200ms).
- Full-duplex: oboje mogą mówić jednocześnie. Model może dawać sygnały ("mhm") lub przerywać. O wiele trudniejsze. Moshi to obsługuje.

Qwen2.5-Omni domyślnie obsługuje half-duplex, z turn-taking przez próg ciszy. Full-duplex wymaga obsługi na poziomie aplikacji.

### Qwen3-Omni (listopad 2025)

Następca. Thinker Qwen3-80B, większy Talker, ulepszony TMRoPE-v2. Opóźnienie bliskie 250ms GPT-4o. Otwarte wagi. Benchmarki na OmniBench konkurencyjne z Gemini 2.0 Live.

### Budżet opóźnienia produkcyjnego

Dla typowej interakcji strumieniowej:

- Mikrofon -> tokeny audio: 40-80ms.
- Prefill (prompt + historia): 100-200ms przy 7B, znacznie więcej przy 70B.
- Pierwszy token tekstowy Thinkera: 40ms.
- Talker przetwarza pierwszy token tekstowy: 20ms.
- Zatwierdzenie pierwszych tokenów mowy: 40ms.
- Dekodowanie residual-VQ: 30ms.
- Dekodowanie przebiegu mowy: 50-80ms.

Całkowity TTFAB: 320-510ms przy 7B, 600-900ms przy 70B. Jakość graniczna zwykle oznacza 70B+; stąd luka opóźnienia granicznego.

### Matematyka tempa tokenów

Przy mowie 16kHz z podstawowymi tokenami mowy 50 Hz, potrzebujesz 50 tokenów mowy na sekundę wyjścia. Talker musi emitować ≥50 tok/s, aby nadążyć. Przy typowej przepustowości LLM 30-80 tok/s na H100, mały Talker (200-300M) jest wystarczająco szybki; Talker 7B zostałby w tyle.

Dlatego istnieją małe dedykowane modele Talker zamiast "po prostu użyj głównego modelu."

## Use It

`code/main.py`:

- Symuluje potok Thinker-Talker z pozorowanymi szybkościami emisji tokenów.
- Oblicza TTFAB dla konfigurowalnych rozmiarów modeli i częstotliwości próbkowania mikrofonu.
- Demonstruje turn-taking half-duplex z progiem ciszy VAD.

## Ship It

Ta lekcja produkuje `outputs/skill-omni-streaming-budget.md`. Na podstawie docelowego TTFAB i zestawu funkcji produktu głosowego czasu rzeczywistego (wizja w, dwujęzyczność, full-duplex) wybiera Qwen2.5-Omni, Qwen3-Omni, Moshi lub Mini-Omni i określa rozmiar Thinkera/Talkera.

## Exercises

1. Twój docelowy TTFAB to 300ms. Na Thinkerze 7B i Talkerze 300M, wypisz opóźnienie każdego komponentu.

2. Qwen2.5-Omni używa TMRoPE. Opisz, co model widzi dla promptu, gdzie użytkownik zaczyna mówić przy t=1s, a kamera rejestruje gest przy t=1.2s.

3. Obsługa full-duplex wymaga, aby model emitował audio podczas słuchania. Zaproponuj format danych treningowych, który tego uczy.

4. Przeczytaj sekcję 4 pracy Moshi. Opisz separację "inner monologue" i dlaczego unika podziału Thinker-Talker.

5. Oblicz budżet przepustowości: jak szybko Talker musi emitować tokeny, aby nadążyć za mową 16kHz przy 50 tokenach warstwy bazowej/sekundę?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Thinker | "Mózg rozumujący" | Duży transformer generujący tekst, produkujący co powiedzieć |
| Talker | "Usta generujące mowę" | Mały transformer produkujący dyskretne tokeny mowy z tekstu Thinkera |
| TTFAB | "Budżet opóźnienia" | Czas do pierwszego bajtu audio: od zakończenia mowy użytkownika do pierwszej próbki audio na wyjściu |
| TMRoPE | "Czasowo-wyrównany RoPE" | Kodowanie pozycyjne używające bezwzględnych znaczników czasu w poprzek wizji, audio i tekstu |
| Half-duplex | "Turn-taking" | Użytkownik i model naprzemiennie; cisza VAD wykrywa koniec użytkownika |
| Full-duplex | "Symultaniczny" | Model może mówić i słuchać jednocześnie; zdolny do backchannel |
| Inner monologue | "Separacja Moshi" | Projekt z jednym modelem, w którym strumień myślenia i strumień mówienia przeplatają się |

## Further Reading

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)