# MIO i Streamingowe Modele Multimodalne Typu Any-to-Any

> GPT-4o dostarcza produkt, którego większość otwartych modeli nie jest w stanie odtworzyć: agenta, który słyszy głos, widzi wideo i odpowiada w czasie rzeczywistym. Odpowiedzią otwartego ekosystemu pod koniec 2024 roku było MIO (Wang i in., wrzesień 2024). MIO tokenizuje tekst, obraz, mowę i muzykę, trenuje jeden transformer przyczynowy na przeplatanych sekwencjach i generuje dowolną modalność na dowolną. AnyGPT (Zhan i in., luty 2024) był koncepcyjnym dowodem; MIO to skalowanie; Unified-IO 2 (Allen AI, grudzień 2023) to kuzyn z ugruntowaniem wizji + działania. Ta lekcja opisuje wzorzec any-to-any — cztery tokenizatory, jeden transformer, dekodowanie przyjazne dla strumieniowania.

**Type:** Learn
**Languages:** Python (stdlib, alokator tokenów dla czterech modalności + pętla dekodowania strumieniowego)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time:** ~120 minutes

## Learning Objectives

- Zaprojektować współdzielone słownictwo mieszczące tokeny tekstu, obrazu, mowy i muzyki bez kolizji.
- Porównać SEED-Tokenizer (obrazy) i SpeechTokenizer residual-VQ (mowa) pod kątem kompresji i rekonstrukcji.
- Wyjaśnić czteroetapowy program nauczania budujący generację any-to-any.
- Wymienić trzy otwarte receptury any-to-any i ich główne kompromisy: MIO, AnyGPT, Unified-IO 2.

## Problem

Łatwo jest deklarować ujednolicony model multimodalny, ale trudno go zbudować w skali. Większość systemów "any-to-any" do 2024 roku była potokowa: model wizyjny → reprezentacja tekstowa → model mowy → audio. Każdy przeskok traci informacje, dodaje opóźnienia i komplikuje trenowanie. Demo GPT-4o pokazało alternatywę z pojedynczym modelem z odpowiedzią poniżej sekundy; otwarte systemy pozostawały w tyle o miesiące.

Wyzwania inżynieryjne:

- Tokenizatory muszą istnieć dla każdej modalności, kompresować wystarczająco bezstratnie dla rekonstrukcji i produkować tokeny w tempie akceptowalnym przez transformer.
- Pojedyncze słownictwo musi pomieścić tekst (32k+), obraz (16k+), mowę (4k+), muzykę (8k+). Minimum czterdzieści tysięcy pozycji.
- Dane treningowe muszą pokrywać każdą parę wejście-wyjście (tekst→obraz, obraz→mowa, mowa→obraz itp.) lub model musi komponować.
- Inferencja musi strumieniować tokeny wyjściowe wystarczająco szybko dla opóźnienia konwersacyjnego (<500ms czasu do pierwszego bajtu audio).

## Koncepcja

### Cztery tokenizatory dla czterech modalności

Stos tokenizatorów MIO:

- Tekst: standardowy BPE, słownictwo ~32000.
- Obraz: SEED-Tokenizer (2023) — skwantowany VAE z dyskretną księgą kodów, 4096 pozycji, 32x32 tokenów na obraz.
- Mowa: SpeechTokenizer residual-VQ (2023) — koduje przebieg 16kHz do 8 hierarchicznych ksiąg kodów; pierwszy poziom to zgrubna treść, kolejne dodają prozodię i tożsamość mówcy.
- Muzyka: podobny residual-VQ (rodzina Meta MusicGen / Encodec), 4-8 ksiąg kodów.

Każda modalność produkuje tokeny całkowitoliczbowe. Tokeny otrzymują rozłączne zakresy ID we współdzielonym słownictwie:

```
text:   0..31999
image:  32000..36095  (4096 tokenów obrazu)
speech: 36096..40191  (4096 podstawowych tokenów mowy plus warstwy rezydualne)
music:  40192..48383  (8192 tokenów muzyki)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, itd.)
```

Razem: ~48k słownictwa. Osadzanie wejściowe i projekcja wyjściowa obejmują całość.

### Dekodowanie strumieniowe

Generowanie mowy wykorzystuje residual-VQ. Transformer przewiduje podstawowe (warstwa 0) tokeny mowy; równolegle dekodowany kwantyzator rezydualny przewiduje kolejne warstwy. Każdy token warstwy 0 to około 50ms audio przy 16kHz.

Wzorzec strumieniowania:

1. Użytkownik mówi do mikrofonu; tokenizator audio w czasie rzeczywistym emituje tokeny mowy co 50ms.
2. MIO konsumuje tokeny w miarę ich napływania (prefill promptu + przyrostowe forward).
3. Tokeny wyjściowe są strumieniowane w miarę generowania; równoległy dekoder mowy konwertuje je na próbki audio z opóźnieniem ~50-150ms.
4. Czas do pierwszego bajtu audio: ~300-500ms w pracy MIO, zbliżając się do ~250ms GPT-4o.

Mini-Omni (arXiv:2408.16725), GLM-4-Voice (arXiv:2412.02612) i Moshi (arXiv:2410.00037) to uzupełniające projekty strumieniowego LLM mowy. Moshi w szczególności osiąga 160ms czasu odpowiedzi na pojedynczym GPU.

### Czteroetapowy program nauczania

Program treningowy MIO:

1. Etap 1 — alignment. Wielkoskalowe korpusy par modalności: tekst-obraz, tekst-mowa, tekst-muzyka. Każda para używa własnego segmentu słownictwa tokenowego. Trenuje współdzielone słownictwo.
2. Etap 2 — przeplatane. Dokumenty z przeplatanymi wieloma modalnościami (blogi z obrazami + wideo, podcasty z transkrypcjami itp.). Trenuje kontekst między-modalności.
3. Etap 3 — wzbogacanie mowy. Dodatkowe dane audio poprawiające jakość mowy bez utraty możliwości tekstowych.
4. Etap 4 — SFT. Dostrajanie instrukcyjne na różnych modalnościach: VQA, captioning, narracja, dialog mowa-mowa.

Pominięcie etapu degraduje konkretne możliwości: pomiń etap 2, a model traci kontekst między-modalności; pomiń etap 3, a mowa jest słaba.

### Łańcuch myśli wizualnej

MIO wprowadza łańcuch myśli wizualnej: model emituje pośrednie tokeny obrazu jako krok rozumowania. Dla "czy kot wspina się na drzewo?" model:

1. Emituje tokeny `<image>` renderujące scenę (z obrazu wejściowego lub szkicu).
2. Emituje tekst analizujący szkic.
3. Emituje ostateczną odpowiedź.

Wyrenderowany pośredni obraz służy jako brudnopis. Benchmarki poprawiają się w zadaniach rozumowania przestrzennego. Pomysł odzwierciedla łańcuch myśli dla rozumowania tekstowego.

### Konkurenci w any-to-any

- AnyGPT (arXiv:2402.12226): 4 modalności (tekst, obraz, mowa, muzyka), podobny projekt.
- Unified-IO 2 (arXiv:2312.17172): dodaje wyjścia akcji wizyjnych, głębię, normalne. Większa różnorodność zadań, mniejsza skala.
- NExT-GPT (arXiv:2309.05519): LLM + modalno-specyficzne dekodery dyfuzyjne. Nie jest podejściem z jednym modelem.
- CoDi (arXiv:2305.11846): komponowalna dyfuzja; any-to-any przez współdzieloną latentną.

MIO jest najbliższe czystemu tokenowemu any-to-any. AnyGPT jest jego koncepcyjnym przodkiem.

### Budżet opóźnienia

Dla produktu konwersacyjnego opóźnienie każdego komponentu ma znaczenie:

- Mikrofon do tokenów audio: ~50ms.
- Prefill (tokeny audio + historia): ~100ms na modelu 8B.
- Pierwszy token wyjściowy: ~50ms.
- Równoległy residual-VQ + dekoder mowy: ~100-150ms.

Całkowity czas do pierwszego bajtu audio: ~300ms minimum. GPT-4o deklaruje ~250ms. Moshi deklaruje 160ms. MIO/AnyGPT mieszczą się w zakresie 400-600ms według publicznych benchmarków.

### Dlaczego any-to-any pozostaje trudne

Nawet w 2026 roku otwarte modele any-to-any pozostają w tyle za zamkniętymi w dwóch obszarach:

- Jakość mowy. Tokenizator residual-VQ jest stratny; mowa konwersacyjna brzmi robotycznie w porównaniu do głosów klasy ElevenLabs.
- Rozumowanie między-modalności. Zadanie "zaśpiewaj o tym, co widzisz" wciąż częściej zawodzi niż czyste zadania wizyjne.

To otwarte problemy badawcze. Qwen3-Omni (Lekcja 12.20) to najbardziej zaawansowana otwarta próba w 2025 roku.

## Use It

`code/main.py`:

- Definiuje alokację słownictwa dla czterech modalności i wypisuje je.
- Kieruje listę multimodalnych wejść (tekst, obraz, klip audio, muzyka) przez router tokenizatora.
- Symuluje strumieniowe dekodowanie dla odpowiedzi tekst-na-mowę z pomiarem opóźnienia.
- Oblicza oczekiwany czas do pierwszego bajtu audio przy zadanych opóźnieniach enkodera, prefillu i dekodera.

## Ship It

Ta lekcja produkuje `outputs/skill-any-to-any-pipeline-auditor.md`. Na podstawie specyfikacji produktu konwersacyjnego (modalności wejściowe, modalności wyjściowe, docelowe opóźnienie) audytuje wybory projektowe z rodziny MIO i oblicza budżet opóźnienia.

## Exercises

1. Twój produkt przyjmuje wejście mowy i zwraca wyjście mowy. Jaki jest docelowy budżet opóźnienia koniec-do-końca? Wymień komponenty, które spędzają czas.

2. SpeechTokenizer residual-VQ używa 8 ksiąg kodów. Zaproponuj, dlaczego równoległe dekodowanie poziomów rezydualnych jest konieczne (w porównaniu do sekwencyjnego) i jakie oszczędności opóźnienia przynosi.

3. Twoje słownictwo ma 32k tekstu + 4k obrazu + 4k mowy. Dodaj 8k muzyki i ~10 separatorów. Jaki jest koszt parametrów macierzy osadzania przy ukrytym wymiarze 4096?

4. Łańcuch myśli wizualnej emituje pośredni obraz. Jakie rodzaje pytań na tym korzystają? Jakie rodzaje są szkodzone przez dodatkowe tokeny?

5. Przeczytaj Moshi (arXiv:2410.00037). Opisz technikę "inner monologue" i porównaj ją z łańcuchem myśli wizualnej MIO.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Any-to-any | "Multimodalne wejście/wyjście" | Pojedynczy model, który przyjmuje i emituje tekst, obraz, mowę i muzykę w dowolnym kierunku |
| Residual-VQ | "Stos tokenizatorów mowy" | Tokenizacja z wieloma księgami kodów, gdzie każda warstwa dodaje informacje; warstwa bazowa to treść, późniejsze warstwy to prozodia |
| SEED-Tokenizer | "Kody obrazu" | Dyskretny tokenizator obrazu z księgą kodów 4096 pozycji używany przez MIO |
| Chain-of-visual-thought | "Wizualny brudnopis" | Model generuje pośredni obraz jako krok rozumowania przed ostateczną odpowiedzią |
| Time-to-first-audio-byte | "TTFAB" | Opóźnienie od głosu użytkownika do pierwszego wyjścia audio; <500ms dla odczucia konwersacyjnego |
| Four-stage curriculum | "Przepis treningowy" | Alignment -> przeplatane -> wzbogacanie mowy -> SFT, w tej kolejności |

## Further Reading

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)