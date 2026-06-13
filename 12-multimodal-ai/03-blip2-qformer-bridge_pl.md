# Od CLIP do BLIP-2 — Q-Former jako Most Modalności

> CLIP wyrównuje obraz i tekst, ale nie może generować podpisów, odpowiadać na pytania ani prowadzić rozmowy. BLIP-2 (Salesforce, 2023) rozwiązał to za pomocą małego, trenowalnego mostu: 32 uczone wektory zapytań uwzględniają cechy zamrożonego ViT przez cross-attention, a następnie wsuwają się bezpośrednio w strumień wejściowy zamrożonego LLM. 188M parametrów mostu połączyło 11B LLM z ViT-g/14. Każdy adapterowy VLM do 2026 roku — MiniGPT-4, InstructBLIP, kuzyni LLaVA — jest potomkiem. Ta lekcja czyta architekturę Q-Former, wyjaśnia jego dwuetapowe trenowanie i buduje zabawkową wersję, która podaje tokeny wizyjne do zamrożonego dekodera tekstu.

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 minutes

## Learning Objectives

- Wyjaśnienie, dlaczego trenowalne wąskie gardło między zamrożonym enkodorem wizyjnym a zamrożonym LLM bije pełne dostrajanie end-to-end pod względem kosztu i stabilności.
- Implementacja bloku cross-attention, gdzie stały zestaw uczonych zapytań uwzględnia zewnętrzne cechy obrazu.
- Przejście przez dwuetapowy pretrening BLIP-2: reprezentacyjny (ITC + ITM + ITG), a następnie generatywny (strata LM z zamrożonym dekoderem).
- Porównanie Q-Former z prostszym projektorem MLP używanym w LLaVA i uzasadnienie, kiedy każdy wybór wygrywa.

## Problem

Masz zamrożony ViT, który produkuje 256 tokenów łat o wymiarze 1408 na obraz. Masz zamrożony 7B LLM, który oczekuje osadzeń tokenów o wymiarze 4096. Oczywisty most — warstwa liniowa z 1408 do 4096 — działa, ale podanie wszystkich 256 tokenów łat do kontekstu LLM kosztuje 256 dodatkowych tokenów na obraz. Przy batchu 32 obrazów to 8192 tokenów skonsumowanych przez samą modalność wizyjną.

Pytanie BLIP-2: czy można skompresować 256-tokenową reprezentację obrazu do znacznie mniejszej liczby tokenów (np. 32), zachowując wystarczająco dużo informacji, aby LLM mógł opisywać, odpowiadać na pytania i rozumować o obrazie? I czy można trenować ten most bez dotykania zamrożonych magazynów, utrzymując koszt treningu tylko na parametrach mostu?

Odpowiedź: Q-Former. 32 uczone wektory "zapytań", które wykonują cross-attention na tokenach łat ViT, produkując 32-tokenowe podsumowanie wizyjne, które LLM konsumuje. 188M parametrów łącznie. Trenowany z kontrastywnymi, dopasowującymi i generatywnymi celami, zanim kiedykolwiek dotknie LLM.

## Koncepcja

### Uczone zapytania

Podstawowa sztuczka Q-Former: zamiast pozwalać tokenom tekstowym LLM na uwzględnianie łat obrazu, wprowadź nowy zestaw 32 uczonych wektorów zapytań `Q` i pozwól *im* uwzględniać łaty obrazu. Zapytania są parametrami modelu — są uczone podczas treningu i te same 32 zapytania są używane dla każdego obrazu.

Po cross-attention każde zapytanie zawiera skompresowane podsumowanie obrazu — "opisz główny obiekt", "opisz tło", "policz obiekty" itp. Zapytania nie specjalizują się dosłownie w semantycznych etykietach; uczą się kodowania, które obniża docelowe straty.

### Architektura

Q-Former to mały transformer (12 warstw, ~100M parametrów) z dwiema ścieżkami:

1. Ścieżka zapytań: 32 wektory zapytań przepływają przez self-attention (między sobą), następnie cross-attention nad tokenami łat zamrożonego ViT, następnie FFN.
2. Ścieżka tekstu: enkoder tekstu podobny do BERT współdzieli wagi self-attention i FFN ze ścieżką zapytań. Cross-attention jest wyłączone dla ścieżki tekstu.

Podczas treningu obie ścieżki działają. Zapytania i tekst oddziałują przez współdzielony self-attention, co oznacza, że zapytania mogą warunkować się na tekście dla zadań, które tego potrzebują (ITM, ITG). Podczas inferencji dla przekazania do VLM, tylko zapytania przepływają, dając 32 tokeny wizyjne.

### Dwuetapowe trenowanie

BLIP-2 trenuje w dwóch etapach:

Etap 1: uczenie reprezentacji (bez LLM). Trzy straty:
- ITC (image-text contrastive): kontrastywna CLIP między uśrednionymi tokenami zapytań a tokenem CLS tekstu.
- ITM (image-text matching): klasyfikator binarny — czy ta para obraz-tekst pasuje? Z hard-negative mining.
- ITG (image-grounded text generation): przyczynowa głowa LM na tekście, warunkowana na zapytaniach. Wymusza na zapytaniach kodowanie treści możliwej do wygenerowania jako tekst.

Tylko Q-Former się trenuje. ViT jest zamrożony. Żaden LLM nie jest zaangażowany.

Etap 2: uczenie generatywne. Dołącz zamrożony LLM (OPT-2.7B lub Flan-T5-XL itp.). Rzutuj 32 wyjścia zapytań do wymiaru osadzeń LLM przez małą warstwę liniową. Poprzedź je promptowi tekstowemu. Trenuj tylko projekcję liniową i Q-Former na stracie LM nad połączoną sekwencją prompt + obraz + podpis.

Po etapie 2 Q-Former + projekcja to pełny adapter wizyjny. Przy inferencji: obraz → ViT → Q-Former → projekcja liniowa → poprzedzona tekstowi → zamrożony LLM emituje wyjście.

### Ekonomika parametrów

BLIP-2 z ViT-g/14 (1.1B, zamrożony) + OPT-6.7B (6.7B, zamrożony) + Q-Former (188M, trenowany) = 8B łącznie, 188M trenowane. Sam Q-Former to ~2.4% parametrów całego stosu. Koszt treningu odzwierciedla to: dni na kilku A100 vs tygodnie dla end-to-end.

Jakość: BLIP-2 dorównuje lub bije Flamingo-80B w zero-shot VQA, będąc 50x mniejszym. Most działa.

### InstructBLIP i świadomy instrukcji Q-Former

InstructBLIP (2023) rozszerza Q-Former o dodatkowe wejście: sam tekst instrukcji. W czasie cross-attention zapytania mają teraz dostęp zarówno do łat obrazu, jak i instrukcji. Zapytania mogą specjalizować się na instrukcję ("policz samochody", "opisz nastrój") zamiast uczyć się jednego stałego podsumowania. Zyski w benchmarkach na zadaniach poza zestawem treningowym.

### MiniGPT-4 i podejście tylko z projektorem

MiniGPT-4 zachował Q-Former, ale trenował tylko wyjściową projekcję liniową, zamrażając wszystko inne. Tanie, ale kosztem jakości — zapytania były z BLIP-2, nie twoje. Dobre do szybkiej iteracji, nie najlepsza architektura.

### Dlaczego LLaVA poszła prostiej

LLaVA (2023, Lekcja 12.05) zastąpiła Q-Former zwykłym 2-warstwowym MLP, który projektuje każdy token łaty ViT do przestrzeni LLM — 576 tokenów na obraz dla siatki 24x24, wszystkie podane do LLM. Gorsza kompresja, ale pozwala LLM uwzględniać surowe łaty. W tamtym czasie było to kontrowersyjne; pod koniec 2023 dominowało, ponieważ dane instrukcji wizyjnych (LLaVA-Instruct-150k) udowodniły, że MLP można wytrenować, aby zachował wystarczający sygnał. Kompromis: kontekst LLaVA wypełnia się szybciej, ale skaluje się naturalnie do wielu obrazów i wideo.

Do 2026 roku pole się podzieliło: Q-Former przetrwa tam, gdzie budżet tokenów ma znaczenie (długie wideo, wiele obrazów); projektor MLP dominuje, gdzie surowa jakość na token jest priorytetem.

### Gated cross-attention: Flamingo, przodek

Flamingo (Lekcja 12.04) poprzedzał BLIP-2 i używał tego samego pomysłu cross-attention, ale w każdej zamrożonej warstwie LLM, a nie jako pojedynczy most. BLIP-2 pokazał, że można skompresować tylko do warstwy wejściowej i wciąż działać. Gemini i Idefics łączą oba: przeplatane tokeny wejściowe plus opcjonalny gated cross-attention dla in-context few-shot.

### Potomkowie w 2026

- Q-Former: BLIP-2, InstructBLIP, MiniGPT-4 i większość modeli wideo-językowych ze względu na budżet tokenów.
- Perceiver resampler: wariant Flamingo (Lekcja 12.04); rodzina Idefics, Eagle, OmniMAE.
- Projektor MLP: LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- Attention pool: VILA, PaliGemma.

Wszystkie cztery są poprawne. Decydujące pytanie brzmi, czy jesteś ograniczony budżetem tokenów, czy jakością na token.

## Użyj Tego

`code/main.py` buduje cross-attention w stylu Q-Former w standardowym Pythonie:

1. Symuluje 256 tokenów łat obrazu (wymiar 128).
2. Tworzy 32 uczone zapytania (wymiar 128).
3. Uruchamia scaled-dot-product cross-attention (Q z zapytań, K/V z łat).
4. Projekcja do wymiaru LLM (512) przez warstwę liniową.
5. Wyjście 32 tokenów wizyjnych gotowych dla LLM.

Cała matematyka w czystym Pythonie (zagnieżdżone pętle po wektorach). Zabawkowe, ale poprawny kształt. Macierz wag attention jest wypisywana, abyś mógł zobaczyć, z których łat każde zapytanie czerpało.

## Dostarcz

Ta lekcja produkuje `outputs/skill-modality-bridge-picker.md`. Dla docelowej konfiguracji VLM (liczba tokenów enkodera wizyjnego, budżet kontekstu LLM, ograniczenia wdrożeniowe, cel jakościowy), rekomenduje Q-Former vs MLP vs Perceiver resampler z krótkim uzasadnieniem i oszacowaniem liczby parametrów dla każdego mostu.

## Ćwiczenia

1. Zaimplementuj blok cross-attention w PyTorch. Zweryfikuj, że przy 32 zapytaniach i 256 kluczach/wartościach, macierz wag attention to 32 x 256 i każdy wiersz sumuje się do 1 po softmax.

2. W etapie 1 BLIP-2 Q-Former uruchamia trzy straty jednocześnie: ITC, ITM, ITG. Napisz sygnaturę forward dla każdej w pseudokodzie. Która wymaga aktywnej ścieżki enkodera tekstu?

3. Porównaj liczby parametrów: Q-Former (12 warstw, 768 ukrytych) vs 2-warstwowy projektor MLP (1408 → 4096, dwie warstwy). Przy jakiej skali LLM koszt 188M Q-Former zwraca się w wydajności treningu?

4. Przeczytaj Sekcję 3.2 artykułu BLIP-2 (arXiv:2301.12597) o tym, jak inicjalizowany jest Q-Former. Wyjaśnij, dlaczego inicjalizacja z BERT-base (nie losowa) przyspiesza zbieżność.

5. Dla 10-minutowego wideo przy 1 FPS próbkowanego do 60 klatek, oblicz koszt tokenów na klatkę dla (Q-Former → 32 tokeny/klatka) vs (projektor MLP → 576 tokenów/klatka). Który mieści się w oknie kontekstu LLM 128k tokenów?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Q-Former | "Transformer zapytań" | Mały transformer z 32 uczonymi wektorami zapytań, które wykonują cross-attention na zamrożonych cechach ViT |
| Uczone zapytania | "Miękki prompt dla wizji" | Stały zestaw parametrów służących jako strona zapytań cross-attention; uczone na model, współdzielone dla wszystkich wejść |
| Cross-attention | "Q stąd, K/V stamtąd" | Attention, gdzie zapytanie, klucz i wartość pochodzą z różnych źródeł; jak zapytania czerpią z łat ViT |
| ITC | "Kontrastywność obraz-tekst" | Strata w stylu CLIP zastosowana do uśrednionych zapytań Q-Former vs CLS tekstu |
| ITM | "Dopasowanie obraz-tekst" | Klasyfikator binarny na parach z hard-negative mining; wymusza na zapytaniach rozróżnianie drobnych niezgodności |
| ITG | "Generacja tekstu zakotwiczona w obrazie" | Przyczynowa strata LM, gdzie tekst jest generowany warunkowo na zapytaniach; wymusza na zapytaniach kodowanie treści dekodowalnej jako tekst |
| Dwuetapowy pretrening | "Reprezentacja, potem generacja" | Etap 1 trenuje sam Q-Former (ITC/ITM/ITG); Etap 2 dołącza zamrożony LLM i trenuje tylko projekcję + Q-Former |
| Zamrożony magazyn | "Nie dostrajaj" | Wagi enkodera wizyjnego i LLM są stałe; tylko most się trenuje |
| Głowica projekcyjna | "Liniowa do wymiaru LLM" | Końcowa warstwa liniowa mapująca wyjście Q-Former do wymiaru osadzeń LLM |
| Perceiver resampler | "Wersja Flamingo" | Podobne cross-attention z uczonymi zapytaniami, używane przez Flamingo w każdej warstwie, a nie jako pojedynczy most |

## Dalsza Lektura

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — główny artykuł.
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — poprzednik z trio ITC/ITM/ITG.
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse" — koncepcyjny przodek trenowania etapu 1.
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — świadomy instrukcji Q-Former.
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — podejście tylko z projektorem.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — ogólna architektura dla cross-attention z uczonymi zapytaniami.