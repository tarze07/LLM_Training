# LLaVA i Dostrajanie Instrukcji Wizyjnych

> LLaVA (kwiecień 2023) jest najczęściej kopiowaną architekturą multimodalną na świecie. Zastąpiła Q-Former BLIP-2 2-warstwowym MLP, zastąpiła gated cross-attention Flamingo naiwną konkatenacją tokenów i trenowała na 158k turach instrukcji wizyjnych wygenerowanych przez GPT-4 z podpisów tekstowych. Każdy praktyk, który zbudował VLM między 2023 a 2026, zbudował jakiś wariant LLaVA. LLaVA-1.5 dodał AnyRes. LLaVA-NeXT podbił rozdzielczość. LLaVA-OneVision zunifikował obraz, wiele obrazów i wideo w jednym przepisie. Ta lekcja czyta przepis, implementuje projektor i wyjaśnia, dlaczego "prostsze wygrało."

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 minutes

## Learning Objectives

- Zbudowanie 2-warstwowego projektora MLP, który mapuje osadzenia łat ViT (wymiar 1024) do wymiaru osadzeń LLM (wymiar 4096).
- Przejście przez dwuetapowy przepis LLaVA: (1) wyrównanie projektora na 558k parach podpisów, (2) dostrajanie instrukcji wizyjnych na 158k turach wygenerowanych przez GPT-4.
- Skonstruowanie promptu w formacie LLaVA z placeholderem tokena obrazu, promptem systemowym i turami użytkownika/asystenta.
- Wyjaśnienie, dlaczego społeczność przeszła z Q-Former na MLP, mimo przewagi Q-Former w budżecie tokenów.

## Problem

Q-Former BLIP-2 (Lekcja 12.03) kompresuje obraz do 32 tokenów. Czysty, wydajny, dobry do benchmarków. Ale ma dwa problemy.

Po pierwsze, Q-Former jest trenowalny, ale jego strata nie jest docelowym zadaniem. Etap 1 trenuje ITC+ITM+ITG. Etap 2 trenuje stratę LM. Zapytania uczą się pewnej pośredniej reprezentacji, którą LLM musi następnie odkodować. Informacje są tracone w wąskim gardle.

Po drugie, Q-Former zajmuje 188M parametrów, a przy skali LLaVA w 2023 roku musiałeś go współprojektować z docelowym LLM. Zmień LLM, trenuj ponownie Q-Former. Zmień enkoder wizyjny, trenuj ponownie. Każda kombinacja była osobnym projektem R&D.

Odpowiedź LLaVA była zawstydzająco prosta: weź 576 tokenów łat ViT, przepuść każdy przez 2-warstwowy MLP (`1024 → 4096 → 4096`) i wrzuć wszystkie 576 do sekwencji wejściowej LLM. Żadnego wąskiego gardła. Żadnego pretreningu etapu 1 na dziwnych celach. Po prostu trenuj MLP na bezpośredniej stracie LM.

Skąd pochodzą dane? Drugi kluczowy wgląd LLaVA: użyj GPT-4 (tylko tekst) do generowania danych instrukcji. Podaj GPT-4 podpis COCO i dane obwiedni dla obrazu, poproś o wygenerowanie rozmów, opisów i złożonych pytań wymagających rozumowania. 158k tur instrukcja-odpowiedź za darmo. Żadnej ludzkiej adnotacji.

Wynik: VLM, który działał na 8 A100 przez jeden dzień, pobił Flamingo na MMMU i wysłał otwarty punkt kontrolny, który społeczność mogła rozszerzać. Pod koniec 2023 roku miał 50+ forków.

## Koncepcja

### Architektura

LLaVA-1.5 przy 13B:
- Enkoder wizyjny: CLIP ViT-L/14 @ 336 (zamrożony podczas etapu 1, opcjonalnie odmrożony etap 2).
- Projektor: 2-warstwowy MLP z aktywacją GELU, `1024 → 4096 → 4096`.
- LLM: Vicuna-13B (później Llama-3.1-8B).

Przejście w przód dla obrazu + promptu tekstowego:

```
img -> ViT -> 576 łat o wymiarze 1024
laty -> MLP -> 576 tokenów o wymiarze 4096
prompt: system + znacznik "<image>" + pytanie użytkownika
zastąp token <image> 576 rzutowanymi tokenami
podaj pełną sekwencję do LLM
dekoduj odpowiedź
```

Obraz zajmuje 576 tokenów kontekstu LLM. Przy kontekście 2048 pozostawia 1472 tokeny dla tekstu. Przy kontekście 32k to błąd zaokrąglenia.

### Etap 1: wyrównanie projektora

Zamroź ViT. Zamroź LLM. Trenuj tylko 2-warstwowy MLP. Zbiór danych: 558k par obraz-podpis (LAION-CC-SBU). Strata: modelowanie języka na podpisie, warunkowane na rzutowanych tokenach obrazu.

W jednej epoce przy batchu 128 jest to gotowe w kilka godzin. Projektor uczy się mapować przestrzeń ViT do przestrzeni LLM. Żaden nadzór specyficzny dla zadania.

### Etap 2: dostrajanie instrukcji wizyjnych

Odmroź projektor (wciąż trenowalny). Odmroź LLM (zwykle w pełni, czasami LoRA). Trenuj na 158k turach instrukcji wizyjnych.

Dane instrukcji są sztuczką. Liu i in. wygenerowali je przez:
1. Weź obraz COCO.
2. Wyodrębnij opis tekstowy (5 ludzkich podpisów + lista obwiedni).
3. Wyślij do GPT-4 z trzema szablonami promptów:
   - Rozmowa: "Wygeneruj dialog między użytkownikiem a asystentem na temat tego obrazu."
   - Szczegółowy opis: "Podaj bogaty, szczegółowy opis obrazu."
   - Złożone rozumowanie: "Zadaj pytanie wymagające rozumowania o obrazie, a następnie odpowiedz na nie."
4. Sparsuj wyjście GPT-4 na pary (instrukcja, odpowiedź).

Żadne z tego nie dotyka obrazu bezpośrednio — tylko opis tekstowy. GPT-4 halucynuje prawdopodobną treść obrazu. Trochę szumu, ale działało: 158k tur wystarczyło, aby odblokować dialog.

### Dlaczego społeczność to skopiowała

- Brak strat specyficznych dla etapu 1 do strojenia. Strata LM przez cały czas.
- Projektor trenuje w godzinach, nie dniach.
- LLM można wymienić (LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3) przez ponowne trenowanie tylko projektora.
- Potok danych instrukcji wizyjnych używa GPT-4 i jest tani do regeneracji dla nowej domeny.

### LLaVA-1.5 i LLaVA-NeXT

LLaVA-1.5 (październik 2023) dodał:
- Dane zadań akademickich (VQA, OKVQA, RefCOCO) zmiksowane w dostrajaniu instrukcji.
- Lepszy prompt systemowy.
- Kontekst 2048 → 32k.

LLaVA-NeXT (styczeń 2024) dodał:
- AnyRes: podziel obrazy o wysokiej rozdzielczości na siatkę 2x2 lub 1x3 wycinków 336x336 plus jedną globalną miniaturę o niskiej rozdzielczości. Każdy wycinek staje się 576 tokenów; łącznie około 2880 tokenów wizyjnych na obraz. Zadania OCR i wykresów skoczyły.
- Lepsza mieszanka danych instrukcji z ShareGPT4V (wysokiej jakości podpisy GPT-4V).
- Silniejsze bazowe LLM (Mistral-7B, Yi-34B).

### LLaVA-OneVision

Lekcja 12.08 obejmuje OneVision szczegółowo. Krótka wersja: ten sam projektor, ale trenowany z programem nauczania obejmującym pojedynczy obraz, wiele obrazów i wideo w jednym modelu ze współdzielonym budżetem tokenów wizyjnych.

### Porównanie z Q-Former

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Tokeny wizyjne na obraz | 32 | 576 (bazowe) lub 2880 (AnyRes) |
| Trenowalne parametry | 188M + LM | 40M + LM |
| Strata etapu 1 | ITC+ITM+ITG | Tylko LM |
| Wymiana LLM | Wymaga ponownego treningu | Zamiana z minimalnym ponownym treningiem |
| Wiele obrazów | Niewygodne | Naturalne (konkatenacja) |
| Wideo | Niewygodne | Naturalne (konkatenacja na klatkę) |
| Budżet tokenów | Mały | Duży |

MLP wygrywa prostotą i elastycznością tokenów. Q-Former wygrywa budżetem tokenów. Pod koniec 2023 roku budżet tokenów nie był już wiążącym ograniczeniem (konteksty LLM urosły do 32k-128k+), a prostota zdominowała.

### Format promptu

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` to token zastępczy. Przed tokenizacją jest zastępowany 576 tokenami wizyjnymi (lub 2880 z AnyRes). Tokenizer widzi nieco dłuższą sekwencję niż ta, na której był trenowany, ale LLM obsługuje nowe wejście, ponieważ etap 1 go tego nauczył.

### Ekonomika parametrów

Podział LLaVA-1.5-7B:
- CLIP ViT-L/14 @ 336: 303M (zamrożony etap 1, często odmrożony etap 2).
- Projektor (2x liniowy): ~22M trenowalne.
- Llama-7B: 7B.
- Razem: 7.3B parametrów. Trenowalne podczas etapu 2: pełne 7B + 22M projektora.

Koszt treningu dla etapu 2: ~20 godzin na 8xA100. To jest kluczowa liczba — jeden dzień, jeden węzeł, do odtworzenia. Dlatego LLaVA się rozprzestrzeniło.

## Użyj Tego

`code/main.py` implementuje:

1. 2-warstwowy projektor MLP (wymiar 16 → 32 → 32 w skali zabawkowej) w czystym Pythonie.
2. Potok budowania promptu: prompt systemowy + `<image>` zastąpiony N rzutowanymi tokenami + tura użytkownika + zastępczy placeholder generowania asystenta.
3. Wizualizator pokazujący, jak blok wizyjny 576 tokenów wygląda w kontekście LLM (procent skonsumowanego kontekstu 2k / 32k / 128k).

## Dostarcz

Ta lekcja produkuje `outputs/skill-llava-vibes-eval.md`. Dla punktu kontrolnego z rodziny LLaVA, uruchamia zestaw 10-promptowej ewaluacji vibes (3 opisywanie, 3 VQA, 2 rozumowanie, 2 odmowa) i raportuje czytelną dla człowieka kartę wyników. Nie benchmark; test dymny potwierdzający, że projektor i LLM są dobrze połączone.

## Ćwiczenia

1. Oblicz liczbę trenowalnych parametrów dla 2-warstwowego projektora MLP przy `1024 → 4096 → 4096`. Z GELU i biasem, jaką frakcję LLaVA-13B to stanowi?

2. Skonstruuj prompt LLaVA dla przypadku "odmowy" — obraz zawiera osobę prywatną. Napisz oczekiwaną odpowiedź asystenta. Dlaczego LLaVA powinna odmówić zero-shot i jakie dane treningowe byłyby potrzebne, aby wzmocnić odmowę?

3. Przeczytaj sekcję AnyRes z bloga LLaVA-NeXT. Oblicz liczbę tokenów wizyjnych dla obrazu 1344x672 przy AnyRes. Porównaj do bazowych 576 tokenów przy 336x336.

4. Projektor etapu 1 LLaVA jest trenowany ze stratą LM na podpisach. Co się stanie, jeśli pominiesz etap 1 i przejdziesz bezpośrednio do etapu 2 (dostrajanie instrukcji wizyjnych)? Zacytuj ablację Prismatic VLMs (arXiv:2402.07865) dla odpowiedzi.

5. LLaVA-Instruct-150k używa GPT-4 z podpisami COCO do generowania instrukcji. Dla nowej domeny (medyczne zdjęcia rentgenowskie, obrazy satelitarne), opisz czteroetapowy potok danych do generowania instrukcji domenowych. Co może pójść nie tak na każdym etapie?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Projektor | "Most MLP" | 2-warstwowy MLP z GELU mapujący wymiar ViT do wymiaru LLM |
| Token obrazu | "Znacznik <image>" | Oznacznik w prompcie zastępowany przez N rzutowanych tokenów wizyjnych przed inferencją |
| Dostrajanie instrukcji wizyjnych | "Etap 2 LLaVA" | Trenowanie na trójkach (obraz, instrukcja, odpowiedź) wygenerowanych przez GPT-4 |
| Wyrównanie etapu 1 | "Pretrening projektora" | Zamroź ViT i LLM, trenuj projektor ze stratą LM na podpisach |
| AnyRes | "Kafelkowanie wielowycinkowe" | Podziel obraz o wysokiej rozdzielczości na siatkę wycinków i konkatenuj tokeny wizyjne każdego wycinka |
| LLaVA-Instruct | "Wygenerowane przez GPT-4" | 158k par instrukcja-odpowiedź zsyntetyzowanych z podpisów COCO + GPT-4 |
| Zamrożenie enkodera wizyjnego | "Magazyn zablokowany" | Wagi CLIP nie aktualizują się w etapie 1, czasami też nie w etapie 2 |
| ShareGPT4V | "Lepsze podpisy" | 1M gęstych podpisów wygenerowanych przez GPT-4V, używanych do wyższej jakości wyrównania |
| VQA | "Wizyjne odpowiadanie na pytania" | Zadanie odpowiadania na swobodne pytanie o obraz |
| Prismatic VLMs | "Artykuł o przestrzeni projektowej" | Ablacja Karamcheti 2024 systematycznie testująca wybory projektora i danych |

## Dalsza Lektura

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — artykuł LLaVA.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5.
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — zbiór gęstych podpisów.
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — ablacje przestrzeni projektowej.
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — zunifikowany pojedynczy obraz, wiele obrazów, wideo.