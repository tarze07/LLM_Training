# Rozumienie Długich Wideo w Kontekście Miliona Tokenów

> 1-godzinne wideo 4K przy 24 FPS, po podziale na łatki i osadzeniu, produkuje około 60 milionów tokenów. 2-godzinny odcinek podcastu po transkrypcji to 30 000 tokenów. Pełny film Blu-ray, nawet po kompresji z agresywnym poolingiem, to setki tysięcy tokenów. Gemini 1.5 Google'a (marzec 2024) otworzyło tę erę z kontekstem 10 milionów tokenów, osiągając niezawodne przywołanie igły-w-stogu-siana w godzinnych wideo. LWM (Liu i in., luty 2024) pokazało ścieżkę skalowania ring attention. LongVILA i Video-XL skalowały dalej proces wczytywania. VideoAgent zamieniło surowy kontekst na wyszukiwanie agencyjne. Każde podejście to inny kompromis między mocą obliczeniową, skutecznością przywołania i złożonością inżynieryjną. Ta lekcja opisuje je obok siebie.

**Type:** Build
**Languages:** Python (stdlib, symulator igły-w-stogu-siana + router wyszukiwania agencyjnego)
**Prerequisites:** Phase 12 · 17 (video temporal tokens)
**Time:** ~180 minutes

## Learning Objectives

- Obliczyć całkowitą liczbę tokenów wizyjnych dla długiego wideo przy różnych FPS i poolingu.
- Wyjaśnić trzy ścieżki skalowania: brute context (Gemini 1.5), ring attention (LWM), kompresja tokenów (LongVILA / Video-XL).
- Porównać surowe wideo VLM z kontekstem vs agencyjne wideo VLM z wyszukiwaniem (VideoAgent) pod kątem dokładności i opóźnienia.
- Zaprojektować test igły-w-stogu-siana dla 30-minutowego wideo i zmierzyć skuteczność przywołania w konkretnej minucie.

## Problem

Pojedyncza klatka łat o rozmiarze Qwen2.5-VL przy natywnej rozdzielczości 384 to ~729 tokenów. Przy poolingu 3x3 to 81 tokenów na klatkę. 30-minutowy klip przy 1 FPS = 1800 klatek = 145 800 tokenów. Wykonalne dla otwartych VLM z 2025 roku, ale ciasno. Przy 2 FPS, 291 600 tokenów — tylko największe konteksty się mieszczą.

2-godzinny film przy 1 FPS to 583k tokenów. Poza zasięgiem większości otwartych modeli z 2026; wymaga Gemini 2.5 Pro lub bardziej agresywnego poolingu.

Pojawiły się trzy ścieżki skalowania.

## Koncepcja

### Ścieżka 1: Brute context (Gemini 1.5, Claude Opus)

Rzuć sprzęt w problem. Skaluj kontekst do milionów tokenów, przetwarzaj wszystko w jednym przebiegu forward.

Gemini 1.5 Pro wystartowało z 1M tokenów; Gemini 1.5 Ultra do 10M; Gemini 2.5 Pro w 2026 robi godziny wideo niezawodnie. Praca (arXiv:2403.05530) dokumentuje przywołanie igły-w-stogu-siana na poziomie 99.7% do ~9.5M tokenów.

Inżynieria: niestandardowa implementacja attention z hierarchią pamięci (lokalna + globalna + sparse) plus routowanie ekspertów MoE dla wydajności długiego kontekstu. Nie opublikowane w pełnym szczególe. Nie open-source.

### Ścieżka 2: Ring attention (LWM, LongVILA)

Ring attention rozdziela długie sekwencje między urządzenia w "pierścień", gdzie każde urządzenie przechowuje fragment. Attention na całej sekwencji odbywa się przez wysyłanie fragmentu do następnego urządzenia w wzorcu pierścienia, obliczanie częściowego attention i agregowanie.

LWM (Liu i in., 2024) trenował model z kontekstem 1M tokenów w ten sposób. Koszt obliczeniowy trenowania skaluje się liniowo z kontekstem, nie kwadratowo — kwadratowy wpływ na attention jest zamortyzowany między urządzeniami pierścienia.

LongVILA (arXiv:2408.10188) zaadaptował wzorzec do VLM. 1400-klatkowe wideo przy 192 tokenach na klatkę = 268k kontekstu, trenowane z ring attention przy 8-kierunkowej równoległości.

### Ścieżka 3: Kompresja tokenów (Video-XL, LongVA)

Tańsze niż brute context: kompresuj agresywnie, zanim LLM zobaczy sekwencję.

Video-XL (arXiv:2409.14485) używa wizyjnego tokena podsumowującego: każdy klip N klatek produkuje pojedynczy token "podsumowania", który attenduje po N klatkach. Przy inferencji LLM widzi jeden token podsumowania na klip, drastycznie zmniejszając kontekst.

LongVA rozszerza kontekst LLM z 200k do 2M przez technikę "long context transfer". Trenuj na długim kontekście tekstowym, przenieś na długi kontekst wideo przez współdzieloną reprezentację.

Kompresja tokenów wymienia dokładność przywołania w konkretnych znacznikach czasu na skalowalność. Model wie ogólnie, co się stało, ale czasami gubi konkretne klatki.

### Ścieżka 4: Wyszukiwanie agencyjne (VideoAgent)

Nie podawaj całego wideo do LLM. Zamiast tego traktuj wideo jako bazę danych i użyj LLM do jej przeszukiwania.

VideoAgent (arXiv:2403.10517):

1. LLM czyta pytanie.
2. LLM pyta narzędzie wyszukiwania o odpowiednie klipy ("pokaż segmenty z kotem").
3. Narzędzie zwraca pasujące znaczniki czasu klipów.
4. LLM czyta te klipy przez VLM.
5. LLM komponuje odpowiedź lub zadaje pytania uzupełniające.

To wzorzec LLM-jako-agent zastosowany do długiego wideo. Tańsza inferencja (tylko odpowiednie klipy są kodowane), trudniejsza inżynieria (jakość wyszukiwania staje się wąskim gardłem).

### Benchmarki igły-w-stogu-siana

Standardowy test długiego kontekstu: wstaw unikalny znacznik wizyjny lub tekstowy w losowym punkcie wideo, a następnie zadaj pytanie wymagające jego przywołania.

Metryka: Recall@k na długości wideo i pozycji znacznika.

Gemini 2.5 Pro osiąga >99% przywołania w wideo do 90 minut. Otwarte modele 72B (Qwen2.5-VL-72B, InternVL3-78B) osiągają ~85-90% przy 30 minutach i degradują się po 60.

VideoAgent może dorównać lub pobić modele z surowym kontekstem przy 2+ godzinach, ponieważ wyszukiwanie trafia igłę, jeśli narzędzie jest dobre.

### Którą ścieżkę wybrać

Dla 15-minutowego klipu z dokładnością na poziomie granicznym: otwarty 72B + natywny kontekst zwykle działa. Wybierz Qwen2.5-VL-72B.

Dla treści 30-minutowych do 1 godziny: LongVILA lub Video-XL dla otwartych; Gemini 2.5 Pro dla zamkniętych. Próg jakości ma znaczenie — modele graniczne idą w stronę zamkniętą.

Dla treści 2+ godzin: VideoAgent lub podobne wzorce wyszukiwania. Alternatywnie, podsumuj do mniejszych fragmentów i podawaj hierarchiczne podsumowania.

### Wzorzec produkcyjny 2026

W praktyce produkcyjne potoki długiego wideo są hybrydowe:

1. Uruchom próbkowanie dynamic-FPS + agresywny pooling na całym wideo (uzyskaj globalną reprezentację 100k tokenów).
2. Przekaż do 72B VLM dla globalnego podsumowania.
3. Jeśli użytkownik zadaje szczegółowe pytania, uruchom wyszukiwanie agencyjne używając podsumowania jako indeksu.

To łączy brute context dla globalnego zrozumienia i wyszukiwanie dla lokalnych szczegółów.

## Use It

`code/main.py`:

- Oblicza budżety tokenów dla wideo od 1 minuty do 3 godzin przy różnych FPS + poolingu.
- Symuluje przebieg igły-w-stogu-siana: wstaw znacznik w losowym znaczniku czasu, zadaj pytanie, oceń przywołanie.
- Zawiera symulator routera wyszukiwania agencyjnego, który wybiera konkretne klipy do przekazania downstream VLM.

Uruchom tabelę budżetu i poczuj lukę skali.

## Ship It

Ta lekcja produkuje `outputs/skill-long-video-strategy-planner.md`. Na podstawie czasu trwania wideo i złożoności zapytania wybiera między brute context, kompresją i wyszukiwaniem agencyjnym oraz oblicza oczekiwane opóźnienie i jakość.

## Exercises

1. 45-minutowy wykład przy 1 FPS, 81 tokenów na klatkę. Łączna liczba tokenów? Mieści się w kontekście których modeli?

2. Zaprojektuj test igły-w-stogu-siana: w której minucie wstawiasz znacznik i jaki jest dokładny format zapytania?

3. Porównaj Qwen2.5-VL-72B z brute context (80k kontekstu) z VideoAgent (Claude 3.5 + wyszukiwanie) na 1-godzinnym wideo. Który wygrywa na przywołaniu? Który na opóźnieniu?

4. Koszt pamięci ring attention skaluje się liniowo z długością sekwencji i liniowo z liczbą urządzeń. Wyjaśnij dlaczego i co się psuje, jeśli pominiesz fazę rotacji pierścienia.

5. Przeczytaj sekcję 5 Gemini 1.5 na temat igły-w-stogu-siana. Co praca odkryła na temat przywołania na granicy 1M vs 10M tokenów?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Brute context | "Po prostu więcej tokenów" | Skaluj kontekst LLM do milionów tokenów; przetwarzaj wszystko w jednym przebiegu |
| Ring attention | "Równoległość w stylu LWM" | Rozproszony wzorzec attention, gdzie każde urządzenie przechowuje fragment i rotuje |
| Token compression | "Tokeny podsumowujące" | Redukuj tokeny na klip przez wyuczony kompresor przed LLM |
| Needle-in-haystack | "Test NIH" | Wstaw unikalny znacznik w losowym punkcie, poproś model o jego przywołanie w czasie testu |
| Agentic retrieval | "LLM jako planista zapytań" | LLM pyta narzędzie wyszukiwania o odpowiednie klipy, czyta je przez VLM, komponuje odpowiedź |
| VideoAgent | "Wzorzec wyszukiwania dla wideo" | Kanoniczny projekt wyszukiwania agencyjnego: pytanie -> narzędzie -> klip -> odpowiedź |

## Further Reading

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)