# InternVL3: Natywny Multimodalny Wstępny Trening

> Każdy otwarty VLM przed InternVL3 podążał za tym samym trzyetapowym przepisem: weź tekstowy LLM wytrenowany na bilionach tokenów tekstu, dołącz enkoder wizyjny, a następnie dostrój połączenia. To działa, ale ma dług dopasowania — tekstowy LLM wydał cały swój budżet wstępnego treningu na czysty tekst i nie rozumie natywnie tokenów wizualnych. Gdy dodajesz wizję post-hoc, LLM musi na nowo nauczyć się, jak odnosić wejście wizualne do swojego rozumowania tekstowego bez zapominania tekstu. InternVL3 (Zhu et al., kwiecień 2025) odrzuca podejście post-hoc: jeden przebieg wstępnego treningu, tekst i dane multimodalne przeplatane od pierwszego kroku. Wynik dorównuje Gemini 2.5 Pro na MMMU-Pro przy 78B parametrów jako otwarty model. Ta lekcja czyta argumenty za natywnym wstępnym treningiem i co się zmienia, gdy go zastosujesz.

**Type:** Learn
**Languages:** Python (stdlib, mikser korpusu treningowego)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 minutes

## Learning Objectives

- Wyjaśnić, dlaczego trening VLM post-hoc gromadzi dług dopasowania, cytując trzy mierzalne objawy (katastroficzne zapominanie, dryf odpowiedzi, niespójność wizualno-tekstowa).
- Opisać natywną mieszankę korpusu wstępnego treningu InternVL3 i dlaczego proporcja tekstu : przeplatanych : podpisów ma znaczenie.
- Porównać V2PE (zmienne wizualne kodowanie pozycyjne) z M-RoPE Qwen2-VL.
- Wymienić optymalizacje wdrożeniowe Wizualnego Routera Rozdzielczości (ViR) i Rozdzielonego Wizyjno-Językowego (DvD).

## The Problem

Trening VLM post-hoc jest domyślny. LLaVA, BLIP-2, Qwen-VL, Idefics — wszystkie biorą już wstępnie wytrenowany LLM (Llama, Vicuna, Qwen, Mistral) i dodają wizję. Etapy treningu zazwyczaj wyglądają tak:

1. Zamrożony LLM + zamrożony enkoder wizyjny + trenowalny projektor, trenowane na parach podpisów, aby dopasować osadzenia.
2. Odmroź LLM, trenuj na danych instrukcyjnych (LLaVA-Instruct, ShareGPT4V).
3. Opcjonalne dostrojenie specyficzne dla zadania.

Trzy objawy długu dopasowania ujawniają się:

- Katastroficzne zapominanie. VLM post-hoc zapomina umiejętności czysto tekstowych. Wyniki GSM8K spadają o 5-10 punktów. Wyniki Hellaswag spadają. Agenci czysto tekstowi regresują.
- Dryf odpowiedzi. Drobne sformułowania tego samego pytania wizualnego dają różne odpowiedzi. Enkoder wizyjny łączy się z LLM słabszymi wiązaniami niż własne tokeny LLM.
- Niespójność wizualno-tekstowa. VLM może poprawnie opisać obraz, a następnie odpowiedzieć na pytanie przeczące własnemu opisowi. Tokeny wizualne nie uczestniczą w wewnętrznych kontrolach spójności LLM w taki sam sposób jak tekst.

Te objawy są dobrze udokumentowane. MM1.5 Sekcja 4 je kwantyfikuje. Ablacje LLaVA-OneVision na nie wskazują. Natywny wstępny trening jest odpowiedzią.

## The Concept

### Natywny multimodalny wstępny trening

InternVL3 trenuje od zera na korpusie, który jest natywnie multimodalny od pierwszego kroku. Mieszanka to:

- 40% danych czysto tekstowych (FineWeb, Proof-Pile-2, itd.)
- 35% przeplatanych danych obraz-tekst (OBELICS, styl MMC4)
- 20% sparowanych danych obraz-podpis
- 5% danych wideo-tekst

Tokeny wizualne, tokeny tekstowe i interakcje między-modalne uczestniczą w tej samej stracie od pierwszego kroku gradientu. Żadnego wstępnego treningu dopasowania, żadnego etapu zamrażania projektora, żadnego katastroficznego zapominania do odzyskiwania.

Trening to pojedynczy etap dla modelu bazowego. Dostrajanie instrukcyjne następuje później, ale model bazowy już rozumie tokeny wizualne jako obywateli pierwszej kategorii.

### V2PE (zmienne wizualne kodowanie pozycyjne)

Qwen2-VL używa M-RoPE ze stałą alokacją osi. InternVL3 wprowadza V2PE: kodowanie pozycyjne zmienia się w zależności od typu modalności (tekst, obraz, wideo) z możliwością uczenia skalowania. W praktyce:

- Tokeny tekstowe dostają 1D pozycję (indeks tekstu).
- Patche obrazu dostają 2D pozycję (wiersz, kolumna).
- Klatki wideo dostają 3D pozycję (czas, wiersz, kolumna).

Trzy dzielą tę samą bazę częstotliwości RoPE, ale alokacja ukrytego wymiaru na pasmo jest nauczonym parametrem, a nie stałym podziałem. Swoboda wymiany rozdzielczości czasowej vs przestrzennej częstotliwości podczas wstępnego treningu.

Twierdzenie ablacyjne V2PE: 1-2 punkty na benchmarkach wideo nad M-RoPE przy tej samej mocy obliczeniowej. Nie rewolucja, ale czyściej.

### Wizualny Router Rozdzielczości (ViR)

Optymalizacja wdrożeniowa. Nie wszystkie obrazy potrzebują kodowania w pełnej rozdzielczości. Zdjęcie z jednym obiektem o niskiej szczegółowości marnuje tokeny, gdy jest kodowane w natywnych 1280px. ViR to mały klasyfikator, który przewiduje minimalną rozdzielczość potrzebną do odpowiedzi na pytanie, przed kodowaniem.

Routing ma trzy poziomy: niska rozdzielczość (256 tokenów), średnia (576), wysoka (2048+). Dla 60% zapytań w ruchu produkcyjnym niska lub średnia jest wystarczająca. Efekt netto: 2-3x przepustowość przy równej jakości.

### Rozdzielone Wizyjno-Językowe wdrożenie (DvD)

Gdy obsługujesz duży VLM, enkoder wizyjny działa raz na obraz, ale LLM działa autoregresywnie dla każdego tokena wyjściowego. Te dwa komponenty mają różne wąskie gardła (wizja = przepustowość pamięci GPU dla konwolucji + uwagi; LLM = pamięć podręczna KV). DvD dzieli je na osobne GPU z przesyłaniem strumieniowym między nimi.

Dla modelu 8B + 400M enkodera, DvD mniej więcej podwaja przepustowość na węzeł w porównaniu do współlokacji.

### Jakość jednoetapowa vs wieloetapowa

Główne twierdzenie benchmarkowe InternVL3: przy 78B parametrach, dorównaj Gemini 2.5 Pro na MMMU-Pro. Przy 38B, dorównaj GPT-4o. Przy 8B, prowadź ranking otwartych 8B. Wszystko na jednoetapowym przepisie wstępnego treningu + dostrajania instrukcyjnego.

Hipoteza długu dopasowania jest mierzalna: InternVL3-8B traci mniej punktów na benchmarkach tekstowych (MMLU, GSM8K) niż Qwen2.5-VL-7B na jednostkę zysku na benchmarku wizyjnym. Model jest bardziej generalistą, ponieważ trening był jedną częścią, nie dwiema.

### InternVL3.5 i InternVL-U

InternVL3.5 (sierpień 2025) skaluje przepis. To samo natywne podejście do wstępnego treningu, więcej danych, więcej parametrów. Ulepszenia MMMU są przyrostowe.

InternVL-U (2026) dodaje ujednolicone generowanie — wyjście obrazu przez głowice MMDiT na tym samym szkielecie. "U" oznacza "Understanding + generation" (rozumienie + generowanie), goniąc za modelami ujednoliconymi w stylu Transfusion (Lekcja 12.13). Ten sam natywny szkielet wstępnego treningu obsługuje zarówno głowice rozumienia, jak i generowania.

### Kompromisy natywnego wstępnego treningu

Natywny wstępny trening nie jest darmowy:

- Moc obliczeniowa. Trenowanie nowego VLM od zera kosztuje tyle samo, co trenowanie tekstowego LLM — miliony godzin GPU. Adaptacja post-hoc wykorzystuje istniejące wagi LLM, oszczędzając większość kosztu.
- Dane. Przeplatane korpusy obraz-tekst na dużą skalę są rzadkie. OBELICS to 141M dokumentów; MMC4 to 571M. Sam tekst dostarcza 15T tokenów. Niedobór danych do multimodalnego wstępnego treningu jest twardym ograniczeniem.
- Ponowne użycie bazowego LLM. Natywny wstępny trening rezygnuje z opcji wrzucenia nowego LLM później. Post-hoc pozwala zamienić Llama-3.1 na Llama-4 przez przekwalifikowanie tylko adaptera.

Zakład InternVL3: dług dopasowania jest gorszy niż strata możliwości ponownego użycia. Benchmarki wspierają to twierdzenie. Koszt produkcji uniemożliwia przyszłym laboratoriom tanie replikowanie. VLM post-hoc będą nadal istnieć, ponieważ pozostają tańsze dla większości projektów.

## Use It

`code/main.py` to mikser korpusu treningowego i symulator routera ViR.:

- Przyjmuje docelową mieszankę korpusu (%tekst, %przeplatane, %podpisy, %wideo) i oblicza oczekiwane kroki na modalność.
- Symuluje routing ViR na partii zapytań (dystrybucja: 50% nisko-szczegółowe, 30% średnio, 20% wysokoszczegółowe) i raportuje średnią liczbę tokenów.
- Raportuje oszacowania przepustowości DvD na podstawie FLOP-ów enkodera vs LLM.
- Wypisuje porównanie obok siebie post-hoc vs natywnego wstępnego treningu pod względem parametrów, mocy obliczeniowej, danych i oczekiwanych objawów długu dopasowania.

## Ship It

Ta lekcja produkuje `outputs/skill-native-vs-posthoc-auditor.md`. Mając proponowany plan treningu VLM, audytuje, czy wybrać podejście natywne czy post-hoc, flaguje ryzyko długu dopasowania i rekomenduje mieszankę korpusu. Użyj tego, gdy wymiarujesz nowy projekt otwartego VLM i potrzebujesz wybrać strategię treningu.

## Exercises

1. Oszacuj różnicę obliczeniową między InternVL3-8B (natywny wstępny trening) a LLaVA-OneVision-7B (post-hoc). Stosunek godzin GPU w przybliżeniu? Co wyjaśnia lukę?

2. InternVL3 raportuje 40% tekst / 35% przeplatane / 20% podpisy / 5% wideo. Jeśli twoje docelowe zadanie jest intensywne wideo, zaproponuj nowy stosunek i uzasadnij, dlaczego model bazowy nadal potrzebuje znacznych danych tekstowych i podpisów.

3. Przeczytaj MM1.5 Sekcja 4 o zapominaniu. Wymień dokładny benchmark, gdzie trening post-hoc pokazał największą regresję. Ile kosztowała regresja?

4. ViR kieruje 60% ruchu do kodowania niskiej rozdzielczości. Jakie rodzaje zapytań błędnie kieruje (wysyła do niskiej rozdzielczości, gdy wysoka była potrzebna)? Zaproponuj trzy tryby awarii routera.

5. DvD dzieli wizję i LLM na osobne GPU. W jakim wzorcu ruchu DvD szkodzi przepustowości zamiast pomagać?

## Key Terms

| Term | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Native multimodal pretraining | "Od zera razem" | Tokeny tekstu + obrazu + wideo uczestniczą w stracie od kroku 1, nie dodane później |
| Alignment debt | "Kara post-hoc" | Mierzalna regresja umiejętności tekstowych i spójności odpowiedzi wynikająca z dołączania wizji do zamrożonego LLM |
| V2PE | "Zmienne wizualne kodowanie pozycyjne" | Nauczalna alokacja kodowania pozycyjnego na modalność; następca M-RoPE InternVL3 |
| ViR | "Router rozdzielczości" | Mały klasyfikator wybierający minimalną rozdzielczość potrzebną na zapytanie przed kodowaniem, oszczędzający tokeny wnioskowania |
| DvD | "Rozdzielone wdrożenie" | Enkoder wizyjny na jednym GPU, LLM na drugim, z przekazaniem strumieniowym; podwaja przepustowość dla dużych VLM |
| InternVL-U | "Ujednolicone rozumienie + generowanie" | Kontynuacja z 2026 dodająca głowice generowania obrazu do natywnego szkieletu wstępnego treningu |
| Interleaved corpus | "OBELICS / MMC4" | Dokumenty z tekstem i obrazami w naturalnej kolejności czytania; surowy materiał dla natywnego wstępnego treningu |

## Further Reading

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)