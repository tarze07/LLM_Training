# ColPali i Natywnie Wizyjny RAG Dokumentowy

> Tradycyjny RAG parsuje PDF na tekst, dzieli na fragmenty, osadza fragmenty, przechowuje wektory. Każdy krok traci sygnał: OCR gubi dane z wykresów, dzielenie łamie wiersze tabel, osadzenia tekstowe ignorują rysunki. ColPali (Faysse i in., lipiec 2024) zadał prostsze pytanie: po co w ogóle wyodrębniać tekst? Osadź obraz strony bezpośrednio przez PaliGemma, użyj późnej interakcji w stylu ColBERT do wyszukiwania i zachowaj cały sygnał układu, rysunków, fontów i formatowania, który niesie dokument. Opublikowane benchmarki: 20–40% lepsza dokładność typu end-to-end niż text-RAG na dokumentach bogatych wizualnie. ColQwen2, ColSmol i VisRAG rozszerzyły ten wzorzec. Ta lekcja przedstawia tezę natywnie wizyjnego RAG i buduje mały indeksator podobny do ColPali.

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## Learning Objectives

- Wyjaśnić różnicę między wyszukiwaniem bi-enkoderowym (jeden wektor na dokument) a wyszukiwaniem z późną interakcją (wiele wektorów na dokument).
- Opisać operację MaxSim ColBERT i jak ColPali uogólnia ją z tokenów tekstowych na fragmenty obrazu.
- Zbudować mały indeksator podobny do ColPali: strona → osadzenia fragmentów → MaxSim nad osadzeniami tokenów zapytania → top-k stron.
- Porównać ColPali + generator Qwen2.5-VL z text-RAG + GPT-4 na przypadku użycia faktur / raportów finansowych.

## The Problem

Text-RAG na PDF-ach odrzuca większość dokumentu. Wzrost przychodów w Q3 w raporcie finansowym jest zwykle na wykresie; wyniki w raporcie medycznym są w oznaczonych obrazach; blok podpisu w kontrakcie prawnym to fakt układu, nie tekstu.

Potok text-RAG:

1. PDF → tekst przez OCR / pdftotext.
2. Tekst → fragmenty po 300–500 tokenów.
3. Fragment → osadzenie bi-enkodera (jeden wektor).
4. Zapytanie użytkownika → osadzenie → podobieństwo cosinusowe → top-k fragmentów.
5. Fragmenty + zapytanie → LLM.

Pięć stratnych kroków. Wykresy nieuchwycone. Tabele połamane na fragmenty. Układ wielokolumnowy spłaszczony. Adnotacje rysunków znikają.

Rozwiązanie ColPali: pomiń OCR, osadź obraz strony bezpośrednio. Użyj późnej interakcji w stylu ColBERT do wyszukiwania, aby model mógł skupić się na drobnoziarnistych fragmentach w czasie zapytania.

## The Concept

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) to metoda wyszukiwania tekstowego. Zamiast jednego wektora na dokument, produkuje jeden wektor na token. W czasie zapytania:

- Tokeny zapytania mają własne osadzenia (N_q wektorów).
- Tokeny dokumentu mają osadzenia (N_d wektorów, zwykle cachowane).
- Wynik = suma po tokenach zapytania z maksimum po tokenach dokumentu podobieństwa cosinusowego: Σ_i max_j cos(q_i, d_j).

To jest operacja MaxSim. Każdy token zapytania "wybiera" swój najlepiej pasujący token dokumentu. Ostateczny wynik to suma.

Zalety: silne recall, obsługuje semantykę na poziomie terminów. Wady: N_d wektorów na dokument, przechowywanie kosztowne.

### ColPali

ColPali (Faysse i in., arXiv:2407.01449) stosuje wzorzec ColBERT do obrazów.

- Każda strona jest kodowana przez PaliGemma (ViT + język) w osadzenia fragmentów: N_p wektorów na stronę.
- Każde zapytanie użytkownika (tekst) jest kodowane w osadzenia tokenów zapytania: N_q wektorów.
- Wynik = Σ_i max_j cos(q_i, p_j), czyli MaxSim nad tokenami tekstu zapytania i fragmentami obrazu strony.
- Pobierz top-k stron według całkowitego wyniku.

W czasie indeksowania dokumentu: osadź każdą stronę z PaliGemma, przechowuj wszystkie osadzenia fragmentów. W czasie zapytania: osadź tokeny zapytania, oblicz MaxSim względem wszystkich przechowywanych osadzeń stron, zwróć top-k stron.

Zalety: end-to-end pokonuje text-RAG o 20–40% na dokumentach bogatych wizualnie. Każdy wektor fragmentu wychwytuje lokalny układ i zawartość.

Wady: N_p fragmentów × 4-bajtowe floaty × D-wymiarowe wektory na stronę = przechowywanie szybko rośnie. Łagodzone przez kwantyzację PQ / OPQ.

### ColQwen2 i ColSmol

ColQwen2 (illuin-tech, 2024–2025) zamienia PaliGemma na Qwen2-VL. Lepszy enkoder bazowy, lepsze wyszukiwanie.

ColSmol to wariant o mniejszej skali do użytku lokalnego / brzegowego. Retriever ColSmol przy ~1B parametrów działa na konsumenckim GPU.

### VisRAG

VisRAG (Yu i in., arXiv:2410.10594) to inny wariant: zamiast MaxSim na fragmentach, agreguje każdą stronę w jeden wektor przez VLM, a następnie wyszukiwanie bi-enkoderowe. Szybsze indeksowanie + mniejsze przechowywanie, słabsze recall.

Kompromis jakość-koszt: ColPali dla jakości, VisRAG dla skali.

### M3DocRAG

M3DocRAG (Cho i in., arXiv:2411.04952) rozszerza multimodalne wyszukiwanie do rozumowania wielostronicowego i wielodokumentowego. Pobiera strony między dokumentami, komponuje wielostronicowy kontekst dla VLM.

### ViDoRe — benchmark

Towarzyszący benchmark ColPali. Visual Document Retrieval Evaluation. Zadania obejmują raporty finansowe, artykuły naukowe, dokumenty administracyjne, dokumentację medyczną, instrukcje. Metryka: nDCG@5.

ColPali-v1 osiąga ~80% nDCG@5 na ViDoRe; text-RAG na tych samych dokumentach osiąga ~50–60%.

### Potok RAG end-to-end

Dla natywnie wizyjnego RAG:

1. Indeksowanie: PDF → obrazy stron → kodowanie PaliGemma → przechowuj wszystkie osadzenia fragmentów.
2. Zapytanie: tekst użytkownika → osadzenia tokenów zapytania → MaxSim względem wszystkich indeksowanych stron → top-k stron.
3. Generowanie: obrazy top-k stron + zapytanie → VLM (Qwen2.5-VL lub Claude) → odpowiedź.

Nigdzie OCR. Rysunki, wykresy, fonty, układ — wszystko wpływa na odpowiedź.

### Matematyka przechowywania

50-stronicowy raport finansowy z 729 fragmentami na stronę i 128-wymiarowymi osadzeniami:

- ColPali: 50 * 729 * 128 * 4 bajty = ~18 MB surowo, ~4 MB po PQ.
- Text-RAG: 50 fragmentów * 768-wym * 4 bajty = ~150 kB.

ColPali to ~30x więcej przechowywania na dokument. Na skalę OPQ / PQ redukuje to do ~5–10x, zwykle akceptowalne.

### Kiedy text-RAG wciąż wygrywa

- Czysto tekstowe dokumenty bez sygnału układu (artykuły wiki, logi czatów). Text-RAG jest prostszy i tańszy w przechowywaniu.
- Archiwa wielomilionowych stron, gdzie koszt przechowywania dominuje.
- Ścisłe wymogi regulacyjne wymagające ekstrahowalnego tekstu OCR obok wyszukiwania.

Dla wszystkiego innego w 2026 — raportów finansowych, artykułów naukowych, kontraktów prawnych, dokumentacji medycznej, dokumentacji UX — natywnie wizyjny RAG wygrywa.

## Use It

`code/main.py`:

- Zabawkowy enkoder fragmentów: mapuje "stronę" (małą siatkę wektorów cech) na tablicę osadzeń fragmentów.
- Scorer MaxSim: oblicza wynik w stylu ColBERT między zbiorem osadzeń tokenów zapytania a zbiorem fragmentów strony.
- Indeksuje 5 zabawnych stron, uruchamia 3 zapytania, zwraca top-k z wynikami.

## Ship It

Ta lekcja produkuje `outputs/skill-vision-rag-designer.md`. Dla projektu RAG dokumentowego wybiera ColPali / ColQwen2 / VisRAG / text-RAG i szacuje przechowywanie.

## Exercises

1. 200-stronicowy raport roczny przy 729 fragmentach na stronę, osadzenia 128-wym, 4-bajtowe floaty. Oblicz surowe przechowywanie i skompresowane PQ (8x).

2. MaxSim to Σ_i max_j cos(q_i, p_j). Co ta suma wychwytuje, czego nie wychwytuje zwykłe średnie podobieństwo?

3. ColPali indeksuje strony jako zbiory fragmentów. Co się zmieni, jeśli będziemy indeksować na poziomie słów (jak ColBERT)? Kompromisy?

4. Zaprojektuj potok end-to-end dla korpusu 1 mln stron z budżetem opóźnienia 500 ms na zapytanie. Wybierz ColQwen2 / VisRAG i uzasadnij.

5. Przeczytaj M3DocRAG (arXiv:2411.04952). Opisz wzorzec uwagi wielostronicowej i czym różni się od jednostronicowego wyszukiwania ColPali.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColBERT-style" | Wyszukiwanie przy użyciu osadzeń per-token lub per-fragment + MaxSim, nie pojedynczego wektora dokumentu |
| MaxSim | "Max-over-patches" | Dla każdego tokena zapytania wybierz token dokumentu o najwyższym podobieństwie; suma po zapytaniu |
| Bi-encoder | "Single-vector" | Jeden wektor na dokument; szybszy, ale traci granularność |
| Multi-vector | "Many-vectors-per-doc" | Przechowuj N_p wektorów na dokument / stronę; koszt przechowywania rośnie, ale recall się poprawia |
| Patch embedding | "Page feature" | Jeden wektor na fragment obrazu z enkodera VLM, cachowany na stronę |
| ViDoRe | "Vision doc bench" | Zestaw benchmarków ColPali do wizyjnego wyszukiwania dokumentów |
| PQ quantization | "Product quantization" | Kompresja, która zachowuje podobieństwo wektorów, zmniejszając przechowywanie ~8x |

## Further Reading

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)