# Capstone 04 — Multimodalne QA Dokumentów (Pierwszoobrazowe PDF, Tabele, Wykresy)

> Granica QA dokumentów w 2026 roku odeszła od OCR-potem-tekst w kierunku pierwszoobrazowej późnej interakcji. ColPali, ColQwen2.5 i ColQwen3-omni traktują każdą stronę PDF jako obraz, osadzają ją z wielowektorową późną interakcją i pozwalają zapytaniu odnosić się bezpośrednio do fragmentów. Na finansowych 10-K, pracach naukowych i odręcznych notatkach ten wzór bije podejście OCR-first z dużą przewagą. Zbuduj potok od końca do końca na 10k stron i opublikuj porównanie obok podejścia OCR-potem-tekst.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 hours

## Problem

Firmy posiadają PDFy, które potoki OCR niszczą: skanowane 10-K z obróconymi tabelami, prace naukowe gęste od równań, wykresy które mają sens tylko jako obrazy, odręczne adnotacje. Traktowanie ich jako pierwszo-tekstowe oznacza utratę połowy sygnału. Odpowiedzią w 2026 roku jest wielowektorowe wyszukiwanie późnej interakcji na surowych obrazach stron. ColPali (Illuin Tech) wprowadził je; ColQwen2.5-v0.2 i ColQwen3-omni zwiększyły dokładność. Na ViDoRe v3, pierwszoobrazowe wyszukiwanie uzyskuje wyniki przewyższające OCR-potem-tekst o znaczące marginesy — a luka powiększa się na wykresach, tabelach i odręcznym piśmie.

Kompromisem jest pamięć masowa i opóźnienie. Osadzenie ColQwen to ~2048 wektorów fragmentów na stronę, nie pojedynczy 1024-wymiarowy wektor. Surowa pamięć masowa eksploduje. DocPruner (2026) zapewnia 50% przycięcia bez mierzalnej utraty dokładności. Zaindeksujesz 10k stron, zmierzysz ViDoRe v3 nDCG@5, dostarczysz odpowiedzi poniżej 2s i porównasz bezpośrednio z baseline OCR-potem-tekst.

## Koncepcja

Późna interakcja oznacza, że każdy token zapytania punktuje przeciwko każdemu tokenowi fragmentu, a maksymalny wynik na token zapytania jest sumowany. Otrzymujesz szczegółowe dopasowanie bez potrzeby pojedynczego połączonego wektora. Wielowektorowy indeks (Vespa, Qdrant multi-vector lub AstraDB) przechowuje osadzenia per-fragment i uruchamia MaxSim w czasie wyszukiwania.

Odpowiadacz to model wizyjno-językowy, który przyjmuje zapytanie plus najlepsze k stron jako obrazy i pisze odpowiedź z regionami dowodów (prostokąty ograniczające lub odniesienia do stron). Qwen3-VL-30B, Gemini 2.5 Pro i InternVL3 to graniczne wybory w 2026 roku. Dla równań i notacji naukowej, zapasowy OCR (Nougat, dots.ocr) jest wpleciony jako opcjonalny kanał tekstowy.

Ewaluacja to dwuwymiarowa macierz. Jedna oś: typ treści (zwykłe akapity tekstu, gęste tabele, wykresy słupkowe/liniowe, odręczne notatki, równania). Druga oś: podejście do wyszukiwania (pierwszoobrazowa późna interakcja vs OCR-potem-tekst vs hybryda). Każda komórka otrzymuje nDCG@5 i dokładność odpowiedzi. Raport jest rezultatem.

## Architektura

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## Stack

- Renderowanie stron: PyMuPDF (fitz) przy 180 DPI, znormalizowane do portretu
- Model późnej interakcji: ColQwen2.5-v0.2 lub ColQwen3-omni (zespół vidore na Hugging Face)
- Indeks: Vespa z polem wielowektorowym, lub Qdrant multi-vector, lub AstraDB z MaxSim
- Przycinanie: DocPruner 2026 policy (zachowaj fragmenty o wysokiej wariancji, 50% kompresji przy < 0.5% utraty dokładności)
- Zapasowy OCR (równania / gęste tabele): dots.ocr lub Nougat
- VLM odpowiadacz: Qwen3-VL-30B samodzielnie hostowany lub Gemini 2.5 Pro hostowany; InternVL3 jako zapasowy
- Ewaluacja: ViDoRe v3 benchmark, M3DocVQA dla wielostronicowego wnioskowania
- UI przeglądarki: Next.js 15 z nakładką canvas dla regionów dowodów

## Build It

1. **Indeksacja.** Przejdź korpus 10k stron PDF przez 10-K, prace naukowe i skanowane dokumenty. Wyrenderuj każdą stronę do PNG 1536x2048. Zapisz `{doc_id, page_num, image_path}`.

2. **Osadź.** Uruchom ColQwen2.5-v0.2 na każdej stronie. Kształt wyjścia ~2048 osadzeń fragmentów o dim 128. Zastosuj DocPruner, aby zachować połowę o najwyższym sygnale. Zapisz do pola wielowektorowego Vespa lub Qdrant multi-vector.

3. **Zapytanie.** Dla każdego przychodzącego zapytania, osadź wieżą zapytania (osadzenia na poziomie tokenów). Uruchom MaxSim przeciw indeksowi: dla każdego tokena zapytania, weź maksymalny iloczyn skalarny po osadzeniach fragmentów strony, zsumuj. Zwróć top-k stron.

4. **Syntezuj.** Wywołaj Qwen3-VL-30B z zapytaniem i 5 najlepszymi stronami. Prompt: "Odpowiedz używając tylko dostarczonych stron. Cytuj każde twierdzenie przez (doc_id, page) i nazwij region (figure, table, paragraph)."

5. **Regiony dowodów.** Po-processuj odpowiedź, aby wyodrębnić cytowane regiony. Jeśli VLM emituje prostokąty ograniczające (Qwen3-VL to robi), wyrenderuj je jako nakładki w przeglądarce.

6. **Zapasowy OCR.** Dla stron zidentyfikowanych jako gęste od równań (heurystyka na wariancji obrazu), uruchom Nougat lub dots.ocr i przekaż tekst OCR jako dodatkowy kanał obok obrazu.

7. **Ewaluacja.** Uruchom ViDoRe v3 (retrieval nDCG@5) i M3DocVQA (wielostronicowa dokładność QA). Uruchom także potok OCR-potem-tekst na tym samym korpusie z tym samym syntezatorem. Wyprodukuj macierz typ treści × podejście.

8. **UI.** Najpierw prototyp Streamlit; Next.js 15 produkcyjna przeglądarka z nakładką regionów dowodów strona po stronie.

## Use It

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## Ship It

`outputs/skill-doc-qa.md` opisuje rezultat: pierwszoobrazowy multimodalny system QA dokumentów dostrojony do konkretnego korpusu i oceniony względem baseline OCR-potem-tekst na ViDoRe v3.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Dokładność ViDoRe v3 / M3DocVQA | Liczby benchmarkowe vs baseline OCR-tekst i opublikowana tabela liderów |
| 20 | Ugruntowanie regionów dowodów | Frakcja cytowanych regionów, które faktycznie zawierają zakres odpowiedzi |
| 20 | Inżynieria pamięci masowej i opóźnień | Współczynnik kompresji DocPruner, p95 indeksu, p95 odpowiedzi |
| 20 | Wielostronicowe wnioskowanie | Dokładność na ręcznie oznaczonym 100-pytaniowym zestawie wielostronicowym |
| 15 | UX inspekcji źródła | Przejrzystość przeglądarki, wierność nakładek, narzędzia do porównania obok siebie |
| **100** | | |

## Ćwiczenia

1. Zmierz ColQwen2.5-v0.2 vs ColQwen3-omni na tym samym korpusie. Które strony jeden dostaje dobrze, a drugi przegapia? Dodaj tag "klasa treści" do indeksu, aby kierować według typu.

2. Przytnij osadzenia agresywnie (75%, 90%). Znajdź klif kompresji: punkt, w którym ViDoRe nDCG@5 spada poniżej baseline OCR.

3. Zbuduj hybrydę: uruchom OCR-potem-tekst i ColQwen równolegle, połącz z RRF, posegreguj cross-encoderem. Czy hybryda bije każdą z osobna? Gdzie pomaga najbardziej?

4. Zamień Qwen3-VL-30B na mniejszy VLM (Qwen2.5-VL-7B). Zmierz krzywą dokładność-na-dolara.

5. Dodaj obsługę odręcznych notatek. Wyrenderuj korpus odręczny, osadź z ColQwen, zmierz wyszukiwanie. Porównaj z potokiem OCR odręcznego pisma.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Late interaction | "Wyszukiwanie w stylu ColPali" | Tokeny zapytania punktują przeciwko fragmentom strony niezależnie; MaxSim agreguje |
| Multi-vector | "Osadzenie per-fragment" | Każdy dokument ma wiele wektorów, nie jeden połączony wektor |
| MaxSim | "Punktacja późnej interakcji" | Dla każdego tokena zapytania, weź maksymalne podobieństwo po wektorach dokumentu; zsumuj |
| DocPruner | "Kompresja fragmentów" | Przycinanie 2026, które zachowuje 50% fragmentów z pomijalną utratą dokładności |
| ViDoRe v3 | "Benchmark wyszukiwania dokumentów" | Standard 2026 do mierzenia wyszukiwania wizualnych dokumentów |
| Evidence region | "Cytowany prostokąt ograniczający" | Bbox na stronie źródłowej, który lokalizuje zakres odpowiedzi |
| OCR fallback | "Kanał równań" | Potok tekstowy używany obok wizji dla stron bogatych w równania lub tabele |

## Dalsza lektura

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — reference late-interaction doc retrieval
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — the foundational method paper
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — production-ready checkpoints
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — multi-page multimodal RAG baseline
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — reference serving stack
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — alternate index
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — alternate managed index
- [Nougat OCR](https://github.com/facebookresearch/nougat) — equation-capable OCR fallback