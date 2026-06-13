# Rozumienie Dokumentów i Diagramów

> Dokumenty to nie zdjęcia. Plik PDF, artykuł naukowy, faktura lub odręczny formularz mają układ, tabele, diagramy, przypisy, nagłówki i strukturę semantyczną, której zwykłe rozumienie obrazu nie jest w stanie uchwycić. Stack sprzed VLM to był potok: Tesseract OCR + LayoutLMv3 + heurystyki ekstrakcji tabel. Fala VLM zastąpiła go modelami bez OCR — Donut (2022), Nougat (2023), DocLLM (2023) — które emitują bezpośrednio strukturalne znaczniki. Do 2026 roku granicą jest po prostu "podaj obraz strony Claude'owi Opus 4.7 w natywnej rozdzielczości 2576px", a wyjście w postaci strukturalnych znaczników pojawia się samoistnie. Ta lekcja przedstawia trzyeryczny łuk rozwoju AI dokumentowej.

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## Learning Objectives

- Wyjaśnić trzy ery AI dokumentowej: potok OCR, bez OCR, natywnie VLM.
- Opisać trzy strumienie wejściowe LayoutLMv3: tekst, układ (bbox), fragmenty obrazu, z ujednoliconą maskowaniem.
- Porównać Donut (bez OCR, obraz → znaczniki), Nougat (artykuł naukowy → LaTeX), DocLLM (generatywny z uwzględnieniem układu), PaliGemma 2 (natywnie VLM).
- Wybrać model dokumentowy dla nowego zadania (faktury, artykuły naukowe, odręczne formularze, chińskie paragony).

## The Problem

"Zrozum ten PDF" jest podstępnie trudne. Informacja znajduje się w:

- Treści tekstowej (90% sygnału).
- Układzie (nagłówki, przypisy, paski boczne, format dwukolumnowy).
- Tabelach (wiersze, kolumny, scalone komórki).
- Rysunkach i diagramach.
- Odręcznych adnotacjach.
- Fontach i typografii (tytuł vs treść).

Surowe OCR wyciąga tekst i traci resztę. System, który dba o faktury, musi wiedzieć, że "Suma: 1245 zł" pochodzi z prawego dolnego rogu, a nie z przypisu.

## The Concept

### Era 1 — Potok OCR (przed 2021)

Klasyczny stack:

1. PDF → obraz na stronę.
2. Tesseract (lub komercyjne OCR) wyodrębnia tekst z ramkami per słowo.
3. Analizator układu identyfikuje bloki (nagłówek, tabela, akapit).
4. Rozpoznawca struktury tabeli parsuje tabele.
5. Reguły domenowe + regex wyodrębniają pola.

Działa dla czystego drukowanego tekstu. Psuje się na odręcznym piśmie, przekrzywionych skanach, złożonych tabelach, skryptach nieanglojęzycznych. Każdy tryb awarii wymaga niestandardowej ścieżki wyjątku.

### TrOCR (2021)

TrOCR (Li i in., arXiv:2109.10282) zastąpił klasyczny CNN-CTC z Tesseracta transformerem kodera-dekodera trenowanym na syntetycznych + rzeczywistych obrazach tekstu. Czyste zwycięstwo w odręcznym i wielojęzycznym tekście. Wciąż potok (detektor → TrOCR → układ), ale krok OCR znacznie się poprawił.

### Era 2 — Bez OCR (2022–2023)

Pierwsze modele bez OCR powiedziały: pomiń całkowicie detekcję, mapuj piksele obrazu bezpośrednio na strukturalne wyjście.

Donut (Kim i in., arXiv:2111.15664):
- Transformer koder-dekoder, koder to Swin-B.
- Wyjście to JSON dla rozumienia formularzy, markdown dla podsumowań, lub dowolny schemat specyficzny dla zadania.
- Żadnego OCR, żadnego układu, żadnej detekcji.

Nougat (Blecher i in., arXiv:2308.13418):
- Trenowany specyficznie na artykułach naukowych.
- Wyjście to LaTeX / markdown.
- Obsługuje równania, układ wielokolumnowy, rysunki.
- Model, którego wywołuje każdy parser arXiv.

To specjaliści, nie generaliści. Donut na artykule naukowym zawodzi; Nougat na fakturze zawodzi.

### LayoutLMv3 (2022)

Inna ścieżka. LayoutLMv3 (Huang i in., arXiv:2204.08387) zachowuje OCR, ale dodaje rozumienie układu:

- Trzy strumienie wejściowe: tokeny tekstowe OCR, 2D ramki per token, fragmenty obrazu.
- Maskowany cel treningowy we wszystkich trzech modalnościach (maskowany tekst, maskowane fragmenty, maskowany układ).
- Zastosowania: klasyfikacja, ekstrakcja encji, QA na tabelach.

LayoutLMv3 to szczyt rozumienia dokumentów opartego na OCR. Mocny w formularzach i fakturach. Wymaga OCR jako elementu poprzedzającego. Najlepsza dokładność przed VLM na standaryzowanych benchmarkach dokumentowych.

### DocLLM (2023)

DocLLM (Wang i in., arXiv:2401.00908) jest generatywnym rodzeństwem LayoutLM. Generuje swobodne odpowiedzi warunkowane tokenami układu. Lepszy do QA na dokumentach; wciąż zależy od wejścia OCR.

### Era 3 — Natywnie VLM (2024+)

W 2024 roku VLM stały się wystarczająco dobre, aby całkowicie zastąpić potok. Podaj obraz pełnej strony w wysokiej rozdzielczości do VLM, zadaj pytanie, otrzymaj odpowiedź.

- LLaVA-NeXT 336-tile AnyRes działa dla małych dokumentów.
- Qwen2.5-VL dynamic-resolution obsługuje natywnie 2048+ pikseli.
- Claude Opus 4.7 obsługuje dokumenty 2576px.
- PaliGemma 2 (kwiecień 2025) trenuje specyficznie dla dokumentów + odręcznego pisma.

Luka między natywnym VLM a potokiem OCR szybko się zamknęła. Do 2026 roku natywne VLM wygrywają w:

- Tekście sceny (odręczny + drukowany, mieszane skrypty).
- Złożonych tabelach ze scalonymi komórkami.
- Równaniach matematycznych osadzonych w tekście.
- Rysunkach z adnotacjami tekstowymi.

Potoki OCR wciąż wygrywają w:

- Czysto-skanowych obciążeniach na masową skalę, gdzie liczy się opóźnienie na stronę.
- Niezawodności potoku (deterministyczne awarie vs halucynacje VLM).
- Regulowanych środowiskach wymagających audytowalnego wyjścia OCR.

### Granica Claude 4.7 / GPT-5

Przy natywnym wejściu 2576 pikseli, graniczne VLM osiągają rozumienie dokumentów z dokładnością bliską ludzkiej. Liczby benchmarkowe z początku 2026:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, potokowy LayoutLMv3 ~83.
- ChartQA: Claude 4.7 ~92.2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

Luka zamkniętych modeli to głównie rozdzielczość i skala bazowego LLM. Otwarte modele 7B są kilka punktów z tyłu, ale nadrabiają.

### Równania matematyczne i wyjście LaTeX

Artykuły naukowe potrzebują dokładnego wyjścia LaTeX dla równań. Nougat był na to trenowany. VLM trenowane z celami LaTeX (Qwen2.5-VL-Math, pochodne Nougat) produkują użyteczny LaTeX. Bez wyraźnego treningu LaTeX, VLM produkują czytelne, ale nieprecyzyjne transkrypcje.

Dla potoków artykułów naukowych w 2026: połącz Nougat na PDF, a następnie VLM na trudnych stronach.

### Odręczne pismo

Wciąż najtrudniejsze podzadanie. Mieszane drukowane + odręczne (notatki lekarzy, wypełnione formularze) to obszar, gdzie potoki OCR wciąż pokonują VLM kosztowo. VLM tylko-odręczne poprawiają się (Claude 4.7, PaliGemma 2).

### Przepis na 2026

Dla nowego projektu AI dokumentowego:

- Czysto drukowane faktury na skalę: LayoutLMv3 + reguły, opłacalne.
- Dokumenty mieszane (naukowe + odręczne + formularze): natywnie VLM (PaliGemma 2 lub Qwen2.5-VL).
- Pełne przetwarzanie arXiv: Nougat dla matematyki, VLM dla rysunków.
- Regulowane: potok OCR + walidator VLM do weryfikacji krzyżowej.

## Use It

`code/main.py`:

- Zabawkowy tokenizer świadomy układu: dla par (tekst, bbox) produkuje wejście w stylu LayoutLMv3.
- Generator schematów zadań w stylu Donut: szablon JSON dla formularzy.
- Porównanie budżetów tokenów na stronę w potoku OCR, Donut, Nougat i natywnym VLM.

## Ship It

Ta lekcja produkuje `outputs/skill-document-ai-stack-picker.md`. Dla projektu AI dokumentowego (domena, skala, jakość, regulacje) wybiera między potokiem OCR, specjalistą bez OCR i natywnym VLM.

## Exercises

1. Twój projekt to 10 mln faktur dziennie. Który stack minimalizuje koszt na stronę bez utraty dokładności?

2. Dlaczego LayoutLMv3 przewyższa czyste CLIP-VLM w QA formularzy, ale osiąga gorsze wyniki w tekście sceny? Co daje strumień bbox?

3. Nougat generuje LaTeX. Zaproponuj przypadek testowy, w którym wyjście natywnego VLM pokonuje Nougat w wierności LaTeX, oraz przypadek, w którym wygrywa Nougat.

4. Przeczytaj publikację PaliGemma 2 (Google, 2024). Jaki był kluczowy dodatek danych treningowych, który podniósł dokładność dokumentową w porównaniu do PaliGemma 1?

5. Zaprojektuj bezpieczny regulacyjnie hybryd: potok OCR jako podstawowy, VLM jako drugorzędna weryfikacja krzyżowa. Jak rozwiązujesz niezgodność?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OCR pipeline | "Tesseract-style" | Stack etapowy: wykryj -> OCR -> układ -> reguły; deterministyczny, kruchy |
| OCR-free | "Donut-style" | Transformer obraz-wyjście pomijający jawny OCR; pojedynczy model |
| Layout-aware | "LayoutLM" | Wejście zawiera współrzędne bbox per token; ujednolicone maskowanie między modalnościami |
| VLM-native | "Frontier VLM" | Podaj obraz strony bezpośrednio do Claude/GPT/Qwen VLM w wysokiej rozdzielczości; bez potoku |
| DocVQA | "Doc benchmark" | Standard dokumentowego VQA; najczęściej cytowany wynik |
| Markup output | "LaTeX / MD" | Strukturalny format wyjścia zamiast swobodnego tekstu; umożliwia dalszą automatyzację |

## Further Reading

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)