# Vision Transformers (ViT)

> Obraz to siatka fragmentów. Zdanie to siatka tokenów. Ten sam transformer przetwarza oba.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## Problem

Przed 2020 rokiem widzenie komputerowe oznaczało konwolucje. Każdy SOTA na ImageNet, COCO i benchmarkach detekcji używał CNN jako szkieletu. Transformery były dla języka.

Dosovitskiy i in. (2020) — "An Image is Worth 16x16 Words" — pokazali, że można całkowicie zrezygnować z konwolucji. Pokrój obraz na fragmenty o stałym rozmiarze, rzutuj liniowo każdy fragment do osadzenia, podaj sekwencję do waniliowego enkodera transformerów. Przy wystarczającej skali (pre-training na ImageNet-21k lub większym), ViT dorównuje lub przewyższa modele oparte na ResNet.

ViT zapoczątkował szerszy trend w 2026 roku: jedna architektura, wiele modalności. Whisper tokenizuje audio. ViT tokenizuje obrazy. Tokeny akcji dla robotyki. Tokeny pikseli dla wideo. Dla transportera nie ma to znaczenia — podaj mu sekwencję, a się nauczy.

Do 2026 roku ViT i jego pochodne (DeiT, Swin, DINOv2, ViT-22B, SAM 3) zdominowały widzenie komputerowe. CNN wciąż wygrywają na urządzeniach brzegowych i w zadaniach wrażliwych na opóźnienia. Wszystko inne ma ViT-a gdzieś w swoim stosie.

## Koncepcja

![Image → patches → tokens → transformer](../assets/vit.svg)

### Krok 1 — dzielenie na fragmenty

Podziel obraz o wymiarach `H × W × C` na sekwencję `N × (P·P·C)` płaskich fragmentów. Typowa konfiguracja: obraz `224 × 224`, fragmenty `16 × 16` → 196 fragmentów po 768 wartości każdy.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Rozmiar fragmentu to dźwignia. Mniejsze fragmenty = więcej tokenów, lepsza rozdzielczość, kwadratowy koszt uwagi. Większe fragmenty = grubsze, tańsze.

### Krok 2 — osadzenie liniowe

Pojedyncza uczona macierz rzutuje każdy płaski fragment na `d_model`. Odpowiednik konwolucji o rozmiarze jądra `P` i kroku `P`. W PyTorch to dosłownie `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — implementacja w 2 liniach.

### Krok 3 — dodanie tokena `[CLS]` i osadzeń pozycyjnych

- Dodaj uczony token `[CLS]` na początku. Jego końcowy stan ukryty jest reprezentacją obrazu używaną do klasyfikacji.
- Dodaj uczone osadzenia pozycyjne (oryginalny ViT) lub sinusoidalne 2D (późniejsze warianty).
- Od 2024 roku RoPE rozszerzone do 2D dla pozycji, czasami bez jawnych osadzeń.

### Krok 4 — standardowy enkoder transformerów

Ułóż L bloków `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Identyczne jak BERT. Brak warstw specyficznych dla widzenia. To jest pedagogiczna puenta artykułu.

### Krok 5 — głowa

Dla klasyfikacji: weź stan ukryty `[CLS]` → liniowy → softmax. Dla DINOv2 lub SAM, odrzuć `[CLS]`, użyj bezpośrednio osadzeń fragmentów.

### Warianty, które miały znaczenie

| Model | Rok | Zmiana |
|-------|------|--------|
| ViT | 2020 | Oryginał. Stały rozmiar fragmentu, pełna globalna uwaga. |
| DeiT | 2021 | Dystylacja; trenowalny tylko na ImageNet-1k. |
| Swin | 2021 | Hierarchiczny z przesuniętymi oknami. Stały subkwadratowy koszt. |
| DINOv2 | 2023 | Samonadzorowany (bez etykiet). Najlepsze ogólne cechy wizyjne. |
| ViT-22B | 2023 | 22B parametrów; działają prawa skalowania. |
| SigLIP | 2023 | ViT + para językowa, sigmoidalna strata kontrastowa. |
| SAM 3 | 2025 | Segmentacja wszystkiego; ViT-Large + dekoder masek z promptem. |

### Dlaczego to zajęło trochę czasu

ViT potrzebuje *dużo* danych, aby dorównać CNN, ponieważ nie ma żadnych indukcyjnych uprzedzeń CNN (niezmienniczość na translację, lokalność). Bez >100M oznakowanych obrazów lub silnego samonadzorowanego pre-treningu, CNN wciąż wygrywają przy porównywalnym nakładzie obliczeniowym. DeiT rozwiązał to w 2021 roku sztuczkami dystylacyjnymi; DINOv2 rozwiązał to trwale w 2023 roku samonadzorem.

## Zbuduj To

Zobacz `code/main.py`. Czysta implementacja z biblioteki standardowej: dzielenie na fragmenty + osadzenie liniowe + testy poprawności. Bez trenowania — ViT w realistycznej skali potrzebuje PyTorch i godzin na GPU.

### Krok 1: sztuczny obraz

Obraz RGB 24 × 24 jako lista wierszy krotek `(R, G, B)`. Używamy fragmentów 6×6 → 16 fragmentów, każdy z 108-wymiarowym wektorem osadzenia.

### Krok 2: dzielenie na fragmenty

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Porządek rastrowy: wiersz po wierszu w poprzek siatki. Każdy ViT używa tego porządku.

### Krok 3: osadzenie liniowe

Pomnóż każdy płaski fragment przez losową macierz `(patch_flat_size, d_model)`. Zweryfikuj, że kształt wyjścia to `(N_patches + 1, d_model)` po dodaniu `[CLS]`.

### Krok 4: policz parametry dla realistycznego ViT

Wypisz liczbę parametrów dla ViT-Base: 12 warstw, 12 głów, d=768, fragment=16. Porównaj z ResNet-50 (~25M). ViT-Base ma ~86M. ViT-Large ~307M. ViT-Huge ~632M.

## Użyj Tego

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**Osadzenia DINOv2 są domyślnym wyborem w 2026 roku dla cech obrazu.** Zamroź szkielet, wytrenuj małą głowę. Działa do klasyfikacji, wyszukiwania, detekcji, opisywania obrazów. Punktory kontrolne DINOv2 Meta przewyższają CLIP we wszystkich zadaniach widzenia niezwiązanych z tekstem.

**Dobór rozmiaru fragmentu.** Małe modele używają 16×16 (ViT-B/16). Gęsta predykcja (segmentacja) używa 8×8 lub 14×14 (SAM, DINOv2). Bardzo duże modele używają 14×14.

## Dostarcz To

Zobacz `outputs/skill-vit-configurator.md`. Umiejętność wybiera wariant ViT i rozmiar fragmentu dla nowego zadania wizyjnego na podstawie rozmiaru zbioru danych, rozdzielczości i budżetu obliczeniowego.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Zweryfikuj, że liczba fragmentów wynosi `(H/P) * (W/P)` i wymiar płaskiego fragmentu wynosi `P*P*C`.
2. **Średnie.** Zaimplementuj 2D sinusoidalne osadzenia pozycyjne — dwa niezależne kody sinusoidalne dla `row` i `col` każdego fragmentu, połączone. Podaj je do małego ViT w PyTorch i porównaj dokładność z uczonymi osadzeniami pozycyjnymi na CIFAR-10.
3. **Trudne.** Zbuduj 3-warstwowy ViT (PyTorch), wytrenuj na 1000 obrazów MNIST z fragmentami 4×4. Zmierz dokładność testową. Teraz dodaj pre-trening DINOv2 na tych samych 1000 obrazach (uproszczony: po prostu trenuj enkoder do przewidywania osadzeń fragmentów z zamaskowanych fragmentów). Czy dokładność się poprawia?

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Fragment (patch) | "Token transportera wizyjnego" | Płaski wektor wartości pikseli dla obszaru `P × P × C` obrazu. |
| Dzielenie na fragmenty (patchify) | "Pokrój + spłaszcz" | Potnij obraz na nienachodzące na siebie fragmenty, spłaszcz każdy do wektora. |
| Token `[CLS]` | "Podsumowanie obrazu" | Uczony token dodany na początku; jego końcowe osadzenie jest reprezentacją obrazu. |
| Uprzedzenie indukcyjne (inductive bias) | "Co model zakłada" | ViT ma mniej założeń wstępnych niż CNN; potrzebuje więcej danych, aby wyrównać różnicę. |
| DINOv2 | "Samonadzorowany ViT" | Trenowany bez etykiet przy użyciu augmentacji obrazu + nauczyciela momentum. Najlepsze ogólne cechy obrazu w 2026 roku. |
| SigLIP | "Następca CLIP" | ViT + enkoder tekstu trenowany z sigmoidalną stratą kontrastową; lepszy niż CLIP przy porównywalnym nakładzie obliczeniowym. |
| Swin | "Okienkowy ViT" | Hierarchiczny ViT z lokalną uwagą + przesuniętymi oknami; subkwadratowy. |
| Tokeny rejestrowe (register tokens) | "Sztuczka z 2023" | Kilka dodatkowych uczonych tokenów, które absorbują zlewy uwagi; poprawia cechy DINOv2. |

## Dalsza Lektura

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — artykuł ViT.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — poprawka tokenów rejestrowych dla DINOv2.