# Wizja o Dowolnej Rozdzielczości: Patch-n'-Pack i NaFlex

> Prawdziwe obrazy nie są kwadratami 224x224. Paragon ma proporcje 9:16, wykres 16:9, skan medyczny może mieć 4096x4096, zrzut ekranu z telefonu to 9:19.5. Przed 2024 rokiem odpowiedź VLM — skalowanie wszystkiego do ustalonego kwadratu — odrzucała sygnał, który umożliwia OCR, rozumienie dokumentów i analizę scen w wysokiej rozdzielczości. NaViT (Google, 2023) pokazał, że można pakować patche o zmiennej rozdzielczości w jedną partię transformera z blokowo-diagonalnym maskowaniem. M-RoPE od Qwen2-VL (2024) całkowicie porzuciło absolutne tablice pozycyjne. AnyRes od LLaVA-NeXT kafelkowało obrazy o wysokiej rozdzielczości na obraz bazowy + pod-obrazy. Wariant NaFlex od SigLIP 2 (2025) jest teraz domyślnym enkoderem dla otwartych VLM, które chcą obsługiwać każdą proporcję w jednym punkcie kontrolnym. Ta lekcja implementuje patch-n'-pack od podstaw.

**Type:** Build
**Languages:** Python (stdlib, packer paczek + maska blokowo-diagonalna)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 minutes

## Learning Objectives

- Pakować patche z partii obrazów o zmiennej rozdzielczości w jedną sekwencję i zbudować blokowo-diagonalną maskę uwagi.
- Wybierać między kafelkowaniem AnyRes (LLaVA-NeXT), NaFlex (SigLIP 2) i M-RoPE (Qwen2-VL) dla danego zadania.
- Obliczać budżety tokenów dla OCR, wykresów i fotografii bez zmiany rozmiaru.
- Wymienić trzy tryby awarii skalowania do kwadratu: zniekształcony tekst, przycięta treść, zmarnowane tokeny na dopełnianie.

## The Problem

Transformery oczekują sekwencji. Partia to stos sekwencji o tej samej długości. Jeśli obrazy mają 224x224, dostajesz 196 tokenów paczki za każdym razem, dopełnianie nie jest wymagane, zadanie zrobione. Trenuj na 224, wnioskuj na 224, nigdy więcej nie myśl o rozdzielczości.

Świat nie współpracuje. Dokumenty są pionowe (8.5x11 cali, około 2:3). Zrzuty wykresów są poziome (16:9). Paragony są wysokie i wąskie (1:3). Obrazowanie medyczne ma 2048x2048 lub więcej. Zrzuty ekranu z urządzeń mobilnych mają 1170x2532 (0.46:1).

Trzy opcje sprzed 2024 roku i dlaczego każda zawodzi:

1. Skalowanie do ustalonego kwadratu (224x224 lub 336x336). Zniekształcenie deformuje tekst i twarze. Pomniejszenie niszczy etykiety wykresów i treść OCR. Standardowa praktyka do czasu LLaVA-1.5.
2. Przycięcie do ustalonych proporcji. Odrzucasz większość obrazu, a wybór miejsca przycięcia to osobny problem wizyjny.
3. Dopełnienie do najdłuższego boku. Naprawia zniekształcenia, ale marnuje 50%+ tokenów na dopełnianie w przypadku obrazów portretowych. Kwadratowy koszt uwagi na wszystkich tych tokenach dopełniających.

Odpowiedź z lat 2024-2025: pozwól transformerowi konsumować patche w natywnej rozdzielczości obrazu i wymyśl, jak spakować heterogeniczną partię w jedną sekwencję bez marnowania mocy obliczeniowej.

## The Concept

### NaViT i patch-n'-pack

NaViT (Dehghani et al., 2023) to praca, która pokazała, że to działa na dużą skalę. Pomysł jest mechaniczny:

1. Dla każdego obrazu w partii oblicz jego natywną siatkę paczek przy wybranym rozmiarze paczki (np. 14).
2. Spłaszcz patche każdego obrazu do jego własnej sekwencji o zmiennej długości.
3. Połącz patche wszystkich obrazów w jedną długą sekwencję dla partii.
4. Zbuduj blokowo-diagonalną maskę uwagi, aby patche obrazu A skupiały się tylko w obrębie obrazu A.
5. Przenoś informację o pozycji na paczkę (2D RoPE lub ułamkowe osadzenia pozycyjne).

Partia trzech obrazów o rozmiarach 336x336 (576 tokenów), 224x224 (256 tokenów) i 448x336 (768 tokenów) staje się jedną sekwencją 1600 tokenów z blokowo-diagonalną maską 1600x1600. Żadnego dopełniania. Żadnego marnowania mocy obliczeniowej. Transformer obsługuje dowolne proporcje.

NaViT wprowadził również ułamkowe odrzucanie paczek podczas treningu — losowe odrzucanie 50% paczek w całej partii — co zarówno regularyzuje, jak i przyspiesza trening. SigLIP 2 odziedziczył to.

### AnyRes (LLaVA-NeXT)

AnyRes od LLaVA-NeXT to pragmatyczna alternatywa. Mając obraz o wysokiej rozdzielczości i stały enkoder (CLIP lub SigLIP przy 336), kafelkuj obraz:

1. Wybierz układ siatki z predefiniowanego zestawu — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2) itd. — który najlepiej pasuje do proporcji obrazu.
2. Pokafelkuj cały obraz w siatkę; każdy kafelek staje się wycinkiem 336x336.
3. Wyprodukuj również miniaturkę: cały obraz skalowany do 336x336 jako token kontekstu globalnego.
4. Zakoduj każdy kafelek przez zamrożony enkoder 336. Połącz tokeny kafelków + tokeny miniatury.

Dla obrazu 672x672 w siatce 2x2 plus miniatura: 4 * 576 + 576 = 2880 tokenów wizualnych. Kosztowne, ale skuteczne — LLM widzi zarówno szczegóły lokalne, jak i kontekst globalny.

AnyRes jest wyborem, gdy twój enkoder jest zamrożony i obsługuje tylko jedną rozdzielczość. Eksploduje liczbę tokenów dla dużych obrazów (obraz 1344x1344 w siatce 4x4 to 9216 + 576 ≈ 9800 tokenów, co wypełnia większość 8k kontekstu LLM).

### M-RoPE (Qwen2-VL)

Qwen2-VL wprowadził Multimodalne Obrótowe Osadzenie Pozycyjne (M-RoPE). Zamiast ułamkowych pozycji NaViT czy kafelkowania z miniaturką AnyRes, każdy patche niesie 3D pozycję (czasową, wysokość, szerokość). Obrót zapytań/kluczy obsługuje dowolne H, W i długość czasową.

M-RoPE zapewnia natywną dynamiczną rozdzielczość bez przekwalifikowywania. Podczas wnioskowania podajesz obraz o dowolnym HxW, embedder paczek produkuje tokeny H/14 x W/14, każdy token dostaje swoją pozycję (t=0, r=wiersz, c=kolumna), RoPE obraca uwagę z odpowiednimi częstotliwościami, gotowe. Qwen2.5-VL i Qwen3-VL kontynuują to. V2PE InternVL3 to ten sam pomysł ze zmiennym kodowaniem na modalność.

W przeciwieństwie do AnyRes, M-RoPE ma O(H x W / P^2) tokenów w natywnej rozdzielczości — bez mnożnikowego narzutu kafelkowania. W przeciwieństwie do NaViT, nadal oczekuje pojedynczego obrazu na przejście. Łączenie w partie przy różnych rozdzielczościach nadal wymaga patch-n'-pack na wierzchu.

### NaFlex (SigLIP 2)

NaFlex to tryb natywnie elastyczny punktu kontrolnego SigLIP 2. Pojedynczy model obsługuje wiele długości sekwencji (256, 729, 1024 tokenów) podczas wnioskowania. Wewnętrznie używa stylu NaViT patch-n'-pack podczas treningu i absolutnych ułamkowych pozycji na paczkę. Główna zaleta: jeden punkt kontrolny, wybierz swój budżet tokenów podczas wnioskowania na podstawie zadania.

Dla zadania semantycznego (klasyfikacja, wyszukiwanie), 256 tokenów. Dla OCR lub rozumienia wykresów, 1024 tokenów. Bez przekwalifikowywania.

### Maska pakowania

Blokowo-diagonalna maska to miejsce, gdzie większość implementacji się potyka. Dla spakowanej sekwencji o długości `N_total` obejmującej obrazy `i=0..B-1` o długościach `n_i`, maska `M` o kształcie `(N_total, N_total)` wynosi 1, jeśli oba indeksy mieszczą się w bloku tego samego obrazu, w przeciwnym razie 0. Można ją zbudować z listy długości skumulowanych:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

To jedna linijka w PyTorch z `torch.block_diag` lub jawnym gather. Ścieżka zmiennej długości FlashAttention (`cu_seqlens`) pomija maskę całkowicie i uwzględnia wewnątrz sekwencji, używając bezpośrednio tensora długości skumulowanych — ~10x szybciej niż gęsta maska dla typowych partii.

### Budżety tokenów

Wybierz swoją strategię według zadania:

- OCR / dokumenty: 1024-4096 tokenów. SigLIP 2 NaFlex przy 1024 lub AnyRes 3x3 + miniatura.
- Wykresy i UI: 729-1024 tokenów przy 384-448 natywnie. Qwen2.5-VL dynamiczna rozdzielczość z ograniczeniem maksymalnej liczby pikseli.
- Naturalne zdjęcia: 256-576 tokenów wystarczy. Podstawowy LLM widzi wystarczająco. Płać za tokeny tam, gdzie gęstość treści jest wysoka.
- Wideo: 64-128 tokenów na klatkę po pooling'u przestrzennym, 2-8 FPS. Lekcja 12.17 omawia to.

Zasada produkcji 2026: wybierz ograniczenie maksymalnej liczby pikseli na zadanie, koduj w natywnych proporcjach do tego limitu, spakuj partię i pomiń dopełnianie. Qwen2.5-VL udostępnia `min_pixels` i `max_pixels` właśnie dla tego pokrętła.

## Use It

`code/main.py` implementuje patch-n'-pack dla heterogenicznej partii obrazów z całkowitymi współrzędnymi pikseli.:

- Przyjmuje listę rozmiarów obrazów (H, W).
- Oblicza długość sekwencji paczek każdego obrazu przy rozmiarze paczki 14.
- Pakuje je w jedną sekwencję o całkowitej długości `sum(n_i)`.
- Buduje blokowo-diagonalną maskę uwagi (gęstą, dla przejrzystości).
- Porównuje koszt pakowania vs skalowanie do kwadratu i kafelkowanie AnyRes.
- Wypisuje tabelę budżetu tokenów dla mieszanej partii (paragon, wykres, zrzut ekranu, zdjęcie).

Uruchom to. Liczby, które wypadną, są powodem, dla którego każdy otwarty VLM w 2026 używa patch-n'-pack.

## Ship It

Ta lekcja produkuje `outputs/skill-resolution-budget-planner.md`. Mając mieszane obciążenie proporcji (OCR, wykresy, zdjęcia, klatki wideo) i całkowity budżet tokenów, wybiera odpowiednią strategię (NaFlex, AnyRes, M-RoPE lub stały kwadrat) i emituje konfigurację na żądanie. Użyj tej umiejętności, gdy wymiarujesz VLM dla produktu — zapobiega to cichej 10-krotnej eksplozji tokenów, która zabija budżety opóźnień.

## Exercises

1. Paragon ma 600x1500 (1:2.5). Przy rozmiarze paczki 14, ile tokenów w natywnej rozdzielczości? Ile po skalowaniu do kwadratu 336? Które traci więcej dokładności OCR w praktyce?

2. Zbuduj blokowo-diagonalną maskę dla partii czterech obrazów o długościach 256, 576, 729, 1024. Zweryfikuj, że macierz uwagi ma 2585x2585 i ma dokładnie `256^2 + 576^2 + 729^2 + 1024^2` niezerowych wpisów.

3. Dla obrazu 1792x896 przy paczce 14, porównaj: (a) skalowanie do kwadratu 336, potem kodowanie, (b) AnyRes 2x1 + miniatura, (c) M-RoPE w natywnej rozdzielczości. Które używa najmniej tokenów? Które zachowuje najwięcej szczegółów?

4. Zaimplementuj ułamkowe odrzucanie paczek: mając spakowaną sekwencję, odrzuć 50% tokenów równomiernie losowo i zaktualizuj odpowiednio blokowo-diagonalną maskę. Zmierz zmianę rzadkości maski.

5. Przeczytaj Sekcję 3.2 artykułu Qwen2-VL (arXiv:2409.12191). Opisz w dwóch zdaniach, co kontrolują `min_pixels` i `max_pixels` i dlaczego oba ograniczenia mają znaczenie.

## Key Terms

| Term | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Patch-n'-pack | "Pakowanie stylu NaViT" | Łączenie sekwencji paczek o zmiennej długości z różnych obrazów w jeden wymiar partii |
| Block-diagonal mask | "Maska pakowania" | Maska uwagi, która ogranicza patche każdego obrazu do skupiania się tylko na sobie, a nie na sąsiadach w paczce |
| AnyRes | "Kafelkowanie LLaVA-NeXT" | Dzielenie obrazu o wysokiej rozdzielczości na siatkę kafelków o stałym rozmiarze plus globalna miniatura; kodowanie każdego kafelka stałym enkoderem |
| NaFlex | "Natywnie elastyczny SigLIP 2" | Pojedynczy punkt kontrolny SigLIP 2 obsługujący budżety 256/729/1024 tokenów podczas wnioskowania bez przekwalifikowywania |
| M-RoPE | "Multimodalne RoPE" | 3D obrotowe kodowanie pozycyjne (czas, wiersz, kolumna) obsługujące dowolne H, W, T bez tablic pozycji |
| cu_seqlens | "Pakowanie FlashAttention" | Tensor długości skumulowanych używany przez ścieżkę zmiennej długości FlashAttention zamiast gęstej blokowo-diagonalnej maski |
| min_pixels / max_pixels | "Ograniczenia rozdzielczości" | Pokrętła na żądanie Qwen2.5-VL ograniczające liczbę tokenów dla bardzo małych lub bardzo dużych wejść |
| Visual token budget | "Ile tokenów na obraz" | Przybliżona liczba tokenów paczki emitowanych na obraz; ustala budżet promptu LLM i koszt uwagi |

## Further Reading

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)