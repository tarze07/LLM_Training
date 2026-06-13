# Generowanie 3D

> 3D to modalność, w której dźwignia 2D-na-3D jest najsilniejsza. Przełomem w 2023 było 3D Gaussian Splatting. Nacisk generatywny lat 2024-2026 nakłada wielowidokową dyfuzję + rekonstrukcję 3D na wierzch, aby tworzyć obiekty i sceny z pojedynczego promptu lub zdjęcia.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 minutes

## Problem

Treść 3D jest uciążliwa:

- **Reprezentacja.** Siatki (meshes), chmury punktów, siatki wokseli, pola odległości ze znakiem (SDF), neuronowe pola promieniowe (NeRF), 3D Gaussians. Każda ma kompromisy.
- **Niedobór danych.** ImageNet ma 14M obrazów. Największy czysty zestaw danych 3D (Objaverse-XL, 2023) ma ~10M obiektów, większość niskiej jakości.
- **Pamięć.** Siatka wokseli 512³ to 128M wokseli; użyteczny NeRF sceny potrzebuje 1M próbek/promień. Generowanie jest trudniejsze niż rekonstrukcja.
- **Nadzór.** Dla obrazu 2D masz piksele. Dla 3D masz zazwyczaj garść widoków 2D i musisz przejść do 3D.

Stos 2026 rozdziela te dwa problemy. Najpierw wygeneruj *obrazy 2D z wielu widoków* za pomocą modelu dyfuzyjnego. Po drugie, dopasuj *reprezentację 3D* (zwykle Gaussian splatting) do tych obrazów.

## Koncepcja

![Generowanie 3D: wielowidokowa dyfuzja + rekonstrukcja 3D](../assets/3d-generation.svg)

### Reprezentacja: 3D Gaussian Splatting (Kerbl et al., 2023)

Przedstaw scenę jako chmurę ~1M trójwymiarowych Gaussianów. Każdy ma 59 parametrów: pozycja (3), kowariancja (6, lub kwaternion 4 + skala 3), nieprzezroczystość (1), kolor harmonicznych sferycznych (48 dla stopnia 3, 3 dla stopnia 0).

Renderowanie = projekcja + alfa-kompozycja. Szybkie (~100 fps przy 1080p na 4090). Różniczkowalne. Dopasowywane przez gradientowe zejście względem zdjęć referencyjnych. Scena mieści się w 5-30 minut na konsumenckim GPU.

Dwie innowacje 2023-2024 na wierzchu:
- **Generatywne splaty Gaussianów.** Modele takie jak LGM, LRM, InstantMesh przewidują chmurę Gaussianów bezpośrednio z jednego lub kilku obrazów.
- **4D Gaussian Splatting.** Gaussiany z offsetami na klatkę dla dynamicznych scen.

### Wielowidokowa dyfuzja

Dostosuj wstępnie wytrenowany model dyfuzji obrazu do generowania wielu spójnych widoków tego samego obiektu z promptu tekstowego lub pojedynczego obrazu. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Zazwyczaj wyjście to 4-16 widoków wokół obiektu, podniesionych do 3D przez Gaussian splatting lub NeRF.

### Potoki tekst-na-3D

| Model | Wejście | Wyjście | Czas |
|-------|---------|---------|------|
| DreamFusion (2022) | tekst | NeRF przez SDS | ~1 godzina na zasób |
| Magic3D | tekst | siatka + tekstura | ~40 min |
| Shap-E (OpenAI, 2023) | tekst | niejawna 3D | ~1 min |
| SJC / ProlificDreamer | tekst | NeRF / siatka | ~30 min |
| LRM (Meta, 2023) | obraz | triplane | ~5 s |
| InstantMesh (2024) | obraz | siatka | ~10 s |
| SV3D (Stability, 2024) | obraz | nowe widoki | ~2 min |
| CAT3D (Google, 2024) | 1-64 obrazy | 3D NeRF | ~1 min |
| TripoSR (2024) | obraz | siatka | ~1 s |
| Meshy 4 (2025) | tekst + obraz | siatka PBR | ~30 s |
| Rodin Gen-1.5 (2025) | tekst + obraz | siatka PBR | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | obraz | siatka | ~30 s |

Kierunek 2025-2026: bezpośrednie modele tekst-na-siatkę z materiałami PBR odpowiednimi dla silników gier. Wielowidokowa dyfuzja jako krok pośredni wciąż jest najlepszym przepisem dla ogólnych obiektów.

### NeRF (dla kontekstu)

Neural Radiance Field (Mildenhall et al., 2020). Małe MLP przyjmuje `(x, y, z, kierunek patrzenia)` i zwraca `(kolor, gęstość)`. Renderuj przez całkowanie wzdłuż promieni. Przewyższa jakością syntezę nowych widoków opartą na siatkach, ale jest 100-1000x wolniejszy w renderowaniu. Wyparty przez Gaussian splatting w większości zastosowań czasu rzeczywistego, ale wciąż dominujący w badaniach.

## Zbuduj to

`code/main.py` implementuje zabawkowe dopasowanie 2D "Gaussian splatting": reprezentuje syntetyczny obraz docelowy (gładki gradient) jako sumę 2D splatów Gaussianów. Optymalizuj pozycje, kolory i kowariancje przez gradientowe zejście, aby dopasować się do celu. Widzisz dwie podstawowe operacje: renderowanie do przodu (splat + alfa-kompozycja) i dopasowanie przez gradientowe zejście.

### Krok 1: 2D splat Gaussa

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Krok 2: renderuj przez sumowanie splatów

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Prawdziwe 3D Gaussian splatting sortuje Gaussiany według głębokości i alfa-komponuje w odpowiedniej kolejności. Nasza zabawka 2D po prostu sumuje.

### Krok 3: dopasuj przez gradientowe zejście

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Pułapki

- **Niespójność widoków.** Jeśli wygenerujesz 4 widoki niezależnie i nie zgadzają się co do struktury obiektu, dopasowanie 3D będzie rozmyte. Naprawa: wielowidokowa dyfuzja ze wspólną uwagą.
- **Halucynacja tylnej strony.** Pojedynczy obraz → 3D musi wymyślić niewidoczną stronę. Jakość różni się ogromnie.
- **Eksplozja splatów Gaussianów.** Nieskrępowane trenowanie rośnie do 10M splatów i przeucza się. Zagęszczanie + heurystyki przycinania (z oryginalnego artykułu 3D-GS) są niezbędne.
- **Problemy topologii.** Siatki z pól niejawnych (SDF) często mają dziury lub samoprzecięcia. Uruchom remesher (np. wokselowy remesh Blendera) przed dostarczeniem.
- **Licencja danych treningowych.** Objaverse ma mieszane licencje; użycie komercyjne różni się w zależności od modelu.

## Użyj tego

| Zadanie | Wybór 2026 |
|---------|------------|
| Rekonstrukcja sceny ze zdjęć | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Tekst-na-3D obiekt do gier | Meshy 4 lub Rodin Gen-1.5 (wyjście PBR) |
| Obraz-na-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Synteza nowych widoków z kilku obrazów | CAT3D, SV3D |
| Rekonstrukcja dynamicznej sceny | 4D Gaussian Splatting |
| Avatar / ubrana postać | Gaussian Avatar, HUGS |
| Badania / SOTA | Cokolwiek wypadło w zeszłym tygodniu |

Do dostarczania produkcyjnego 3D w potoku gry lub e-commerce: Meshy 4 lub Rodin Gen-1.5 zwracają siatki PBR, które idą bezpośrednio do Unity / Unreal.

## Dostarcz to

Zapisz `outputs/skill-3d-pipeline.md`. Skill przyjmuje brief 3D (wejście: tekst / jeden obraz / kilka obrazów; wyjście: siatka / splat / NeRF; zastosowanie: render / gra / VR) i zwraca: potok (wielowidokowa dyfuzja + dopasowanie lub bezpośredni model siatki), model bazowy, budżet iteracji, post-processing topologii, potrzebne kanały materiałowe.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` z 4, 16, 64 Gaussianami. Raportuj końcowy MSE względem celu.
2. **Średnie.** Rozszerz na kolorowe Gaussiany (RGB). Potwierdź, że rekonstrukcja pasuje do wzorca kolorów celu.
3. **Trudne.** Używając gsplat lub Nerfstudio, zrekonstruuj prawdziwy obiekt z 50-zdjęciowej sesji. Raportuj czas dopasowania i końcowy SSIM na odłożonych widokach.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| 3D Gaussian Splatting | "3DGS" | Scena jako chmura 3D Gaussianów; różniczkowalne renderowanie alfa-kompozycji. |
| NeRF | "Neuronowe pole promieniowe" | MLP, które zwraca kolor + gęstość w punkcie 3D; renderowanie przez całkowanie promieni. |
| Triplane | "Trzy płaszczyzny 2D" | Faktoryzacja 3D na trzy osiowo-wyrównane siatki cech 2D; tańsze niż wolumetryczne. |
| SDS | "Score distillation sampling" | Trenuj model 3D, używając wyniku 2D-dyfuzji jako pseudo-gradientu. |
| Wielowidokowa dyfuzja | "Wiele widoków naraz" | Model dyfuzyjny, który zwraca partię spójnych widoków kamery. |
| PBR | "Fizycznie oparte renderowanie" | Materiał z kanałami albedo, chropowatości, metaliczności, normalnych. |
| Zagęszczanie | "Rozwijanie splatów" | Heurystyka treningowa 3DGS: dzielenie / klonowanie splatów w obszarach o wysokim gradiencie. |

## Uwaga produkcyjna: 3D nie ma jeszcze wspólnego podłoża

W przeciwieństwie do obrazu (latentna dyfuzja + DiT) i wideo (czasoprzestrzenny DiT), 3D nie ma jednego dominującego środowiska uruchomieniowego w 2026 roku. Drzewo decyzyjne produkcji rozgałęzia się na reprezentację:

- **NeRF / triplane.** Inferencja to ray-marching + forward MLP na próbkę. Render 512² wymaga milionów forwardów MLP. Agresywnie grupuj próbki promieni; stosuje się SDPA/xformers.
- **Wielowidokowa dyfuzja + rekonstrukcja LRM.** Potok dwuetapowy. Etap 1 (wielowidokowy DiT) to serwer dyfuzyjny tak jak w Lekcji 07. Etap 2 (transformer LRM) to jednorazowy forward na widokach. Ogólny profil opóźnienia to "dyfuzja + jednorazowy" — wybierz prymitywy serwowania per-etap odpowiednio.
- **SDS / DreamFusion.** Optymalizacja na zasób, nie inferencja. Buduj zadania, nie handlerów żądań.

Dla większości produktów 2026 roku właściwą odpowiedzią jest "uruchom wielowidokowy model dyfuzyjny na żądanie, zrekonstruuj do 3DGS asynchronicznie, serwuj 3DGS do podglądu w czasie rzeczywistym". To dzieli obciążenie czysto między serwer inferencji GPU (szybki) a optymalizator offline (wolny).

## Dalsza lektura

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — wielowidokowa dyfuzja.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.