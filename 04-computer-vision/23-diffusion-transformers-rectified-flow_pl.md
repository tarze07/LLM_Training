# Transformery Dyfuzyjne & Przepływ Prostowany

> U-Net nie jest sekretem dyfuzji. Zastąp go transformerem, zamień harmonogram szumu na prostoliniowy przepływ i nagle masz SD3, FLUX i każdy model tekst-na-obraz z 2026.

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 10 (Diffusion DDPM), Phase 4 Lesson 14 (ViT), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~75 minut

## Cele Kształcenia

- Prześledzić ewolucję od U-Net DDPM (Lekcja 10) do Diffusion Transformer (DiT), MMDiT (SD3) i single+double-stream DiT (FLUX)
- Wyjaśnić przepływ prostowany: dlaczego prostoliniowa trajektoria między szumem a danymi pozwala modelom próbkować w 20 krokach zamiast 1000
- Zaimplementować mały blok DiT i pętlę treningową przepływu prostowanego, oba poniżej 100 linii
- Rozróżnić warianty modeli (SD3, FLUX.1-dev, FLUX.1-schnell, Z-Image, Qwen-Image) według architektury, liczby parametrów i licencji

## Problem

Lekcja 10 zbudowała DDPM z denoiserem U-Net. Ten przepis dominował w latach 2020-2023: U-Net + harmonogram beta + strata predykcji szumu. Wyprodukował Stable Diffusion 1.5 i 2.1 oraz DALL-E 2.

Każdy model tekst-na-obraz z 2026 roku odszedł od tego. Stable Diffusion 3, FLUX, SD4, Z-Image, Qwen-Image, Hunyuan-Image — żaden nie używa U-Net. Używają Transformerów Dyfuzyjnych (DiT). SD3 i FLUX zamieniają również harmonogram szumu DDPM na przepływ prostowany, który prostuje ścieżkę od szumu do danych i umożliwia inferencję w 1-4 krokach z wariantami konsystencji lub destylacji.

Ta zmiana ma znaczenie, ponieważ jest powodem, dla którego generowanie obrazów oparte na dyfuzji stało się sterowalne, dokładne względem promptu (SD3/SD4 rozwiązały renderowanie tekstu) i szybkie produkcyjnie. Zrozumienie DiT + przepływu prostowanego to zrozumienie stosu generatywnych obrazów w 2026.

## Koncepcja

### Od U-Net do transfornera

```mermaid
flowchart LR
    subgraph UNET["DDPM U-Net (2020)"]
        U1["Enkoder Conv"] --> U2["Wąskie gardło Conv"] --> U3["Dekoder Conv"]
    end
    subgraph DIT["DiT (2023)"]
        D1["Osadź łaty"] --> D2["Bloki transfornera"] --> D3["Odepchnij"]
    end
    subgraph MMDIT["MMDiT (SD3, 2024)"]
        M1["Strumień tekstu"] --> M3["Wspólna uwaga<br/>(oddzielne wagi na modalność)"]
        M2["Strumień obrazu"] --> M3
    end
    subgraph FLUX["FLUX (2024)"]
        F1["Bloki dwustrumieniowe<br/>(tekst + obraz oddzielnie)"] --> F2["Bloki jednostrumieniowe<br/>(konkatenacja + wspólne wagi)"]
    end

    style UNET fill:#e5e7eb,stroke:#6b7280
    style DIT fill:#dbeafe,stroke:#2563eb
    style MMDIT fill:#fef3c7,stroke:#d97706
    style FLUX fill:#dcfce7,stroke:#16a34a
```

- **DiT** (Peebles & Xie, 2023) — zastąp U-Net transformerem podobnym do ViT na łatach latentnych. Warunkowanie przez adaptacyjną normalizację warstw (AdaLN).
- **MMDiT** (SD3, Esser et al., 2024) — dwa strumienie z oddzielnymi wagami dla tokenów tekstu i obrazu, które dzielą wspólną uwagę.
- **FLUX** (Black Forest Labs, 2024) — pierwsze N bloków dwustrumieniowych jak SD3, późniejsze bloki konkatenują i dzielą wagi (jednostrumieniowe) dla wydajności przy większej głębokości.
- **Z-Image** (2025) — wydajny jednostrumieniowy DiT przy 6B parametrach, który kwestionuje "skalować za wszelką cenę".

### Przepływ prostowany w jednym akapicie

DDPM definiuje proces w przód jako zaszumione SDE, gdzie `x_t` jest coraz bardziej zniekształcone. Nauczona odwrotność to drugie SDE, rozwiązywane przez 1000 małych kroków.

Przepływ prostowany definiuje **prostoliniową** interpolację między czystymi danymi a czystym szumem:

```
x_t = (1 - t) * x_0 + t * epsilon,     t in [0, 1]
```

Trenuj sieć do przewidywania prędkości `v_theta(x_t, t) = epsilon - x_0` — kierunku w przód wzdłuż prostoliniowej ścieżki od czystych danych do szumu (`dx_t/dt`). Podczas próbkowania całkujesz tę prędkość wstecz, aby przejść od szumu w kierunku danych. Wynikowe ODE jest znacznie bliższe prostej linii, więc potrzeba znacznie mniej kroków całkowania do próbkowania.

SD3 nazywa to **Rectified Flow Matching**. FLUX, Z-Image i większość modeli z 2026 używa tego samego celu. Typowa inferencja: 20-30 kroków Eulera (deterministycznych) vs 50+ kroków DDIM w starym reżimie DDPM. Warianty dystylowane / turbo / schnell / LCM schodzą do 1-4 kroków.

### Warunkowanie AdaLN

DiT warunkują na kroku czasowym i klasie/tekście przez **adaptacyjną normalizację warstw**: przewiduj `scale` i `shift` z wektora warunkującego i zastosuj je po LayerNorm. Znacznie czystsze niż modulacja w stylu FiLM w U-Net i domyślne w każdym nowoczesnym DiT.

```
cond -> MLP -> (scale, shift, gate)
norm(x) * (1 + scale) + shift, następnie dodanie resztkowe * gate
```

### Enkodery tekstu w SD3 i FLUX

- **SD3** używa trzech enkoderów tekstu: dwóch modeli CLIP + T5-XXL. Embeddingi konkatenowane i podawane do strumienia obrazu jako warunkowanie tekstowe.
- **FLUX** używa jednego CLIP-L + T5-XXL.
- **Qwen-Image / Z-Image** warianty używają własnych enkoderów tekstu dopasowanych do ich bazowych LLM.

Enkoder tekstu to duża część tego, dlaczego SD3/FLUX rozumieją prompty znacznie lepiej niż SD1.5. Sam T5-XXL ma 4.7B parametrów.

### Wskazówki bez klasyfikatora wciąż obowiązują

Przepływ prostowany zmienia próbnik, nie warunkowanie. Wskazówki bez klasyfikatora (upuszczanie tekstu z 10% prawdopodobieństwem podczas treningu, mieszanie warunkowych i bezwarunkowych predykcji w inferencji) działają identycznie z przepływem prostowanym. Większość modeli z 2026 używa skali wskazówek 3.5-5 — niższej niż SD1.5 7.5, ponieważ modele przepływu prostowanego domyślnie ściślej podążają za promptami.

### Consistency, Turbo, Schnell, LCM

Cztery nazwy dla tego samego pomysłu: zdystyluj wolny model wielokrokowy w szybki model kilkukrokowy.

- **LCM (Latent Consistency Model)** — trenuj ucznia, który przewiduje końcowy `x_0` z dowolnego pośredniego `x_t` w jednym kroku.
- **SDXL Turbo / FLUX schnell** — modele 1-4 krokowe trenowane z adversarial diffusion distillation.
- **SD Turbo** — Modele konsystencji w stylu OpenAI zaadaptowane do latentnej dyfuzji.

Produkcyjne serwowanie każdego nowego modelu dostarcza zarówno checkpoint "pełnej jakości", jak i wariant "turbo / schnell". Schnell ("szybki" po niemiecku, konwencja Black Forest Labs) działa w 1-4 krokach i pasuje do pipeline czasu rzeczywistego.

### Krajobraz modeli w 2026

| Model | Rozmiar | Architektura | Licencja |
|-------|---------|--------------|----------|
| Stable Diffusion 3 Medium | 2B | MMDiT | SAI Community |
| Stable Diffusion 3.5 Large | 8B | MMDiT | SAI Community |
| FLUX.1-dev | 12B | Podwójny + Pojedynczy Strumień DiT | non-commercial |
| FLUX.1-schnell | 12B | ten sam, zdystylowany | Apache 2.0 |
| FLUX.2 | — | iterowany FLUX.1 | mieszana |
| Z-Image | 6B | S3-DiT (Skalowalny Jednostrumieniowy) | permisywna |
| Qwen-Image | ~20B | DiT + wieża Qwen | Apache 2.0 |
| Hunyuan-Image-3.0 | ~80B | DiT | badawcza |
| SD4 Turbo | 3B | DiT + destylacja | SAI Commercial |

FLUX.1-schnell to domyślny wybór open source w 2026. Z-Image to lider wydajności. FLUX.2 i SD4 to obecne szczyty jakości.

### Dlaczego ta zmiana fazy ma znaczenie

DDPM + U-Net działały. DiT + przepływ prostowany działa **lepiej, szybciej i skaluje się czyściej**. To przejście jest równoległe do tego od RNN do transformerów w NLP: obie architektury rozwiązywały ten sam problem, ale transformery skalowały się i teraz dominują. Każdy artykuł z 2026 o generowaniu obrazu, wideo lub 3D używa denoisera w kształcie DiT i zwykle celu przepływu prostowanego. U-Net DDPM jest teraz przede wszystkim pedagogiczny (Lekcja 10).

## Zbuduj To

### Krok 1: Blok DiT z AdaLN

```python
import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """
    Adaptacyjna LayerNorm z bramką. Przewiduje (scale, shift, gate) z warunkowania.
    Zainicjalizowana tak, że cały blok zaczyna jako identyczność ("zero init").
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.mlp(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4, cond_dim=192):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln2 = AdaLNZero(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        h, gate1 = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a
        h, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.mlp(h)
        return x
```

`AdaLNZero` zaczyna jako odwzorowanie tożsamościowe, ponieważ jego wagi MLP są inicjalizowane zerami. Trening wypycha blok od tożsamości; to dramatycznie stabilizuje głębokie transformery dyfuzyjne.

### Krok 2: Mały DiT

```python
def timestep_embedding(t, dim):
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TinyDiT(nn.Module):
    def __init__(self, image_size=16, patch_size=2, in_channels=3, dim=96, depth=4, heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, cond_dim=dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        n = x.size(0)
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) + self.pos
        t_emb = self.time_mlp(timestep_embedding(t, self.pos.size(-1)))
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.norm_out(x)
        x = self.head(x)
        return self._unpatchify(x, n)

    def _unpatchify(self, x, n):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.view(n, h, w, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(n, -1, h * p, w * p)
        return x
```

### Krok 3: Trening przepływu prostowanego

```python
import torch.nn.functional as F

def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon

    target_velocity = epsilon - x0
    pred_velocity = model(x_t, t)

    loss = F.mse_loss(pred_velocity, target_velocity)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

Porównaj z stratą predykcji szumu DDPM (Lekcja 10): ta sama struktura, inny cel. Zamiast przewidywać szum `epsilon`, przewidujemy **prędkość** `epsilon - x_0`, która wskazuje od danych do szumu wzdłuż prostoliniowej interpolacji.

### Krok 4: Próbnik Eulera

Przepływ prostowany to ODE. Metoda Eulera jest najprostsza i, dla dobrze wytrenowanego modelu przepływu prostowanego, prawie tak dokładna jak solvery wyższego rzędu przy 20+ krokach.

```python
@torch.no_grad()
def rectified_flow_sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

20 kroków. Na wytrenowanym modelu to produkuje próbki porównywalne z 1000-krokowym DDPM.

### Krok 5: Test dymny end-to-end

```python
import numpy as np

def synthetic_blobs(num=200, size=16, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((num, 3, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(num):
        cx, cy = rng.uniform(4, size - 4, size=2)
        r = rng.uniform(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        colour = rng.uniform(-1, 1, size=3)
        for c in range(3):
            out[i, c][mask] = colour[c]
    return torch.from_numpy(out)
```

Wytrenuj `TinyDiT` na tym z przepływem prostowanym. Po 500 krokach próbkowane wyjścia powinny wyglądać jak blade plamy koloru.

## Użyj Tego

Do rzeczywistego generowania obrazów z FLUX / SD3 / Z-Image, `diffusers` dostarcza każdy z ujednoliconym API:

```python
from diffusers import FluxPipeline, StableDiffusion3Pipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

out = pipe(
    prompt="a golden retriever surfing a tsunami, hyperrealistic, studio lighting",
    guidance_scale=0.0,           # schnell był trenowany bez CFG
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("surf.png")
```

Trzy linie. `FLUX.1-schnell` w czterech krokach. Zamień id modelu na `black-forest-labs/FLUX.1-dev` dla wyższej jakości przy 20-30 krokach z CFG.

Dla SD3:

```python
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
).to("cuda")
out = pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]
```

## Dostarcz To

Ta lekcja produkuje:

- `outputs/prompt-dit-model-picker.md` — wybiera między SD3, FLUX.1-dev, FLUX.1-schnell, Z-Image, SD4 Turbo dla danych ograniczeń jakości, opóźnienia i licencji.
- `outputs/skill-rectified-flow-trainer.md` — pisze kompletną pętlę treningową dla przepływu prostowanego z AdaLN DiT i próbkowaniem Eulera.

## Ćwiczenia

1. **(Łatwe)** Wytrenuj TinyDiT powyżej na syntetycznym zbiorze blob przez 500 kroków. Porównaj próbki wyprodukowane z 10, 20 i 50 krokami Eulera.
2. **(Średnie)** Dodaj warunkowanie tekstem przez konkatenację nauczonego embeddingu klasy do embeddingu czasu (10 "klas" blob według koloru). Próbkuj z klasą 0, 5 i 9 i zweryfikuj, że kolory się zgadzają.
3. **(Trudne)** Oblicz odległość Frécheta (proxy FID) między wygenerowanymi próbkami z wersji przepływu prostowanego i DDPM tej samej sieci o tym samym rozmiarze, trenowanej na tych samych danych przez tę samą liczbę kroków. Raportuj, która zbiega szybciej.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|--------|-----------------|----------------------|
| DiT | "Transformer dyfuzyjny" | Transformer zastępujący U-Net jako denoiser dyfuzyjny; operuje na łatowanych latentach |
| AdaLN | "Adaptacyjna norma warstw" | Warunkowanie czasem/tekstem przez nauczone scale, shift, gate stosowane po LayerNorm; standard w każdym nowoczesnym DiT |
| MMDiT | "Wielomodalny DiT (SD3)" | Oddzielne strumienie wag dla tokenów tekstu i obrazu dzielące wspólną uwagę własną |
| Jednostrumieniowy / dwustrumieniowy | "Trik FLUX" | Pierwsze N bloków dwustrumieniowych (oddzielne wagi na modalność), późniejsze bloki jednostrumieniowe (konkatenacja + wspólne wagi) dla wydajności |
| Przepływ prostowany | "Prostoliniowy szum-do-danych" | Liniowa interpolacja między danymi a szumem; sieć przewiduje prędkość; mniej kroków ODE potrzebnych w inferencji |
| Cel prędkości | "epsilon - x_0" | Cel regresji w przepływie prostowanym; wskazuje od czystych danych do szumu |
| Wskazówki CFG | "wskazówki bez klasyfikatora" | Mieszaj warunkowe i bezwarunkowe predykcje; wciąż używane w modelach przepływu prostowanego |
| Schnell / turbo / LCM | "Destylacja 1-4 krokowa" | Warianty małokrokowe zdystylowane z modeli pełnej jakości; produkcja czasu rzeczywistego |

## Dalsza Lektura

- [Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) — artykuł DiT
- [Scaling Rectified Flow Transformers (Esser et al., SD3 paper)](https://arxiv.org/abs/2403.03206) — MMDiT i przepływ prostowany w skali
- [FLUX.1 model card and technical report (Black Forest Labs)](https://huggingface.co/black-forest-labs/FLUX.1-dev) — szczegóły podwójnego + pojedynczego strumienia
- [Z-Image: Efficient Image Generation Foundation Model (2025)](https://arxiv.org/html/2511.22699v1) — jednostrumieniowy DiT przy 6B
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) — referencja dla każdego kompromisu projektowego dyfuzji
- [Latent Consistency Models (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) — jak LCM-LoRA daje inferencję w 4 krokach