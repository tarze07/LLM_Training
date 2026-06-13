# Konfiguracja GPU i chmura

> Trenowanie na CPU jest w porządku do nauki. Trenowanie na poważnie wymaga GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Learning Objectives

- Zweryfikuj lokalną dostępność GPU za pomocą `nvidia-smi` i API CUDA w PyTorch
- Skonfiguruj Google Colab z GPU T4 do darmowych eksperymentów w chmurze
- Porównaj wydajność mnożenia macierzy na CPU vs GPU i zmierz przyspieszenie
- Oszacuj największy model mieszczący się w twoim VRAM używając reguły fp16

## The Problem

Większość lekcji w fazach 1-3 działa dobrze na CPU. Ale gdy zaczniesz trenować CNN, transformery lub LLM (fazy 4+), potrzebujesz akceleracji GPU. Trening, który zajmuje 8 godzin na CPU, zajmuje 10 minut na GPU.

Masz trzy opcje: lokalne GPU, chmurowe GPU lub Google Colab (darmowe).

## The Concept

```
Twoje opcje:

1. Lokalne GPU NVIDIA
   Koszt: 0 zł (już je masz)
   Konfiguracja: Zainstaluj CUDA + cuDNN
   Najlepsze do: Regularnego użytku, dużych zbiorów danych

2. Google Colab (darmowy poziom)
   Koszt: 0 zł
   Konfiguracja: Żadna
   Najlepsze do: Szybkich eksperymentów, gdy nie masz GPU w domu

3. Chmurowe GPU (Lambda, RunPod, Vast.ai)
   Koszt: 0,20-2,00 USD/godz.
   Konfiguracja: SSH + instalacja
   Najlepsze do: Poważnego trenowania, dużych modeli
```

## Build It

### Option 1: Lokalne GPU NVIDIA

Sprawdź, czy masz:

```bash
nvidia-smi
```

Zainstaluj PyTorch z CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Option 2: Google Colab

1. Przejdź do [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Uruchom `!nvidia-smi`, aby zweryfikować

Przesyłaj notebooki z tego kursu bezpośrednio do Colaba.

### Option 3: Chmurowe GPU

Dla Lambda Labs, RunPod lub Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Brak GPU? Żaden problem.

Większość lekcji działa na CPU. Te, które potrzebują GPU, będą to wyraźnie mówić i zawierać linki do Colaba.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It: Benchmark GPU vs CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Exercises

1. Uruchom powyższy benchmark i porównaj czasy CPU vs GPU
2. Jeśli nie masz GPU, uruchom go na Google Colab i porównaj
3. Sprawdź, ile pamięci GPU masz i oszacuj największy model, jaki zmieścisz (reguła: 2 bajty na parametr dla fp16)

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "Programowanie GPU" | Platforma równoległych obliczeń NVIDIA pozwalająca uruchamiać kod na GPU |
| VRAM | "Pamięć GPU" | Pamięć wideo na GPU, oddzielona od pamięci systemowej. Ogranicza rozmiar modelu. |
| fp16 | "Połowiczna precyzja" | 16-bitowy zmiennoprzecinkowy, używa połowy pamięci fp32 przy minimalnej utracie dokładności |
| Tensor Core | "Szybki sprzęt macierzowy" | Wyspecjalizowane rdzenie GPU do mnożenia macierzy, 4-8x szybsze niż zwykłe rdzenie |