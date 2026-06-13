# Ekonomika platform inferencyjnych — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> Rynek inferencji w 2026 roku to już nie wynajem czasu GPU. Dzieli się na krzem niestandardowy (Groq, Cerebras, SambaNova), platformy GPU (Baseten, Together, Fireworks, Modal) i marketplace'y API (Replicate, DeepInfra). Fireworks podniósł cenę o 1 USD/h za GPU 1 maja 2026, a wycena 4 mld USD przy 10+ bilionach tokenów/dzień mówi, że model oparty na wolumenie działa. Baseten zamknął Serię E o wartości 300 mln USD przy wycenie 5 mld USD w styczniu 2026. Zasada pozycjonowania konkurencyjnego jest prosta: Fireworks optymalizuje latencję, Together optymalizuje szerokość katalogu, Baseten optymalizuje polor korporacyjny, Modal optymalizuje natywne dla Pythona DX, Replicate optymalizuje zasięg multimodalny, Anyscale optymalizuje rozproszonego Pythona. Ta lekcja daje matrycę, którą możesz podać założycielowi.

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## Learning Objectives

- Wymień trzy segmenty rynku (krzem niestandardowy, platformy GPU, API-first) i przypisz każdego dostawcę do segmentu.
- Wyjaśnij, dlaczego model cenowy „za token" kompresuje się w kierunku krzywej kosztów silnika serwującego, a nie sprzętu.
- Oblicz efektywny koszt na żądanie u co najmniej trzech dostawców i wyjaśnij, kiedy cena za minutę (Baseten, Modal) bije cenę za token.
- Zidentyfikuj, która platforma jest właściwym domyślnym wyborem dla danego obciążenia (bezserwerowe zrywne, stałe wysokiej przepustowości, dostrojone warianty, multimodalne).

## The Problem

Oceniłeś zarządzane platformy hiperskalerów. Zdecydowałeś, że potrzebujesz węższego, szybszego dostawcy — Fireworks dla latencji, Together dla szerokości, Baseten dla dostrojonego modelu niestandardowego. Teraz masz sześć realnych wyborów, a strony cenowe nie są spójne. Fireworks pokazuje $/M tokenów; Baseten pokazuje $/minutę; Modal pokazuje $/sekundę; Replicate pokazuje $/predykcję. Nie możesz ich porównać bezpośrednio bez modelowania obciążenia.

Co gorsza, model biznesowy za każdą stroną cenową jest inny. Fireworks uruchamia własny niestandardowy silnik (FireAttention) na współdzielonych GPU; stawka za token odzwierciedla ich krzywą wykorzystania. Baseten daje Truss + dedykowane GPU; cena za minutę odzwierciedla ekskluzywność. Modal to prawdziwy bezserwerowy Python — rozliczenie za sekundę z subsekundowym zimnym startem. Ten sam wynik (odpowiedź LLM), trzy różne funkcje kosztu.

Ta lekcja modeluje wszystkich sześciu i mówi, kiedy każdy wygrywa.

## The Concept

### Trzy segmenty

**Krzem niestandardowy** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). Zazwyczaj 5-10x szybszy dekod niż klaster GPU na tym samym modelu. Wyższa cena za token (Groq ~0,99 USD/M na Llama-70B późnym 2025), ale niezrównany dla przypadków użycia wrażliwych na latencję. Groq to produkcyjny wybór dla agentów głosowych i tłumaczenia w czasie rzeczywistym.

**Platformy GPU** — Baseten, Together, Fireworks, Modal, Anyscale. Działają na NVIDIA (H100, H200, B200 w 2026) lub czasami AMD. Warstwa ekonomiczna między „surowym wynajmem GPU" (RunPod, Lambda) a „zarządzaną usługą hiperskalera" (Bedrock).

**Marketplace'y API-first** — Replicate, DeepInfra, OpenRouter, Fal. Szeroki katalog, płatność za predykcję lub za sekundę, kładą nacisk na czas do pierwszego wywołania.

### Fireworks — platforma GPU zoptymalizowana pod latencję

- Silnik FireAttention (niestandardowy); reklamowany jako 4x niższa latencja niż vLLM na równoważnych konfiguracjach.
- Poziom wsadowy za ~50% stawki bezserwerowej dla obciążeń nieinteraktywnych.
- Dostrojony model serwowany po tej samej stawce co model bazowy — prawdziwy wyróżnik w porównaniu z dostawcami, którzy pobierają premię za Twoją LoRA.
- Połowa 2026: podniesiono wynajem GPU na żądanie o 1 USD/h, efektywnie od 1 maja 2026. Ceny wolumenowe negocjowalne przy skali.
- Sygnał finansowy: wycena 4 mld USD, obsłużone 10+ bilionów tokenów/dzień.

### Together — zoptymalizowany pod szerokość katalogu

- 200+ modeli, w tym wydania open-source w ciągu dni od publikacji upstream.
- 50-70% tańszy niż Replicate na równoważnych modelach LLM — pozycjonowanie „AI Native Cloud" to wolumen i katalog.
- Inferencja + fine-tuning + trenowanie w jednym API.

### Baseten — zoptymalizowany pod polor korporacyjny

- Framework Truss: pakowanie modeli z zależnościami, sekretami, konfiguracją serwowania w jednym manifeście.
- Zakres GPU od T4 do B200. Rozliczenie za minutę z rozsądną mitigacją zimnego startu.
- SOC 2 Type II, gotowy na HIPAA. Typowy wybór w fintech i opiece zdrowotnej.
- Wycena 5 mld USD, Seria E w styczniu 2026 (300 mln USD od CapitalG, IVP, NVIDIA).

### Modal — zoptymalizowany pod natywny Python

- Infrastruktura-jako-kod w czystym Pythonie. Ozdób funkcję `@modal.function(gpu="A100")` i wdróż jednym poleceniem.
- Rozliczenie za sekundę. Zimny start 2-4 s z wstępnym ogrzewaniem; <1 s dla małych modeli.
- Seria B o wartości 87 mln USD przy wycenie 1,1 mld USD (2025). Najwyższy wynik doświadczenia programisty w niezależnych ankietach.

### Replicate — szerokość multimodalna

- Płatność za predykcję. Domyślna platforma dla modeli obrazu, wideo i audio.
- Ekosystem integracji (Zapier, Vercel, wtyczki CMS).
- Mniej konkurencyjny na stawkach za token LLM, ale wygrywa na różnorodności multimodalnej.

### Anyscale — natywny Ray

- Zbudowany na Ray; RayTurbo to zastrzeżony silnik inferencyjny Anyscale (konkuruje z vLLM).
- Najlepszy dla rozproszonych obciążeń Pythona, gdzie krok inferencji jest jednym węzłem w większym grafie.
- Zarządzane klastry Ray; ścisła integracja z Ray AIR i Ray Serve.

### Za token versus za minutę — kiedy każde wygrywa

Za token ma sens, gdy obciążenie jest niewrażliwe na latencję i zrywne — płacisz tylko za to, czego używasz. Za minutę ma sens, gdy wykorzystanie jest wysokie i przewidywalne — bijesz cenę za token, gdy nasycasz GPU.

Zgrubna zasada: dla obciążeń powyżej ~30% stałego wykorzystania dedykowanego GPU, cena za minutę (Baseten, Modal) zaczyna bić cenę za token (Fireworks, Together). Poniżej tego, cena za token wygrywa, ponieważ unikasz płacenia za bezczynność.

### Niestandardowy silnik to prawdziwa fosa

Każda platforma powyżej vLLM i SGLang twierdzi, że ma niestandardowy silnik. FireAttention, RayTurbo, stos inferencyjny Baseten. Twierdzenia o niestandardowych silnikach ocierają się o marketing — uczciwe ramy mówią, że vLLM + SGLang stanowią około 80% produkcji open-source'owej inferencji, a wyróżnikami na poziomie platformy są DX, atrybucja i SLA.

### Liczby, które powinieneś zapamiętać

- Wynajem GPU Fireworks: podwyżka o 1 USD/h, efektywnie od 1 maja 2026.
- Twierdzenie Fireworks: 4x niższa latencja niż vLLM na równoważnych konfiguracjach.
- Together: 50-70% taniej niż Replicate na LLM.
- Wycena Baseten: 5 mld USD (Seria E, styczeń 2026, runda 300 mln USD).
- Wycena Modal: 1,1 mld USD (Seria B, 2025).
- Cena za minutę bije cenę za token powyżej ~30% stałego wykorzystania.

```figure
cost-per-token
```

## Use It

`code/main.py` porównuje sześciu dostawców na syntetycznym obciążeniu w różnych modelach cenowych. Raportuje $/dzień i efektywne $/M tokenów. Uruchom, aby znaleźć próg rentowności między ceną za token a za minutę.

## Ship It

Ta lekcja produkuje `outputs/skill-inference-platform-picker.md`. Biorąc profil obciążenia, SLA i budżet, wybiera podstawową platformę inferencyjną i wymienia wicemistrza.

## Exercises

1. Uruchom `code/main.py`. Przy jakim stałym wykorzystaniu Baseten (za minutę) bije Fireworks (za token) dla modelu 70B na jednym H100? Wyprowadź punkt przecięcia samodzielnie i porównaj z regułą kciuka.
2. Twój produkt serwuje generowanie obrazów plus czat plus zamianę mowy na tekst. Wybierz platformy dla każdej modalności i nazwij wzorzec bramy, który je ujednolica.
3. Fireworks podnosi ceny o 1 USD/h na Twoim podstawowym modelu. Zamodeluj łączny wpływ kosztów, jeśli 40% Twojego ruchu przejdzie na poziom wsadowy (50% zniżki).
4. Regulowany klient wymaga SOC 2 Type II + HIPAA + dedykowanych GPU. Które trzy platformy są realne i która wygrywa na FinOps?
5. Porównaj koszt na 1000 predykcji dla Llama 3.1 70B na Fireworks serverless, Together na żądanie, Baseten dedykowany i Replicate API. Co jest najtańsze przy 10 predykcjach/dzień? Przy 10 000?

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Custom silicon | „chipspoza GPU" | Groq LPU, Cerebras WSE, SambaNova RDU — zoptymalizowane pod dekod |
| FireAttention | „silnik Fireworks" | Niestandardowe jądro attention; reklamowane jako 4x niższa latencja niż vLLM |
| Truss | „format Baseten" | Manifest pakowania modeli; zależności + sekrety + konfiguracja serwowania |
| Per-token | „ceny API" | Opłata za zużyte tokeny; brak płatności za bezczynność |
| Per-minute | „ceny dedykowane" | Opłata za rzeczywisty czas GPU; wygrywa przy wysokim wykorzystaniu |
| Per-prediction | „ceny Replicate" | Opłata za wywołanie modelu; powszechne dla obrazów/wideo |
| RayTurbo | „silnik Anyscale" | Zastrzeżona inferencja na Ray; konkuruje z vLLM na klastrach Ray |
| Batch tier | „50% zniżki" | Kolejka nieinteraktywna po obniżonej stawce; powszechna na Fireworks, OpenAI |
| Fine-tuned at base rate | „LoRA Fireworks" | Opłata za żądania LoRA po stawce modelu bazowego (wyróżnik) |

## Further Reading

- [Fireworks Pricing](https://fireworks.ai/pricing) — stawki za token, poziom wsadowy, wynajem GPU.
- [Baseten Pricing](https://www.baseten.co/pricing/) — stawki za minutę, pojemność zobowiązana, poziomy korporacyjne.
- [Modal Pricing](https://modal.com/pricing) — stawki GPU za sekundę i darmowy poziom.
- [Together AI Pricing](https://www.together.ai/pricing) — katalog modeli i stawki za token.
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo i zarządzane ceny Ray.
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — ocena porównawcza.
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — krajobraz dostawców.

(End of file - total 135 lines)