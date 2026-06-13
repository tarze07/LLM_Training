# Wieloregionowe serwowanie LLM i lokalność pamięci podręcznej KV

> Równoważenie obciążenia round-robin jest aktywnie szkodliwe dla buforowanej inferencji LLM. Żądanie, które nie trafi na węzeł przechowujący swój prefiks, płaci pełny koszt prefillu — zgrubnie 800 ms przy P50 na długim prompcie wobec ~80 ms z trafieniem w pamięci podręcznej. W 2026 produkcyjnym wzorcem jest router świadomy pamięci podręcznej (vLLM Router w Rust, router llm-d), który konsumuje zdarzenia pamięci podręcznej KV i kieruje na podstawie dopasowania skrótu prefiksu. Najnowsze badania (GORGO) czynią wieloregionowe opóźnienie sieciowe jawnym terminem w funkcji celu routingu. Komercyjne oferty „wieloregionowej inferencji" (Bedrock cross-region inference, GKE multi-cluster gateways) traktują inferencję jako nieprzezroczystą — obsługują dostępność, a nie TTFT. JPMorgan i Mayo Clinic uruchomili failover us-east-1 w listopadzie 2024 w ~22 minuty. Rzeczywistość DR: 32% awarii DR LLM wynika z tego, że zespoły wykonały kopię zapasową wag, ale zapomniały o plikach tokenizera lub konfiguracjach kwantyzacji.

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Learning Objectives

- Wyjaśnij, dlaczego równoważenie obciążenia round-robin psuje buforowaną inferencję i określ ilościowo karę TTFT.
- Narysuj diagram routera świadomego pamięci podręcznej: dane wejściowe (zdarzenia pamięci podręcznej KV), algorytm (dopasowanie skrótu prefiksu), rozstrzygacz (wykorzystanie GPU).
- Nazwij główną przyczynę 32% awarii DR dla LLM (brakujące pliki tokenizera / konfiguracje kwantyzacji) i podaj trzypunktową listę kontrolną DR.
- Odróżnij komercyjne oferty wieloregionowe (Bedrock CRI, GKE Multi-Cluster Gateway) od routingu świadomego KV.

## The Problem

Twoja usługa działa w us-east-1, us-west-2 i eu-west-1. Umieszczasz ALB z przodu z round-robin. Współczynnik trafień pamięci podręcznej prefiksów w produkcji spada do 8%. P50 TTFT się potraja. Twoje logi vLLM pokazują, że każde żądanie płaci pełny koszt prefillu.

Round-robin jest optymalne dla usług bezstanowych. Inferencja LLM jest z założenia stanowa — pamięć podręczna KV koduje wszystko, co model widział. Routing na ślepo to routing w niewłaściwą pamięć podręczną.

Osobno, Twój zespół ma plan DR. Tworzysz kopię zapasową wag modelu do S3 w innym regionie. Występuje awaria regionalna; próbujesz failover; replika odmawia uruchomienia. Zapomniałeś, że tokenizer.json, konfiguracja kwantyzacji i konfiguracja skalowania RoPE były w osobnym buckecie, którego nie zsynchronizowałeś.

Wieloregionowe serwowanie LLM to problem pamięci podręcznej, problem routingu i problem higieny DR — a nie problem równoważenia obciążenia.

## The Concept

### Routing świadomy pamięci podręcznej

Przychodzi żądanie z promptem. Router haszuje prefiks (powiedzmy, pierwsze 512 tokenów); pyta każdą replikę „czy masz ten prefiks w pamięci podręcznej?" Repliki publikują zdarzenia pamięci podręcznej KV na kanale pub/sub w miarę alokowania i wywłaszczania bloków. Router wybiera replikę z dopasowaniem, w przeciwnym razie stosuje rozstrzygacz oparty na wykorzystaniu GPU.

**vLLM Router** (Rust, stos produkcyjny 2026): subskrybuje zdarzenia `kv.cache.block_added`, utrzymuje mapę skrótu prefiksu → indeks repliki, kieruje z wyszukiwaniem O(1). Rozstrzyga na najmniejszą głębokość kolejki, gdy brak dopasowania.

**Router llm-d**: ten sam wzorzec, natywny dla Kubernetes. Publikuje zdarzenia przez API ControlPlane.

**SGLang RadixAttention** (Faza 17 · 06) jest odpowiednikiem wewnątrz repliki. Routing między replikami jest ściśle nadrzędny.

### Liczby

P50 TTFT na 2K-tokenowym prompcie, Llama 3.3 70B FP8, H100:
- Trafienie w pamięci podręcznej (ta sama replika, prefiks rezydentny): ~80 ms.
- Chybienie pamięci podręcznej (zimny prefill): ~800 ms.

10x różnica. Jeśli Twój router osiąga 60-80% trafień pamięci podręcznej prefiksów w replikach, osiągasz wydajność pojedynczej repliki przy pojemności N replik. Jeśli osiąga 10%, osiągasz naiwne skalowanie.

### Wieloregionowe ma nowe ograniczenie — opóźnienie sieciowe

RTT między regionami:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-southeast-1: ~220 ms.

Jeśli routing kieruje żądanie z us-east-1 do gorącego prefiksu w ap-southeast-1, zaoszczędzony prefill (800 → 80 ms) jest przyćmiony przez 440 ms podróży w obie strony. GORGO (badania 2026) czyni to jawnym — minimalizuj `prefill_time + network_latency` łącznie, a nie sam prefill. Często odpowiedzią jest utrzymanie routingu regionalnego, z wyjątkiem masywnych prefiksów wielo-MB, gdzie prefill dominuje.

### Komercyjna „wieloregionowa inferencja" tu nie pomaga

AWS Bedrock cross-region inference automatycznie kieruje żądania do innych regionów podczas presji pojemnościowej. Optymalizuje dostępność, a nie TTFT, i traktuje inferencję jako nieprzezroczystą. GKE Multi-Cluster Gateway to to samo — failover na poziomie usługi, brak świadomości pamięci podręcznej KV.

Nadal potrzebujesz routera świadomego pamięci podręcznej na poziomie aplikacji, nawet gdy ich używasz. One obsługują przypadek „us-east-1 płonie". Routing świadomy pamięci podręcznej obsługuje przypadek TTFT.

### Higiena DR — problem 32% brakujących plików

Szeroko cytowana statystyka 2026: 32% awarii DR LLM zdarza się, ponieważ zespoły wykonały kopię zapasową wag, ale zapomniały:

- `tokenizer.json` lub `tokenizer.model`
- Konfiguracje kwantyzacji (`quantize_config.json`, skale AWQ, punkty zerowe GPTQ)
- Konfiguracje specyficzne dla modelu (skalowanie RoPE, maski attention, szablony czatu)
- Konfiguracja silnika (`vllm_config.yaml`, domyślne próbkowania, manifesty adapterów LoRA)

Poprawką jest minimalny manifest DR z trzema plikami:

1. Wszystkie pliki w repozytorium modelu HF (wagi + konfiguracje + tokenizer).
2. Konfiguracja serwowania specyficzna dla silnika.
3. Manifest wdrożenia (YAML K8s, Dockerfile, blokada zależności).

Plus: przeprowadzaj ćwiczenie DR kwartalnie. Ćwiczenie JPMorgan us-east-1 osiągnęło 22 minuty odzyskiwania w listopadzie 2024 tylko dlatego, że podręcznik był przećwiczony.

### Rezydencja danych jest ortogonalna

Dane PHI klienta UE nie mogą opuścić UE. Jeśli Twój router świadomy pamięci podręcznej wysyła żądanie pochodzące z Paryża do us-east-1 dla dopasowania prefiksu, naruszyłeś GDPR niezależnie od zysku TTFT. Partycjonuj routery według granic rezydencji przed optymalizacją pod kątem pamięci podręcznej.

### Liczby, które powinieneś zapamiętać

- Różnica TTFT trafienie vs chybienie: ~10x (80 ms vs 800 ms na 2K prompcie).
- RTT między regionami US-EU: ~75 ms.
- Awaria DR: 32% brakuje tokenizera/konfiguracji kwantyzacji.
- Failover JPMorgan us-east-1 listopad 2024: 22 minuty (SLA 30 minut).

## Use It

`code/main.py` symuluje trzy strategie routingu (round-robin, świadomy pamięci podręcznej regionalny, świadomy pamięci podręcznej globalny) na wieloregionowym obciążeniu. Raportuje współczynnik trafień pamięci podręcznej, P50/P99 TTFT i rachunek międzyregionowy.

## Ship It

Ta lekcja produkuje `outputs/skill-multi-region-router.md`. Biorąc regiony, ograniczenia rezydencji i SLA, projektuje plan routingu.

## Exercises

1. Uruchom `code/main.py`. Przy jakiej długości promptu routing międzyregionalny bije routing tylko lokalny, przy RTT 75 ms?
2. Twój współczynnik trafień pamięci podręcznej spada z 70% do 12%. Zdiagnozuj trzy możliwe przyczyny i obserwowalne, które potwierdziłyby każdą.
3. Zaprojektuj manifest DR dla modelu 70B skwantowanego AWQ serwowanego w vLLM z 5 adapterami LoRA. Wymień każdy plik i konfigurację.
4. Przedstaw argument, czy Bedrock cross-region inference jest „wystarczające" dla fintech z rygorystycznymi SLO TTFT. Powołaj się na konkretne zachowania.
5. Żądanie pochodzące z Paryża dopasowuje prefiks w us-east-1. Czy kierujesz je? Napisz politykę.

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Cache-aware routing | „inteligentny LB" | Kierowanie na dopasowanie skrótu prefiksu do repliki przechowującej pamięć podręczną KV |
| KV-cache events | „pub-sub pamięci podręcznej" | Repliki publikują dodanie/wywłaszczenie bloku; router indeksuje |
| Prefix hash | „klucz pamięci podręcznej" | Skrót pierwszych N tokenów używany jako wyszukiwanie routera |
| GORGO | „badania routingu międzyregionalnego" | arXiv 2602.11688; opóźnienie sieciowe jako jawny termin |
| Cross-region inference | „Bedrock CRI" | Produkt AWS; failover dostępności, a nie świadomość TTFT |
| DR manifest | „lista kopii zapasowej" | Każdy plik potrzebny do przywrócenia — nie tylko wagi |
| Data residency | „granica GDPR" | Ograniczenie prawne, który region widzi dane użytkownika |
| RTT | „czas podróży w obie strony" | Opóźnienie sieciowe; 75 ms US-EU, 220 ms US-APAC |
| LLM-aware LB | „LB z trafieniem w pamięci podręcznej" | Router świadomy pamięci podręcznej jako kategoria produktu |

## Further Reading

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — wieloregionowe ponowne użycie pamięci podręcznej KV z terminem opóźnienia sieciowego.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — dokumentacja failover dostępności.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — źródło routera świadomego pamięci podręcznej.

(End of file - total 126 lines)