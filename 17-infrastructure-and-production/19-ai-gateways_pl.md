# Bramki AI — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Bramka znajduje się między Twoimi aplikacjami a dostawcami modeli. Podstawowe funkcje to routing dostawców, fallback, ponowienia, ograniczanie szybkości, referencje sekretów, obserwowalność, zabezpieczenia. Podział rynku w 2026: **LiteLLM** jest MIT OSS z 100+ dostawcami, kompatybilny z OpenAI, ale załamuje się przy ~2000 RPS (8 GB pamięci, kaskadowe awarie w opublikowanych benchmarkach); najlepszy dla Python, <500 RPS, dev/prototypowanie. **Portkey** jest pozycjonowany na płaszczyźnie sterowania (zabezpieczenia, maskowanie PII, wykrywanie jailbreaków, ślady audytowe), przeszedł na open-source Apache 2.0 w marcu 2026, narzut latencji 20-40 ms, poziom produkcyjny $49/miesiąc. **Kong AI Gateway** zbudowany na Kong Gateway — benchmark Konga na tych samych 12 CPU: 228% szybciej niż Portkey, 859% szybciej niż LiteLLM; cennik $100/model/miesiąc (max 5 na poziomie Plus); dopasowanie enterprise jeśli już jesteś na Kong. **Bifrost** (Maxim AI) — automatyczne ponowienia z konfigurowalnym backoffem, fallback do Anthropic przy 429 OpenAI. **Cloudflare / Vercel AI Gateways** — zarządzane, zero-ops, podstawowe ponowienia. Rezydencja danych napędza decyzję o samodzielnym hostowaniu; Portkey i Kong znajdują się pośrodku z OSS + opcjonalnym zarządzaniem.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Learning Objectives

- Wymienić sześć podstawowych funkcji bramki (routing, fallback, ponowienia, ograniczenia szybkości, sekrety, obserwowalność, zabezpieczenia).
- Dopasować cztery bramki 2026 (LiteLLM, Portkey, Kong AI, Bifrost) do limitów skali i przypadków użycia.
- Przytoczyć benchmark Konga (228% vs Portkey, 859% vs LiteLLM) i wyjaśnić, dlaczego ma znaczenie dla >500 RPS.
- Wybrać między samodzielnym hostowaniem a zarządzanym w zależności od rezydencji danych i budżetu operacyjnego.

## The Problem

Twój produkt wywołuje OpenAI, Anthropic i samodzielnie hostowanego Llamę. Każdy dostawca ma inne SDK, model błędów, limit szybkości i schemat autoryzacji. Potrzebujesz failover (jeśli OpenAI 429, spróbuj Anthropic), pojedynczy magazyn poświadczeń, ujednoliconą obserwowalność i ograniczenia szybkości na tenanta.

Wynalezienie tego od nowa na poziomie aplikacji łączy każdą usługę z każdym dostawcą. Warstwa bramki konsoliduje to w jeden proces z jednym API (zwykle kompatybilnym z OpenAI), które rozdziela do dostawców.

## The Concept

### Sześć podstawowych funkcji

1. **Routing dostawców** — OpenAI, Anthropic, Gemini, samodzielnie hostowane itp. za jednym API.
2. **Fallback** — przy 429, 5xx lub awarii jakości, ponów gdzie indziej.
3. **Ponowienia** — wykładniczy backoff, ograniczone próby.
4. **Ograniczenia szybkości** — na tenanta, na klucz, na model.
5. **Referencje sekretów** — pobieranie poświadczeń z vaultu w czasie wykonania (nigdy w aplikacji).
6. **Obserwowalność** — OTel + atrybuty GenAI (Phase 17 · 13) + atrybucja kosztów.
7. **Zabezpieczenia** — maskowanie PII, wykrywanie jailbreaków, filtry dozwolonych tematów.

### LiteLLM — MIT OSS, Python

- 100+ dostawców, kompatybilny z OpenAI, konfiguracja routera, fallback, podstawowa obserwowalność.
- Załamuje się przy ~2000 RPS w benchmarku Konga; 8 GB pamięci, kaskadowe awarie pod ciągłym obciążeniem.
- Najlepsze dopasowanie: aplikacja Python, <500 RPS, bramki dev/staging, eksperymentalny routing.
- Koszt: $0 za OSS; istnieje darmowy poziom w chmurze.

### Portkey — pozycjonowanie na płaszczyźnie sterowania

- Apache 2.0 OSS od marca 2026. Zabezpieczenia, maskowanie PII, wykrywanie jailbreaków, ślady audytowe.
- 20-40 ms narzutu latencji na żądanie.
- $49/miesiąc za poziom produkcyjny z retencją + SLA.
- Najlepsze dopasowanie: regulowane branże potrzebujące zabezpieczeń + obserwowalności w pakiecie.

### Kong AI Gateway — podejście skali

- Zbudowany na Kong Gateway (dojrzały produkt bramki API, lua+OpenResty).
- Własny benchmark Konga na odpowiedniku 12 CPU: 228% szybciej niż Portkey, 859% szybciej niż LiteLLM.
- Cennik: $100/model/miesiąc, max 5 na poziomie Plus.
- Najlepsze dopasowanie: już na Kong; >1000 RPS; gotowość do licencjonowania.

### Bifrost (Maxim AI)

- Automatyczne ponowienia z konfigurowalnym backoffem.
- Fallback do Anthropic przy 429 OpenAI to kanoniczny przepis.
- Nowszy uczestnik; komercyjny.

### Cloudflare AI Gateway / Vercel AI Gateway

- Zarządzane, zero-ops. Podstawowe ponowienia i obserwowalność.
- Najlepsze dopasowanie: aplikacje JavaScript na brzegu sieci na Cloudflare/Vercel.
- Ograniczone w porównaniu z Kong/Portkey w zakresie zabezpieczeń i ograniczeń szybkości.

### Samodzielne hostowanie vs zarządzane

Rezydencja danych jest czynnikiem wymuszającym. Ochrona zdrowia i finanse domyślnie samodzielnie hostują (LiteLLM lub Portkey OSS lub Kong). Produkty konsumenckie domyślnie zarządzane (Cloudflare AI Gateway) lub średni poziom (Portkey zarządzane). Hybryda: samodzielnie hostowane dla regulowanego tenanta, zarządzane dla innych.

### Budżet latencji

- LiteLLM: 5-15 ms typowy narzut.
- Portkey: 20-40 ms narzut.
- Kong: 3-8 ms narzut.
- Cloudflare/Vercel: 1-3 ms narzut (przewaga brzegu sieci).

Latencja bramki dodaje się bezpośrednio do TTFT. Dla SLA TTFT P99 < 100 ms, Kong lub Cloudflare. Dla P99 < 500 ms, dowolna.

### Semantyka ograniczeń szybkości ma znaczenie

Prosty token-bucket działa do umiarkowanej skali. Multi-tenant wymaga okna przesuwnego + dodatku na burst + warstwowania na tenanta. LiteLLM dostarcza token-bucket; Kong dostarcza okno przesuwne; Portkey dostarcza warstwowanie.

### Bramka + obserwowalność + routing komponują się

Phase 17 · 13 (obserwowalność) + 16 (routing modeli) + 19 (bramki) to ta sama warstwa w produkcji. Wybierz jedno narzędzie pokrywające wszystkie trzy lub połącz je starannie: większość wdrożeń 2026 łączy Helicone (obserwowalność) lub Portkey (zabezpieczenia) z Kong (skala) dla rozdzielonych ról.

### Liczby, które warto zapamiętać

- LiteLLM: załamuje się przy ~2000 RPS, 8 GB pamięci.
- Portkey: 20-40 ms narzut; Apache 2.0 od marca 2026.
- Kong: 228% szybciej niż Portkey, 859% szybciej niż LiteLLM.
- Cennik Kong: $100/model/miesiąc, max 5 na poziomie Plus.
- Cloudflare/Vercel: 1-3 ms narzut na brzegu sieci.

## Use It

`code/main.py` symuluje routing bramki z fallbackiem między 3 dostawcami przy wstrzykiwaniu 429/5xx. Raportuje latencję, wskaźnik ponowień i wskaźnik trafień fallbacku.

## Ship It

Ta lekcja produkuje `outputs/skill-gateway-picker.md`. Na podstawie skali, postawy operacyjnej, zgodności i budżetu latencji wybiera bramkę.

## Exercises

1. Uruchom `code/main.py`. Skonfiguruj fallback OpenAI→Anthropic→samodzielnie hostowane. Jaki jest oczekiwany wskaźnik trafień przy 5% wskaźniku błędów dostawcy?
2. Twoje SLA to TTFT P99 < 200 ms przy bazowej 300 ms. Które bramki mieszczą się w budżecie?
3. Klient z ochrony zdrowia wymaga samodzielnego hostowania + maskowania PII + audytu. Wybierz Portkey OSS lub Kong.
4. Porównaj LiteLLM vs Kong: przy jakim pułapie RPS zespół powinien migrować?
5. Zaprojektuj politykę ograniczania szybkości dla SaaS multi-tenant: poziom darmowy, próbny, płatny. Token-bucket czy okno przesuwne?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## Further Reading

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)