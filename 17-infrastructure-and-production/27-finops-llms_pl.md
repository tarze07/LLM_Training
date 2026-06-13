# FinOps dla LLM — Ekonomika Jednostkowa i Atrybucja Wielotenantowa

> Tradycyjny FinOps załamuje się na wydatkach LLM. Koszty to transakcje tokenowe, nie czas korzystania z zasobów. Tagi nie mapują się — wywołanie API to transakcja, nie zasób. Decyzje inżynieryjne (projekt promptu, okno kontekstu, długość wyjścia) są decyzjami finansowymi. Instrukcja 2026 ma trzy wymiary atrybucji do instrumentowania od dnia pierwszego: na użytkownika (`user_id`) dla cennika miejscowego i ekspansji, na zadanie (`task_id` + `route`) dla kosztu powierzchni produktu i priorytetyzacji, na tenanta (`tenant_id`) dla ekonomiki jednostkowej i odnowienia. Cztery warstwy tokenów — prompt, narzędzie, pamięć, odpowiedź — jeden koszyk ukrywa koszt. Drabina egzekwowania dla produktów wielotenantowych: ograniczenia szybkości na tenanta (2-3x oczekiwanego szczytu, jasne 429 + retry-after); dzienny limit wydatków (1.5-3x umownego pułapu; uruchamia zaostrzenie ograniczeń + alarm); wyłączniki awaryjne przy z-score wydatków > 4 (auto-pauza + zgłoś dyżurnego). Wzorce atrybucji: taguj-i-agreguj, łącznik telemetrii (trace-ID → rozliczenia; najwyższa dokładność), próbkowanie-i-ekstrapolacja, alokacja oparta na modelu, event-sourced, strumieniowanie w czasie rzeczywistym. Metryka jednostkowa: koszt na rozwiązane zapytanie, koszt na wygenerowany artefakt — nie $/M tokenów. Retrospektywne tagowanie zawsze zawodzi; instrumentuj przy tworzeniu żądania.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 minutes

## Learning Objectives

- Wyjaśnić, dlaczego tradycyjny FinOps (tagi + poziomy) załamuje się na wydatkach LLM i wymienić trzy nowe wymiary atrybucji.
- Wymienić cztery warstwy tokenów (prompt, narzędzie, pamięć, odpowiedź) i dlaczego rozliczanie jednym koszykiem ukrywa koszt.
- Zaprojektować drabinę egzekwowania (ograniczenia → limit wydatków → wyłącznik awaryjny) dla produktu wielotenantowego.
- Wybrać metrykę jednostkową (koszt na rozwiązane zapytanie / artefakt) zamiast $/M tokenów.

## The Problem

Twój rachunek mówi $40,000. Nie wiesz:
- Który tenant to wydał.
- Która funkcja produktu to napędziła.
- Czy którykolwiek użytkownik był nadużywający.
- Czy rozdęcie promptu, wywołania narzędzi czy amplifikacja pamięci były winowajcą.

Tagowanie-i-agregacja po stronie dostawcy działa dla zasobów chmurowych (EC2, S3), gdzie tagi propagują się do pozycji na fakturze. Wywołania API LLM nie tagują się automatycznie — musisz oznaczyć user/task/tenant w miejscu wywołania i przenieść przez system. Retrospektywna atrybucja zawsze zawodzi w przypadkach brzegowych.

## The Concept

### Trzy wymiary atrybucji

**Na użytkownika** (`user_id`): kto ile kosztuje. Napędza cennik miejscowy, rozmowy o ekspansji, identyfikuje zaawansowanych użytkowników.

**Na zadanie** (`task_id` + `route`): która powierzchnia produktu ile kosztuje. Napędza priorytetyzację funkcji, decyzje o zabiciu drogich funkcji.

**Na tenanta** (`tenant_id`): który klient jest rentowny. Napędza ekonomikę jednostkową, ceny odnowienia, progi poziomów.

Instrumentuj wszystkie trzy w miejscu wywołania od dnia pierwszego. Retrospektywa jest zawsze gorsza.

### Cztery warstwy tokenów

| Warstwa | Przykład | Typowy % całości |
|-------|---------|---------------------|
| Prompt | system + wejście użytkownika | 40-60% |
| Narzędzie | wyniki wywołań narzędzi zwrócone | 20-40% (obciążenia agentowe) |
| Pamięć | poprzednia rozmowa / pobrane dokumenty | 10-30% |
| Odpowiedź | wyjście modelu | 10-30% |

Łączenie wszystkich czterech w jeden koszyk powoduje ślepotę optymalizacyjną. Rozdziel je w swoim schemacie atrybucji.

### Drabina egzekwowania

1. **Ograniczenie szybkości** na tenanta. 2-3x oczekiwanego szczytu. Zwróć 429 z `Retry-After`. Tenant widzi tarcie; brak niespodziewanego rachunku.

2. **Dzienny limit wydatków** na tenanta. 1.5-3x umownego pułapu. Uruchom: zaostrz ograniczenie szybkości + powiadom obsługę klienta.

3. **Wyłącznik awaryjny** przy z-score wydatków > 4 względem bazowej tenanta. Auto-pauza tenanta; zgłoś dyżurnego; eskaluj do operacji + CS.

### Wzorce atrybucji

- **Taguj-i-agreguj**: znacz metadane w nagłówkach; agreguj później. Proste; przybliżone.
- **Łącznik telemetrii**: łącz ślady z rozliczeniami przez ID śladów. Najwyższa dokładność. To, co robią dojrzałe zespoły.
- **Próbkowanie + ekstrapolacja**: próbkuj 5-10%, mnoż. Opłacalne dla przybliżonych wydatków; pomija ogony.
- **Alokacja oparta na modelu**: regresja do wnioskowania o czynniku kosztu. Dla danych historycznych bez tagów.
- **Event-sourced**: koszt jako zdarzenia w strumieniu (Kafka / Kinesis). Czas rzeczywisty.
- **Strumieniowanie w czasie rzeczywistym**: dashboard aktualizujący się sub-sekundowo.

### Koszt na X jest metryką jednostkową

$/M tokenów to żargon dostawcy. Metryki produktu:

- Koszt na rozwiązane zgłoszenie pomocy.
- Koszt na wygenerowany artykuł.
- Koszt na udane zadanie agenta.
- Koszt na minutę sesji użytkownika.

Powiąż koszt z wynikiem produktu. W przeciwnym razie optymalizacja jest niezakotwiczona.

### Kształt śladu atrybucji kosztów

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Emituj przy każdym wywołaniu. Przechowuj w jeziorze danych. Agreguj według wymiaru. Stos obserwowalności Phase 17 · 13 jest tam, gdzie to żyje.

### Stos skumulowanych oszczędności

Stos: cache + batch + route + gateway. Przy wszystkich czterech:
- Cache L2 (Phase 17 · 14): ~10x tańsze wejście.
- Batch (Phase 17 · 15): 50% zniżki.
- Route do taniego modelu (Phase 17 · 16): 60% redukcji kosztów.
- Wydajność bramki (Phase 17 · 19): nadmiarowość + ponowienia.

W najlepszym przypadku skumulowane: ~5-10% naiwnej bazowej. Większość zespołów ma 2-3 dźwignie zaangażowane; niewielu składa wszystkie cztery.

### Liczby, które warto zapamiętać

- Wymiary atrybucji: na użytkownika, na zadanie, na tenanta.
- Cztery warstwy tokenów: prompt, narzędzie, pamięć, odpowiedź.
- Wyłącznik awaryjny: z-score wydatków > 4.
- Metryka jednostkowa: koszt na rozwiązane zapytanie, nie $/M tokenów.
- Skumulowane optymalizacje: ~5-10% bazowej możliwe.

## Use It

`code/main.py` symuluje usługę LLM wielotenantową z trójpoziomową drabiną egzekwowania. Wstrzykuje nadużywającego tenanta i demonstruje uruchomienie wyłącznika awaryjnego.

## Ship It

Ta lekcja produkuje `outputs/skill-finops-plan.md`. Na podstawie produktu i skali projektuje schemat atrybucji i drabinę egzekwowania.

## Exercises

1. Uruchom `code/main.py`. Przy jakim z-score uruchamia się wyłącznik awaryjny? Jak wybierasz próg?
2. Zaprojektuj dashboard kosztów na tenanta, na zadanie. Jakie są 5 widoków, które budujesz najpierw?
3. Twój największy tenant jest ujemny w ekonomice jednostkowej. Zaproponuj trzy interwencje uporządkowane według wpływu na klienta.
4. Oblicz koszt na rozwiązane zgłoszenie dla produktu pomocy: 3M tokenów/zgłoszenie, ~800 zgłoszeń/dzień, stawka cachowana GPT-5.
5. Uzasadnij, czy retrospektywne tagowanie kiedykolwiek może działać. Kiedy jest akceptowalne?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Per-user attribution | "user-level cost" | `user_id` stamped on every call |
| Per-task attribution | "feature cost" | `task_id` + `route` identify product surface |
| Per-tenant attribution | "customer cost" | `tenant_id`; drives unit economics |
| Four token layers | "cost layers" | prompt + tool + memory + response |
| Rate limit | "429 guard" | Per-tenant ceiling enforced at gateway |
| Daily spend cap | "daily ceiling" | Tenant-scoped budget with alert |
| Kill switch | "auto-pause" | Spend z-score > 4 triggers auto-suspension |
| Cost per resolved | "product unit metric" | Cost tied to product outcome, not tokens |
| Telemetry joiner | "trace-to-billing" | Highest-accuracy attribution pattern |
| Stacked optimization | "cache+batch+route+gateway" | Compounding savings to ~5-10% baseline |

## Further Reading

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)