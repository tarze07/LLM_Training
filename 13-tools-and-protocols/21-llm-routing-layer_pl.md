# Warstwa Rutingu LLM — LiteLLM, OpenRouter, Portkey

> Uzależnienie od jednego dostawcy jest kosztowne. Różne obciążenia związane z wywoływaniem narzędzi wymagają różnych modeli. Bramki rutingu zapewniają jedną powierzchnię API, ponawiania, awaryjne przełączanie, śledzenie kosztów i zabezpieczenia. W 2026 roku dominują trzy archetypy: LiteLLM (open-source, samodzielnie hostowany), OpenRouter (zarządzany SaaS), Portkey (klasa produkcyjna, open-source od marca 2026). Ta lekcja wymienia kryteria decyzyjne i przeprowadza przez implementację bramki rutingu w stdlib.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## Learning Objectives

- Rozróżnić opcje rutingu: samodzielnie hostowane, zarządzane i klasy produkcyjnej.
- Zaimplementować łańcuch awaryjny, który ponawia próby w przypadku awarii dostawcy w określonej kolejności priorytetów.
- Śledzić koszt na żądanie i użycie tokenów u różnych dostawców.
- Zdecydować między LiteLLM, OpenRouter i Portkey dla konkretnych ograniczeń produkcyjnych.

## The Problem

Scenariusze, w których routing dostawcy ma znaczenie:

1. **Koszt.** Claude Sonnet kosztuje 3× więcej niż Haiku. Do zadania triage'owego wystarczy Haiku; do zadania syntezy Sonnet jest wart swojej ceny. Kieruj na poziomie żądania.

2. **Awaryjne przełączanie.** OpenAI ma złą godzinę. Każde żądanie zawodzi. Potrzebujesz automatycznego przejścia na Anthropic bez ponownego wdrażania.

3. **Opóźnienie.** Żywy czat UI potrzebuje szybkiego czasu do pierwszego tokena. Pakietowy sumaryzator nie potrzebuje. Kieruj według SLA opóźnienia.

4. **Zgodność.** Użytkownicy z UE muszą pozostać w regionach UE. Kieruj według regionu.

5. **Eksperymenty.** Test A/B dwóch modeli na tym samym obciążeniu. Kieruj według grupy testowej.

Ręczne kodowanie tego wszystkiego dla każdej integracji jest powtarzalne. Bramka rutingu daje jeden interfejs API kompatybilny z OpenAI i obsługuje resztę.

## The Concept

### Kształt proxy kompatybilnego z OpenAI

Wszyscy mówią kształtem OpenAI. Bramka rutingu udostępnia `/v1/chat/completions`, przyjmuje schemat OpenAI i wewnętrznie przekierowuje do Anthropic / Gemini / Cohere / Ollama / czegokolwiek. Klient się nie przejmuje.

### Aliasy modeli

Zamiast `claude-3-5-sonnet-20251022`, twój kod mówi `our_smart_model`. Bramka mapuje aliasy na rzeczywiste modele. Gdy Anthropic wypuści Claude 4, zmieniasz alias po stronie serwera; twój kod nie dotyka niczego.

### Łańcuchy awaryjne

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Bramki definiują to w konfiguracji. Ponawiania liczą się do budżetu, aby kaskady awaryjne nie wybuchły kosztem.

### Buforowanie semantyczne

Identyczne lub prawie identyczne prompty trafiają do pamięci podręcznej zamiast do dostawcy. Oszczędności na powtarzalnych pętlach agenta mogą wynosić 30 do 60 procent. Klucze są oparte na embeddingach; prawie identyczne prompty współdzielą miejsce w pamięci podręcznej.

### Zabezpieczenia

Na poziomie bramki:

- **Maskowanie PII.** Przejście oparte na regex lub ML przed wysłaniem promptów.
- **Naruszenia polityki.** Odrzucanie promptów z zabronioną treścią.
- **Filtry wyjściowe.** Czyszczenie odpowiedzi przed wyciekami.

Portkey i Kong dostarczają narzucających się zabezpieczeń. LiteLLM pozostawia je opcjonalne.

### Limity stawek na klucz

Jeden klucz API = jeden zespół. Budżety na klucz zapobiegają konsumowaniu współdzielonego limitu przez jeden zespół. Większość bramek to obsługuje.

### Kompromisy: samodzielnie hostowane vs zarządzane

| Czynnik | LiteLLM (samodzielnie hostowany) | OpenRouter (zarządzany) | Portkey (produkcyjny) |
|--------|----------------------|----------------------|----------------------|
| Kod | Open source, Python | Zarządzany SaaS | Open source (mar 2026) + zarządzany |
| Konfiguracja | Wdróż proxy | Zarejestruj się | Albo jedno, albo drugie |
| Dostawcy | 100+ | 300+ | 100+ |
| Rozliczenia | Twoje własne klucze | Kredyty OpenRouter | Twoje własne klucze |
| Obserwowalność | OpenTelemetry | Dashboard | Pełny OTel + maskowanie PII |
| Najlepsze dla | Zespołów chcących pełnej kontroli | Szybkiego prototypowania | Produkcji z wymogami zgodności |

LiteLLM wygrywa, gdy masz zespół SRE i chcesz suwerenności danych. OpenRouter wygrywa, gdy chcesz pojedynczej subskrypcji i żadnej infrastruktury. Portkey wygrywa, gdy potrzebujesz zabezpieczeń i zgodności od razu po wyjęciu z pudełka.

### Śledzenie kosztów

Każde żądanie niesie `provider`, `model`, `input_tokens`, `output_tokens`. Pomnóż przez ceny za token na model (pobrane z arkusza cenowego utrzymywanego przez bramkę). Agregacja na użytkownika / zespół / projekt.

### MCP plus routing

Bramka może kierować zarówno wywołania LLM, JAK I żądania próbkowania MCP. Gdy `modelPreferences` w żądaniu próbkowania preferuje konkretny model, bramka tłumaczy to na odpowiedni backend. To miejsce, gdzie Faza 13 · 17 (bramka MCP) i bramka rutingu z tej lekcji czasami łączą się w jedną usługę.

### Strategie rutingu

- **Statyczny priorytet.** Pierwszy na liście; przełącz na błąd.
- **Równoważenie obciążenia.** Round-robin lub ważone.
- **Świadomość kosztów.** Wybierz najtańszy model spełniający wymagania opóźnienia/jakości.
- **Świadomość opóźnienia.** Wybierz najszybszy model w ostatnich N minutach.
- **Świadomość zadania.** Klasyfikator promptów kieruje kodowanie do jednego modelu, a podsumowywanie do innego.

## Use It

`code/main.py` implementuje bramkę rutingu w ~150 liniach: przyjmuje żądania w kształcie OpenAI, tłumaczy na zaślepki dla poszczególnych dostawców, uruchamia łańcuch awaryjny według priorytetu, śledzi koszt na żądanie i stosuje maskowanie PII na wejściach. Uruchom z trzema scenariuszami: normalne żądanie, awaria głównego dostawcy wywołująca awaryjne przełączenie, wyciek PII przechwycony przez maskowanie.

Na co zwrócić uwagę:

- Słownik `ROUTES`: alias -> lista konkretnych dostawców uporządkowana według priorytetu.
- Pętla awaryjna ponawia na 5xx.
- Licznik kosztów mnoży użycie tokenów przez stawki na model.
- Maska PII usuwa wzorce przypominające numery SSN przed przekazaniem dalej.

## Ship It

Ta lekcja produkuje `outputs/skill-routing-config-designer.md`. Mając profil obciążenia (opóźnienie, koszt, zgodność), umiejętność wybiera LiteLLM / OpenRouter / Portkey i produkuje konfigurację rutingu.

## Exercises

1. Uruchom `code/main.py`. Wyzwól scenariusz awarii; potwierdź, że awaryjne przełączenie trafia na drugiego dostawcę, a koszt jest poprawnie przypisany.

2. Dodaj buforowanie semantyczne: SHA256 promptu jest kluczem wyszukiwania; trafienia w pamięci podręcznej zwracają natychmiast. Zmierz oszczędności kosztów na powtarzalnym wywołaniu.

3. Dodaj klasyfikator promptów, który kieruje prompty "kod ..." do aliasu faworyzującego inteligencję, a prompty "podsumuj ..." do aliasu faworyzującego szybkość.

4. Zaprojektuj budżety na zespół: każdy zespół ma miesięczny limit wydatków; bramka odrzuca żądania po osiągnięciu limitu. Wybierz szczegółowość egzekwowania (na żądanie lub w oknie).

5. Przeczytaj dokumentację LiteLLM, OpenRouter i Portkey obok siebie. Wymień jedną funkcję, którą każdy z nich dostarcza, a której nie mają pozostałe dwa.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routing gateway | "Proxy LLM" | Warstwa jednej powierzchni API przed wieloma dostawcami |
| OpenAI-compatible | "Mówi schematem OpenAI" | Przyjmuje kształt `/v1/chat/completions`, tłumaczy na dowolny backend |
| Model alias | "our_smart_model" | Nazwa w twoim kodzie, którą bramka mapuje na konkretny model |
| Fallback chain | "Lista ponawiania" | Uporządkowana lista dostawców próbowanych w przypadku awarii |
| Semantic caching | "Pamięć podręczna embeddingów promptów" | Kluczem jest embedding promptu; bliskie duplikaty współdzielą trafienie w pamięci |
| Guardrails | "Filtry wejścia/wyjścia" | Maskują PII, odrzucają naruszenia polityki |
| Per-key rate limit | "Budżet zespołu" | Limit przypisany do klucza API |
| Cost tracking | "Wydatek na żądanie" | Agregacja użycia tokenów × cena za model |
| LiteLLM | "Otwarte proxy" | Bramka rutingu OSS do samodzielnego hostowania |
| OpenRouter | "Zarządzany SaaS" | Hostowana bramka z rozliczeniem opartym na kredytach |
| Portkey | "Opcja produkcyjna" | Open-source + zarządzana z wbudowanymi zabezpieczeniami |

## Further Reading

- [LiteLLM — docs](https://docs.litellm.ai/) — samodzielnie hostowana bramka rutingu
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — zarządzany SaaS rutingu
- [Portkey — docs](https://portkey.ai/docs) — produkcyjny routing z zabezpieczeniami
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — przewodnik decyzyjny
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — przegląd dostawców