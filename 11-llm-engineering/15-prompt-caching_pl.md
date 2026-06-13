# Buforowanie promptów i buforowanie kontekstu

> Twój prompt systemowy ma 4000 tokenów. Twój kontekst RAG ma 20 000 tokenów. Wysyłasz oba z każdym żądaniem. I płacisz za oba — za każdym razem. Buforowanie promptów pozwala dostawcy utrzymać ten prefiks po swojej stronie i obciążyć Cię 10% normalnej stawki przy ponownym użyciu. Użyte prawidłowo, obniża koszt inferencji o 50–90%, a opóźnienie pierwszego tokena o 40–85%.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 minutes

## Problem

Agent programistyczny wysyła ten sam 15 000-tokenowy prompt systemowy do Claude'a w każdej turze rozmowy. Dwadzieścia tur przy $3/M input tokenów to $0,90 samego kosztu wejściowego — zanim jeszcze pojawi się jakakolwiek wiadomość użytkownika. Pomnóż przez 10 000 dziennych rozmów i rachunek sięga $9 000/dzień za tekst, który nigdy się nie zmienia.

Nie możesz zmniejszyć promptu bez pogorszenia jakości. Nie możesz uniknąć jego wysyłania — model potrzebuje go w każdej turze. Jedynym rozwiązaniem jest przestać płacić pełną cenę za prefiks, który dostawca już widział.

Tym rozwiązaniem jest buforowanie promptów. Anthropic wdrożył je w sierpniu 2024 (z wariantem o przedłużonym TTL 1 godziny w 2025), OpenAI zautomatyzował je później w tym samym roku, Google wprowadził jawne buforowanie kontekstu wraz z Gemini 1.5, a wszystkie trzy oferują je teraz jako funkcję pierwszej klasy na swoich czołowych modelach.

## Koncepcja

![Buforowanie promptów: zapisz raz, czytaj tanio](../assets/prompt-caching.svg)

**Mechanizm.** Gdy prefiks żądania pasuje do prefiksu z niedawnego żądania, dostawca serwuje KV-cache z poprzedniego uruchomienia zamiast ponownie kodować tokeny. Płacisz niewielką premię za zapis za pierwszym razem i dużą zniżkę za odczyt za każdym kolejnym razem.

**Trzy smaki dostawców w 2026.**

| Provider | API style | Hit discount | Write premium | Default TTL | Min cacheable |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Explicit `cache_control` markers on content blocks | 90% off input | 25% surcharge | 5 min (extendable to 1 hour) | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) |
| OpenAI | Automatic prefix detection | 50% off input | none | Up to 1 hour (best-effort) | 1,024 tokens |
| Google (Gemini) | Explicit `CachedContent` API | Storage-billed; read at ~25% of normal | Storage fee per token·hour | User-set (default 1 hour) | 4,096 tokens (Flash), 32,768 (Pro) |

**Niezmiennik.** Wszyscy trzej buforują tylko prefiksy. Jeśli jakikolwiek token różni się między żądaniami, wszystko po pierwszym różniącym się tokenie jest chybieniem. Umieść *stabilne* części na górze, a *zmienne* części na dole.

### Układ przyjazny buforowaniu

```
[prompt systemowy]          <-- to buforuj
[definicje narzędzi]        <-- to buforuj
[przykłady few-shot]        <-- to buforuj
[pobrane dokumenty]         <-- buforuj jeśli ponownie użyte, w przeciwnym razie nie
[historia rozmowy]          <-- buforuj do ostatniej tury
[bieżąca wiadomość użytkownika] <-- nigdy nie buforuj (inna za każdym razem)
```

Naruszenie kolejności — umieszczenie wiadomości użytkownika nad promptem systemowym, przeplatanie dynamicznych pobrań między przykładami few-shot — powoduje, że cache nigdy nie trafia.

### Obliczenie progu opłacalności

25% premia za zapis u Anthropica oznacza, że buforowany blok musi zostać odczytany co najmniej dwa razy, aby zaoszczędzić pieniądze netto. 1 zapis + 1 odczyt daje średnio 0,675x kosztu na żądanie (oszczędność 32%); 1 zapis + 10 odczytów daje średnio 0,205x (oszczędność 80%). Zasada kciuka: buforuj wszystko, czego spodziewasz się użyć ponownie co najmniej 3 razy w ciągu TTL.

## Zbuduj to

### Krok 1: Buforowanie promptów Anthropic z jawnymi znacznikami

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

Znacznik `cache_control` informuje Anthropica, aby przechowywał blok przez 5 minut. Ponowne użycie w tym oknie trafia; użycie po wygaśnięciu powoduje ponowny zapis.

**Pola użycia w odpowiedzi:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # płatne 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # płatne 0.1x
```

Sprawdzaj oba pola w CI — jeśli `cache_read_input_tokens` pozostaje na zero między żądaniami, Twoje klucze cache dryfują.

### Krok 2: Rozszerzony TTL na godzinę

W przypadku długotrwałych zadań wsadowych domyślny 5-minutowy TTL wygasa między zadaniami. Ustaw `ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1-godzinny TTL kosztuje 2x premię za zapis (50% ponad bazę zamiast 25%), ale szybko się zwraca przy każdym zadaniu wsadowym ponownie używającym prefiksu więcej niż 5 razy.

### Krok 3: Automatyczne buforowanie OpenAI

OpenAI nie daje nic do konfiguracji. Każdy prefiks powyżej 1024 tokenów, który pasuje do niedawnego żądania, automatycznie otrzymuje 50% zniżki.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # długi i stabilny
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # część objęta zniżką
```

Ta sama zasada układu przyjaznego buforowaniu ma zastosowanie. Dwie rzeczy zabijają cache OpenAI, które nie zabijają cache Anthropica: zmiana pola `user` (używanego jako składnik klucza cache) i zmiana kolejności narzędzi.

### Krok 4: Jawne buforowanie kontekstu Gemini

Gemini traktuje cache jako obiekt pierwszej klasy, który tworzysz i nazywasz:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini nalicza opłatę za przechowywanie za token·godzinę przez cały czas życia cache, a odczyty kosztują ~25% normalnej stawki wejściowej. To odpowiedni kształt, gdy wielokrotnie używasz tego samego ogromnego promptu w wielu sesjach przez kilka dni.

### Krok 5: Pomiar wskaźnika trafień w produkcji

Zobacz `code/main.py` dla symulowanego kalkulatora trzech dostawców, który śledzi liczbę zapisów/odczytów/chybień i oblicza średni koszt na 1000 żądań. Uzależnij wdrożenia od docelowego wskaźnika trafień — większość produkcyjnych konfiguracji Anthropica powinna osiągać >80% frakcji odczytów po rozgrzaniu.

## Pułapki, które wciąż występują w 2026

- **Dynamiczne znaczniki czasu na górze.** `"Current time: 2026-04-22 15:30:02"` na górze promptu systemowego. Każde żądanie jest chybieniem. Przesuń znaczniki czasu poniżej punktu podziału cache.
- **Zmiana kolejności narzędzi.** Serializuj narzędzia w stabilnej kolejności — przetasowanie słownika między wdrożeniami psuje każde trafienie.
- **Bliskie duplikaty w wolnym tekście.** "You are helpful." vs "You are a helpful assistant." — jeden bajt różnicy = całkowite chybienie.
- **Zbyt małe bloki.** Anthropic egzekwuje próg 1024 tokenów (2048 dla Haiku). Mniejsze bloki są po cichu niebuforowane.
- **Ślepe pulpity kosztów.** Podziel "tokeny wejściowe" na buforowane i niebuforowane. W przeciwnym razie spadek ruchu wygląda jak sukces cache.

## Użyj tego

Stos buforowania w 2026:

| Sytuacja | Wybór |
|-----------|------|
| Agent ze stabilnym promptem systemowym 10k+, wiele tur | Anthropic `cache_control` z 5-minutowym TTL |
| Zadanie wsadowe ponownie używające prefiksu przez 30+ minut | Anthropic z `ttl: "1h"` |
| Endpointy serverless na GPT-5, bez własnej infrastruktury | OpenAI automatyczne (po prostu uczyń prefiks stabilnym i długim) |
| Wielodniowe użycie ogromnego korpusu kodu/dokumentów | Gemini jawne `CachedContent` |
| Fallback między dostawcami | Utrzymuj identyczny układ buforowalnego prefiksu u wszystkich dostawców, aby każde trafienie działało |

Połącz z buforowaniem semantycznym (Faza 11 · 11) dla warstwy wiadomości użytkownika: buforowanie promptów obsługuje ponowne użycie *identycznych tokenów*, buforowanie semantyczne obsługuje ponowne użycie *identycznego znaczenia*.

## Wdróż to

Zapisz `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## Ćwiczenia

1. **Łatwe.** Weź 10-turową rozmowę z 5000-tokenowym promptem systemowym przeciwko Claude'owi. Uruchom ją bez `cache_control`, a następnie z. Raportuj rachunek za tokeny wejściowe dla każdego przypadku.
2. **Średnie.** Napisz harness testowy, który dla danego szablonu promptu i dziennika żądań oblicza oczekiwany wskaźnik trafień i oszczędności kosztów dla każdego dostawcy (Anthropic 5m, Anthropic 1h, OpenAI automatyczne, Gemini jawne).
3. **Trudne.** Zbuduj optymalizator układu: dla danego promptu i listy pól oznaczonych `stable=True/False`, przepisz prompt tak, aby umieścić pojedynczy punkt podziału cache w najbardziej przyjaznej dla cache pozycji bez utraty informacji. Zweryfikuj na prawdziwym endpointcie Anthropica.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Prompt caching | "Sprawia, że długie prompty są tanie" | Ponowne użycie KV-cache dostawcy dla pasujących prefiksów; 50-90% zniżki na powtarzające się tokeny wejściowe. |
| `cache_control` | "Znacznik Anthropica" | Atrybut bloku treści deklarujący "wszystko do tego miejsca jest buforowalne"; `{"type": "ephemeral"}`. |
| Cache write | "Płacenie premii" | Pierwsze żądanie, które wypełnia cache; rozliczane ~1.25x stawki wejściowej u Anthropica, darmowe u OpenAI. |
| Cache read | "Zniżka" | Kolejne żądania pasujące do prefiksu; rozliczane po 10% (Anthropic), 50% (OpenAI), ~25% (Gemini). |
| TTL | "Jak długo żyje" | Sekundy, przez które cache pozostaje ciepły; Anthropic domyślnie 5m (rozszerzalny do 1h), OpenAI best-effort do 1h, Gemini ustawiany przez użytkownika. |
| Extended TTL | "1-godzinny cache Anthropica" | `{"type": "ephemeral", "ttl": "1h"}`; 2x premia za zapis, ale opłacalny przy ponownym użyciu wsadowym. |
| Prefix match | "Dlaczego mój cache chybił" | Cache trafia tylko wtedy, gdy każdy token od początku do punktu podziału jest bajtowo identyczny. |
| Context caching (Gemini) | "Ten jawny" | Nazwany obiekt cache Google rozliczany za przechowywanie; najlepszy do wielodniowego użycia dużych korpusów. |

## Dalsze czytanie

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, 1h TTL, break-even tables.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic prefix matching.
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API and storage pricing.
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — original launch post with latency numbers.
- Phase 11 · 05 (Context Engineering) — where to slice the prompt so the cache can land.
- Phase 11 · 11 (Caching and Cost) — pair prompt caching with a semantic cache on user messages.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — the KV-cache memory model that prompt caching exposes to users; explains why a cached prefix is ~10× cheaper to reread than to recompute.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill is the phase prompt caching shortcuts; this paper explains why TTFT drops dramatically on cache hit while TPOT is unaffected.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching sits alongside speculative decoding, Flash Attention, and MQA/GQA as levers that bend the inference cost curve; read this for the other three.