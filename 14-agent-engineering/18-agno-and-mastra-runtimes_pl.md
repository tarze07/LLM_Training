# Agno i Mastra: Produkcyjne Środowiska Uruchomieniowe

> Agno (Python) i Mastra (TypeScript) to para produkcyjnych środowisk uruchomieniowych roku 2026. Agno celuje w mikrosekundową instancję agenta i bezstanowe backendy FastAPI. Mastra dostarcza agentów, narzędzia, przepływy pracy, ujednolicone routowanie modeli i magazyn kompozytowy na podłożu Vercel AI SDK.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Learning Objectives

- Zidentyfikuj cele wydajnościowe Agno i kiedy mają znaczenie.
- Wymień trzy prymitywy Mastry — Agenci, Narzędzia, Przepływy Pracy — oraz obsługiwane adaptery serwerów.
- Wyjaśnij, dlaczego bezstanowy backend FastAPI z zakresem sesji jest zalecaną ścieżką produkcyjną Agno.
- Wybierz Agno vs Mastra dla danego stosu (Python-first vs TypeScript-first).

## Problem

LangGraph, AutoGen, CrewAI są ciężkie frameworkowo. Zespoły, które chcą „tylko pętlę agenta, szybko, w swoim środowisku uruchomieniowym", sięgają po Agno (Python) lub Mastrę (TypeScript). Oba wymieniają część prymitywów należących do frameworka na surową szybkość i ściślejsze dopasowanie do otaczającego stosu.

## Koncepcja

### Agno

- Środowisko Python, dawniej Phi-data.
- „Żadnych grafów, łańcuchów czy pokrętnych wzorców — tylko czysty Python."
- Cele wydajnościowe z ich dokumentacji: ~2μs instancja agenta, ~3,75 KiB pamięci na agenta, ~23 dostawców modeli.
- Ścieżka produkcyjna: bezstanowy backend FastAPI z zakresem sesji. Każde żądanie uruchamia świeżego agenta; stan sesji żyje w bazie danych.
- Natywna multimodalność (tekst, obraz, audio, wideo, plik) i agentowe RAG.

Cele szybkości mają znaczenie, gdy masz tysiące krótkożyjących agentów na sekundę (zbieżność czatów, potoki ewaluacyjne). Mają mniejsze znaczenie, gdy jeden agent działa przez 10 minut.

### Mastra

- TypeScript, zbudowany na Vercel AI SDK.
- Trzy prymitywy: **Agenci**, **Narzędzia** (typowane Zod), **Przepływy Pracy**.
- Ujednolicony Router Modeli — 3300+ modeli u 94 dostawców (marzec 2026).
- Magazyn kompozytowy: pamięć, przepływy pracy, obserwowalność do różnych backendów; ClickHouse zalecany dla obserwowalności na dużą skalę.
- Apache 2.0 z katalogami `ee/` na licencji enterprise z dostępem do kodu źródłowego.
- Adaptery serwerów dla Express, Hono, Fastify, Koa; pierwszorzędna integracja z Next.js i Astro.
- Dostarcza Mastra Studio (localhost:4111) do debugowania.
- 22k+ gwiazdek na GitHub, 300k+ tygodniowych pobrań npm w wersji 1.0 (styczeń 2026).

### Pozycjonowanie

Żadne z nich nie próbuje być LangGraphem. Konkurują na:

- **Dopasowanie językowe.** Agno dla zespołów Python-first; Mastra dla TypeScript-first.
- **Ergonomia środowiska uruchomieniowego.** Agno = blisko zerowego narzutu; Mastra = zintegrowana z ekosystemem Vercel.
- **Obserwowalność.** Oba integrują się z Langfuse/Phoenix/Opik (Lekcja 24), ale Mastra Studio jest pierwszorzędne.

### Kiedy wybrać które

- **Agno** — backend Python, wiele krótkożyjących agentów, silne wymagania wydajnościowe, sklep FastAPI.
- **Mastra** — backend TypeScript, wdrożenie Next.js / Vercel, ujednolicone routowanie modeli wielu dostawców, narzędzia typowane Zod.
- **LangGraph** (Lekcja 13) — gdy trwały stan i jawne rozumowanie grafowe mają większe znaczenie niż surowa szybkość.
- **OpenAI / Claude Agent SDK** — gdy chcesz produktowego kształtu dostawcy (Lekcje 16–17).

### Gdzie ten wzorzec zawodzi

- **Wydajność dla wydajności.** Wybór Agno, bo „2μs" brzmi dobrze, gdy obciążenie to jedno wolne wywołanie agenta na żądanie. Narzut nie jest wąskim gardłem.
- **Zablokowanie w ekosystemie.** Integracja Mastry w stylu Vercel to zaleta na Vercelu, wada poza nim.
- **Zamieszanie licencyjne dla przedsiębiorstw.** Katalogi `ee/` Mastry są z dostępem do kodu źródłowego, a nie Apache 2.0. Przeczytaj licencje, jeśli planujesz fork.

## Build It

Ta lekcja ma charakter głównie porównawczy — pojedynczy artefakt kodu nie oddałby sprawiedliwości obu frameworkom. Zobacz `code/main.py` dla zestawienia obok siebie: minimalny przepływ „uruchom agenta, strumieniuj wynik, utrwal sesję" zaimplementowany dwukrotnie (raz w stylu Agno, raz w stylu Mastry).

Uruchom:

```
python3 code/main.py
```

Dwa strukturalnie różne, ale funkcjonalnie równoważne ślady.

## Use It

- **Agno** — backend Python, który potrzebuje szybkości i kształtu FastAPI.
- **Mastra** — backend TypeScript z wieloma dostawcami i prymitywami przepływów pracy.
- Oba dostarczają pierwszorzędne haki obserwowalności. Oba integrują się z Langfuse.

## Ship It

`outputs/skill-runtime-picker.md` wybiera Agno, Mastrę, LangGraph lub SDK dostawcy na podstawie stosu, budżetu opóźnienia i kształtu operacyjnego.

## Ćwiczenia

1. Przeczytaj dokumentację Agno. Przenieś pętlę ReAct ze stdlib (Lekcja 01) do Agno. Co zniknęło? Co zostało?
2. Przeczytaj dokumentację Mastry. Przenieś tę samą pętlę do Mastry. Co zmieniło się w typowaniu narzędzi (Zod vs nic)?
3. Benchmark: zmierz opóźnienie instancji agenta na swoim stosie. Czy 2μs Agno ma znaczenie dla twojego obciążenia?
4. Zaprojektuj migrację: jeśli używałeś CrewAI w Pythonie, co się psuje po przejściu na Agno?
5. Przeczytaj warunki licencji `ee/` Mastry. Jakie ograniczenia wpłynęłyby na open-source'owy fork?

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| Agno | „Szybcy agenci w Pythonie" | Bezstanowe środowisko uruchomieniowe agenta z zakresem sesji |
| Mastra | „Agenci TypeScript na Vercel AI SDK" | Agenci + Narzędzia + Przepływy Pracy + Router Modeli |
| Ujednolicony Router Modeli | „Dostęp do wielu dostawców" | Pojedynczy klient dla 3300+ modeli u 94 dostawców |
| Magazyn kompozytowy | „Wiele backendów" | Pamięć/przepływy pracy/obserwowalność każdy do innego magazynu |
| Mastra Studio | „Lokalny debugger" | Interfejs localhost:4111 do inspekcji agentów |
| Dostęp do kodu źródłowego | „Nie OSS" | Licencja pozwala czytać kod źródłowy, ale ogranicza użycie komercyjne |

## Dalsza lektura

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — cele wydajnościowe, integracja FastAPI
- [Mastra docs](https://mastra.ai/docs) — prymitywy, adaptery serwerów, Router Modeli
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — alternatywa z grafem stanowym
- [Comet Opik](https://www.comet.com/site/products/opik/) — porównania obserwowalności cytowane przez integracje Mastry