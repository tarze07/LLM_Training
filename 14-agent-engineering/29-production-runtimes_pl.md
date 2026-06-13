# Środowiska produkcyjne: Kolejka, Zdarzenie, Cron

> Produkcyjni agenci działają na sześciu kształtach środowiska uruchomieniowego: request-response, strumieniowanie, trwałe wykonanie, tło oparte na kolejce, sterowane zdarzeniami i zaplanowane. Wybierz kształt przed wyborem frameworka. Obserwowalność jest nośna na każdym kształcie.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## Learning Objectives

- Wymień sześć produkcyjnych kształtów środowiska uruchomieniowego i dopasuj każdy do frameworka / wzorca produktu.
- Wyjaśnij, dlaczego trwałe wykonanie (LangGraph) ma znaczenie dla zadań długoterminowych.
- Opisz środowisko sterowane zdarzeniami i kiedy pasuje Claude Managed Agents.
- Wyjaśnij twierdzenie o obserwowalności jako nośnej dla agentów wieloetapowych.

## Problem

Produkcyjni agenci zawodzą w sposób, którego notatnik Jupyter nie ujawnia: przekroczenia czasu sieci na kroku 37, użytkownik rozłącza się w trakcie rozmowy głosowej, zadanie cron umiera przy restarcie maszyny, pracownik tła kończy pamięć. Kształt środowiska uruchomieniowego określa, które awarie są do przeżycia.

## Koncepcja

### Request-response

- Synchroniczne HTTP. Użytkownik czeka na zakończenie.
- Opłacalne tylko dla krótkich zadań (<30s).
- Stosy: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Obserwowalność: standardowe logi dostępu HTTP + spany OTel.

### Strumieniowanie

- SSE lub WebSocket dla progresywnego wyjścia.
- LiveKit rozszerza to na WebRTC dla głosu/wideo (Lekcja 22).
- Stosy: dowolny framework z obsługą strumieniowania + frontend obsługujący SSE/WS.
- Obserwowalność: czas na fragment, opóźnienie pierwszego tokena, opóźnienie ogona.

### Trwałe wykonanie

- Stan punktu kontrolnego po każdym kroku; automatycznie wznawia po awarii.
- Model aktora AutoGen v0.4 izoluje awarie do jednego agenta (Lekcja 14).
- Główny wyróżnik LangGraph (Lekcja 13).
- Niezbędne, gdy liczba kroków jest nieznana, a koszt odzyskiwania wysoki.

### Oparte na kolejce / tło

- Zadanie trafia do kolejki, pracownicy je podejmują, wyniki wracają przez webhooki lub pub/sub.
- Niezbędne dla agentów długoterminowych (dziesiątki do setek kroków na zadanie, według ogłoszenia Anthropic o computer use).
- Stosy: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), własne.
- Obserwowalność: głębokość kolejki, rozkład opóźnień na zadanie, rozmiar DLQ.

### Sterowane zdarzeniami

- Agenci subskrybują wyzwalacze: nowy e-mail, otwarty PR, ogień cron.
- Claude Managed Agents obsługuje to od razu (Lekcja 17).
- CrewAI Flows (Lekcja 15) strukturyzuje deterministyczne przepływy pracy sterowane zdarzeniami.
- Obserwowalność: źródło wyzwalacza, opóźnienie od zdarzenia do startu, opóźnienie agenta.

### Zaplanowane

- Agenci w kształcie cron, którzy uruchamiają się okresowo.
- Połącz z trwałym wykonaniem, aby nieudane nocne uruchomienie wznowiło się przy następnym tiknięciu.
- Stosy: Kubernetes CronJob + trwały framework; hostowane (Render cron, Vercel cron).

### Wzorce wdrożeniowe 2026

- **CrewAI Flows** dla produkcji sterowanej zdarzeniami.
- **Agno** bezstanowe FastAPI dla mikrousług Python.
- **Mastra** adaptery serwerowe (Express, Hono, Fastify, Koa) do osadzania.
- **Pipecat Cloud / LiveKit Cloud** dla zarządzanego głosu (Lekcja 22).
- **Claude Managed Agents** dla hostowanego długotrwałego asynchronicznego.

### Obserwowalność jest nośna

Bez spanów OpenTelemetry GenAI (Lekcja 23) plus backendu Langfuse/Phoenix/Opik (Lekcja 24) nie możesz debugować wieloetapowego agenta, który zawiódł na kroku 40. To nie jest opcjonalne dla produkcji. To różnica między "debugujemy szybko" a "odtwarzamy od zera z większą ilością logowania."

### Gdzie środowiska produkcyjne zawodzą

- **Zły wybór kształtu.** Wybór request-response dla 5-minutowego zadania. Użytkownicy się rozłączają; pracownicy się piętrzą; ponowienia się kumulują.
- **Brak DLQ.** Pracownicy kolejki bez martwych liter. Nieudane zadania znikają.
- **Nieprzezroczysta praca w tle.** Agent tła działa bez eksportu śladów. Awarie są niewidoczne, dopóki użytkownik ich nie zgłosi.
- **Pomijanie trwałego stanu.** Każde uruchomienie > 30 sekund, gdzie nie stać cię na restart, potrzebuje trwałego wykonania.

## Build It

`code/main.py` to demo wielu kształtów w stdlib:

- Endpoint request-response (zwykła funkcja).
- Handler strumieniowania (generator).
- Pracownik oparty na kolejce z DLQ.
- Rejestr wyzwalaczy zdarzeń.
- Planista w kształcie cron.

Uruchom:

```bash
python3 code/main.py
```

Wynik: pięć śladów pokazujących zachowanie każdego kształtu na tym samym zadaniu. Ta sama logika agenta, różne zewnętrzne powłoki. Trwałe wykonanie (szósty kształt) jest celowo omówione w Lekcji 13 z punktami kontrolnymi LangGraph.

## Use It

- **Request-response** dla UX w stylu czatu.
- **Strumieniowanie** dla progresywnych odpowiedzi.
- **Trwałe** dla zadań długoterminowych.
- **Kolejka** dla wsadu / asynchronicznego / długotrwałego.
- **Zdarzenie** dla reaktywności agenta.
- **Cron** dla utrzymania (konsolidacja pamięci, ewaluacje, raporty kosztów).

## Ship It

`outputs/skill-runtime-shape.md` wybiera kształt środowiska uruchomieniowego dla zadania i okablowuje wymagania obserwowalności.

## Exercises

1. Przenieś swoją pętlę ReAct z Lekcji 01 na wszystkie sześć kształtów w swoim stosie. Który kształt pasuje do której powierzchni produktu?
2. Dodaj DLQ do dema opartego na kolejce. Symuluj 10% awarii zadań; pokaż rozmiar DLQ.
3. Napisz agenta ewaluacyjnego wyzwalanego cron, który uruchamia się nocnie na twoich 20 najlepszych śladach z dnia.
4. Zaimplementuj strumieniowanie z przeciwciśnieniem: jeśli klient jest wolny, wstrzymaj agenta. Jak to współdziała z budżetem tur?
5. Przeczytaj dokumentację Claude Managed Agents. Kiedy przeniósłbyś samodzielnie hostowanego agenta długoterminowego do zarządzanego?

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| Request-response | "Synchroniczne" | Użytkownik czeka; tylko krótkie zadania |
| Streaming | "SSE / WS" | Progresywne wyjście; lepszy UX; opóźnienie obserwowalne na fragment |
| Durable execution | "Wznów po awarii" | Stan punktu kontrolnego; restart na ostatnim kroku |
| Queue-based | "Zadania tła" | Producent / pula pracowników / DLQ |
| Event-driven | "Oparte na wyzwalaczach" | Agent reaguje na zdarzenia zewnętrzne |
| DLQ | "Kolejka martwych liter" | Parking dla nieudanych zadań |
| Claude Managed Agents | "Hostowany harness" | Hostowane przez Anthropic długotrwałe asynchroniczne z cachowaniem + kompakcją |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — szczegóły trwałego wykonania
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hostowane długotrwałe asynchroniczne
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "dziesiątki do setek kroków na zadanie"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — izolacja błędów modelu aktora