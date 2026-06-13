# AutoGen v0.4: Model aktora i framework agentów

> AutoGen v0.4 (Microsoft Research, styczeń 2025) przeprojektował orkiestrację agentów wokół modelu aktora. Asynchroniczna wymiana wiadomości, agenci sterowani zdarzeniami, izolacja błędów, naturalna współbieżność. Framework jest obecnie w trybie utrzymania, podczas gdy Microsoft Agent Framework (publiczna wersja zapoznawcza październik 2025) staje się następcą.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Learning Objectives

- Opisz model aktora: agenci jako aktorzy, wiadomości jako jedyne IPC, izolacja błędów na aktora.
- Wymień trzy warstwy API AutoGen v0.4 — Core, AgentChat, Extensions — i do czego każda służy.
- Wyjaśnij, dlaczego oddzielenie dostarczania wiadomości od obsługi daje izolację błędów i naturalną współbieżność.
- Zaimplementuj w stdlib środowisko uruchomieniowe aktora w Pythonie i przenieś na nie dwuagentowy przepływ przeglądu kodu.

## The Problem

Większość frameworków agentów jest synchroniczna: jeden agent produkuje, jeden agent konsumuje, w stosie wywołań. Awarie niszczą stos. Współbieżność jest doczepiona. Dystrybucja wymaga przepisania.

Odpowiedź AutoGen v0.4: model aktora. Każdy agent jest aktorem z prywatną skrzynką odbiorczą. Wiadomości są jedyną interakcją. Środowisko uruchomieniowe oddziela dostarczanie od obsługi. Awarie izolują się do jednego aktora. Współbieżność jest natywna. Dystrybucja to tylko inny transport.

## The Concept

### Aktorzy

Aktor ma:

- Prywatny stan (nigdy bezpośrednio dotykany z zewnątrz).
- Skrzynkę odbiorczą (kolejkę wiadomości).
- Obsługiwacz: `receive(message) -> effects` gdzie efekty mogą być "odpowiedz", "wyślij do innego aktora", "utwórz nowego aktora", "zaktualizuj stan", "zatrzymaj siebie."

Dwaj aktorzy nie mogą współdzielić pamięci. Mogą tylko wysyłać wiadomości.

### Trzy warstwy API w AutoGen v0.4

1. **Core.** Niskopoziomowy framework aktora. `AgentRuntime`, `Agent`, `Message`, `Topic`. Asynchroniczna wymiana wiadomości, sterowana zdarzeniami.
2. **AgentChat.** Wysokopoziomowe API sterowane zadaniami (zamiennik dla ConversableAgent z v0.2). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** Integracje — OpenAI, Anthropic, Azure, narzędzia, pamięć.

### Dlaczego oddzielenie ma znaczenie

W modelu v0.2 wywołanie `agent_a.chat(agent_b)` synchronicznie blokuje agent_a do czasu powrotu agent_b. W v0.4, `send(agent_b, msg)` umieszcza wiadomość w skrzynce agent_b i wraca. Środowisko uruchomieniowe dostarcza później. Trzy konsekwencje:

- **Izolacja błędów.** Awaria agenta B nie powoduje awarii agenta A — środowisko łapie błąd w obsługiwaczu B i decyduje, co zrobić (loguj, ponów, martwa litera).
- **Naturalna współbieżność.** Wiele wiadomości w locie jednocześnie; aktorzy przetwarzają swoje skrzynki współbieżnie.
- **Gotowość do dystrybucji.** Skrzynka + transport to ta sama abstrakcja, niezależnie od tego, czy aktor jest w procesie, czy na innym hoście.

### Topologie

- **RoundRobinGroupChat.** Agenci zmieniają się w ustalonej rotacji.
- **SelectorGroupChat.** Agent selektor wybiera, kto idzie dalej na podstawie kontekstu rozmowy.
- **Magentic-One.** Referencyjny zespół wieloagentowy do przeglądania sieci, wykonywania kodu, obsługi plików. Zbudowany na AgentChat.

### Obserwowalność

Wsparcie OpenTelemetry jest wbudowane. Każda wiadomość emituje span; wywołania narzędzi niosą atrybuty `gen_ai.*` zgodnie z konwencjami semantycznymi OTel GenAI 2026 (Lekcja 23).

### Status: tryb utrzymania

Początek 2026: AutoGen v0.7.x jest stabilny do badań i prototypowania. Microsoft przeniósł aktywny rozwój do Microsoft Agent Framework (publiczna wersja zapoznawcza 1 października 2025; 1.0 GA planowane na koniec Q1 2026). Wzorce AutoGen przenoszą się czysto — model aktora jest trwałym pomysłem.

## Build It

`code/main.py` implementuje w stdlib środowisko uruchomieniowe aktora:

- `Message` — typowany ładunek z `sender`, `recipient`, `topic`, `body`.
- `Actor` — abstrakcyjny z `receive(message, runtime)`.
- `Runtime` — pętla zdarzeń ze współdzieloną kolejką, dostarczaniem, izolacją błędów.
- Demo dwuaktorowe: `ReviewerAgent` przegląda kod, `ChecklistAgent` uruchamia listę kontrolną; wymieniają wiadomości aż do konsensusu.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje dostarczanie wiadomości, symulowaną awarię w jednym aktorze, która nie powoduje awarii drugiego, i zbieżność do wspólnego werdyktu.

## Use It

- **AutoGen v0.4/v0.7** (utrzymanie) — stabilny do badań, prototypowania, wzorców wieloagentowych.
- **Microsoft Agent Framework** (publiczna wersja zapoznawcza) — ścieżka naprzód; te same idee modelu aktora w odświeżonym API.
- **Topologia roju LangGraph** (Lekcja 13) — podobny wzór przez przekazywanie przez współdzielone narzędzia.
- **Niestandardowe środowisko aktora** — gdy potrzebujesz konkretnego transportu (NATS, RabbitMQ, gRPC).

## Ship It

`outputs/skill-actor-runtime.md` generuje minimalne środowisko aktora plus szablon zespołu (RoundRobin lub Selector) dla danego zadania wieloagentowego.

## Exercises

1. Dodaj kolejkę martwych liter: gdy obsługiwacz zgłasza wyjątek, odłóż nieudaną wiadomość do inspekcji przez człowieka. Jak często DLQ jest trafiana w twojej zabawce?
2. Zaimplementuj `SelectorGroupChat`: aktor selektor wybiera, kto przetwarza następną wiadomość na podstawie stanu rozmowy.
3. Dodaj transport rozproszony: zamień kolejkę wewnątrzprocesową na serwer JSON-over-HTTP, aby aktorzy mogli działać w oddzielnych procesach.
4. Podłącz span OTel na wiadomość (lub atrapę no-op). Emituj `gen_ai.agent.name`, `gen_ai.operation.name` zgodnie z Lekcją 23.
5. Przeczytaj post architektoniczny AutoGen v0.4. Przenieś swoją zabawkę na prawdziwe API `autogen_core`. Co pominąłeś, co ma znaczenie w produkcji?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Aktor | "Agent" | Prywatny stan + skrzynka + obsługiwacz; brak współdzielonej pamięci |
| Wiadomość | "Zdarzenie" | Typowany ładunek; jedyny sposób interakcji aktorów |
| Skrzynka | "Skrzynka pocztowa" | Kolejka oczekujących wiadomości na aktora |
| Środowisko | "Host agenta" | Pętla zdarzeń, która routuje wiadomości i izoluje błędy |
| Temat | "Kanał" | Nazwana trasa publikuj-subskrybuj między aktorami |
| Izolacja błędów | "Niech się wyłoży" | Awaria jednego aktora nie powoduje awarii innych |
| RoundRobinGroupChat | "Zespół o stałej rotacji" | Agenci zmieniają się w kolejności |
| SelectorGroupChat | "Zespół routowany kontekstem" | Selektor wybiera, kto idzie dalej |
| Magentic-One | "Zespół referencyjny" | Drużyna wieloagentowa do sieci + kodu + plików |

## Further Reading

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — post o przeprojektowaniu
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — alternatywa w kształcie grafu
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spany, które AutoGen emituje domyślnie