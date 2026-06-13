# Wzorce orkiestracji: Supervisor, Swarm, Hierarchiczny

> Cztery wzorce orkiestracji powtarzają się w ramach 2026: supervisor-worker, swarm / peer-to-peer, hierarchiczny, debata. Wytyczne Anthropic: "Chodzi o zbudowanie odpowiedniego systemu dla twoich potrzeb." Zacznij prosto; dodawaj topologię tylko wtedy, gdy pojedynczy agent plus pięć wzorców przepływu pracy to za mało.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 minutes

## Learning Objectives

- Wymień cztery powtarzające się wzorce orkiestracji i kiedy każdy pasuje.
- Opisz zalecenie LangChain z 2026: nadzór oparty na wywołaniach narzędzi vs biblioteki supervisor.
- Wyjaśnij regułę Anthropic "zbuduj odpowiedni system" i jak warunkuje wybór topologii.
- Zaimplementuj wszystkie cztery w stdlib na wspólnym skryptowym LLM.

## Problem

Zespoły sięgają po "wieloagentowość" zanim jej potrzebują. Cztery wzorce powtarzają się w ramach; gdy potrafisz je nazwać, możesz wybrać właściwy — lub całkowicie pominąć topologię.

## Koncepcja

### Supervisor-worker

- Centralny LLM routera wysyła zadania do wyspecjalizowanych agentów.
- Decyduje: wróć do siebie, przekaż specjaliście, zakończ.
- Specjaliści nie rozmawiają ze sobą; całe routowanie przechodzi przez supervisor.

Ramy: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**Zalecenie LangChain z 2026:** wykonuj nadzór przez bezpośrednie wywołania narzędzi zamiast `create_supervisor`. Daje to dokładniejszą kontrolę inżynierii kontekstu — sam decydujesz, co dokładnie widzi każdy specjalista.

### Swarm / peer-to-peer

- Agenci przekazują sobie sterowanie bezpośrednio przez współdzieloną powierzchnię narzędzi.
- Brak centralnego routera.
- Niższe opóźnienie niż supervisor (mniej przeskoków).
- Trudniejsze do analizy (brak pojedynczego punktu kontroli).

Ramy: topologia roju LangGraph, przekazania OpenAI Agents SDK (gdy wszyscy agenci mogą przekazywać do wszystkich innych).

### Hierarchiczny

- Supervisorzy zarządzający pod-supervisorami zarządzającymi pracownikami.
- Zaimplementowany jako zagnieżdżone podgrafy w LangGraph; zagnieżdżone załogi w CrewAI.
- Skaluje się do dużych populacji agentów kosztem złożoności operacyjnej.

Kiedy potrzebujesz: gdy budżet kontekstu pojedynczego supervisor nie może pomieścić opisów wszystkich specjalistów.

### Debata

- Równolegli proponenci + iteracyjna wzajemna krytyka (Lekcja 25).
- To nie do końca orkiestracja — bardziej weryfikacja — ale pojawia się jako wybór topologii w ramach.

### CrewAI Crew vs Flow

CrewAI formalizuje dwa tryby wdrożenia:

- **Flow** dla deterministycznej automatyki sterowanej zdarzeniami (zalecany punkt startowy dla produkcji).
- **Crew** dla autonomicznej współpracy opartej na rolach.

Jest to ortogonalne do czterech powyższych wzorców, ale mapuje się na topologię: Flow to zazwyczaj supervisor lub hierarchiczny; Crew to zazwyczaj supervisor z routerem LLM.

### Wytyczne Anthropic

"Sukces w przestrzeni LLM nie polega na budowaniu najbardziej wyrafinowanego systemu. Chodzi o zbudowanie odpowiedniego systemu dla twoich potrzeb."

Kolejność decyzji:

1. Pojedynczy agent + wzorce przepływu pracy (Lekcja 12) — zacznij tutaj.
2. Supervisor-worker — gdy masz 2-4 specjalistów.
3. Swarm — gdy opóźnienie ma większe znaczenie niż klarowność rozumowania.
4. Hierarchiczny — tylko gdy budżet kontekstu supervisor zawodzi.
5. Debata — gdy dokładność ma większe znaczenie niż koszt.

### Gdzie ten wzór zawodzi

- **Myślenie topologią najpierw.** "Potrzebujemy wieloagentowości" przed zidentyfikowaniem, jaki problem wieloagentowość rozwiązuje.
- **Odbijające się przekazania w roju.** A -> B -> A -> B. Używaj liczników przeskoków.
- **Fałszywa hierarchia.** Trzy warstwy, bo "przedsiębiorstwo"; dwa rzeczywiste zespoły. Zwiń.

## Build It

`code/main.py` implementuje wszystkie cztery wzorce w stdlib na skryptowym LLM:

- `Supervisor` — centralny router.
- `Swarm` — peer-to-peer z bezpośrednimi przekazaniami.
- `Hierarchical` — supervisorzy supervisorów.
- `Debate` — równolegli proponenci + krytyka.

Każdy wzorzec obsługuje to samo zadanie trzech intencji (zwrot / błąd / sprzedaż). Kształty śladów różnią się.

Uruchom:

```
python3 code/main.py
```

Wynik: ślad na wzorzec + liczba operacji. Supervisor jest najczystszy; swarm najkrótszy; hierarchiczny najgłębszy; debata najdroższa.

## Use It

- **LangGraph** dla supervisor i hierarchicznego (zagnieżdżone podgrafy).
- **OpenAI Agents SDK** dla przekazań jako narzędzi (kształt supervisor).
- **CrewAI Flow** dla produkcyjnej deterministyki.
- **Własne** dla debaty lub gdy chcesz dokładnej kontroli.

## Ship It

`outputs/skill-orchestration-picker.md` wybiera topologię i implementuje ją.

## Exercises

1. Przekształć supervisor-worker w swarm, usuwając router. Co się psuje? Co się poprawia?
2. Dodaj licznik przeskoków do swarma: odmów po 3 przekazaniach. Czy łapie odbijanie A->B->A?
3. Zbuduj dwupoziomowy system hierarchiczny dla domeny z 12 specjalistami. Gdzie budżet kontekstu zawodzi bez zagnieżdżania?
4. Profiluj cztery wzorce na obciążeniu produkcyjnym. Który wygrywa na której metryce (opóźnienie, koszt, dokładność, możliwość debugowania)?
5. Przeczytaj post Anthropic "Building Effective Agents". Mapuj każdy ze swoich produkcyjnych przepływów na jeden z czterech. Są takie, które nie mapują się czysto?

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specjaliści" | Centralny LLM wysyła do specjalistów; nie rozmawiają ze sobą |
| Swarm | "Peer-to-peer" | Bezpośrednie przekazania przez współdzielone narzędzia; brak centralnego routera |
| Hierarchical | "Supervisorzy supervisorów" | Zagnieżdżone podgrafy dla dużych populacji |
| Debate | "Proponent + krytyka" | Równolegli proponenci, wzajemna krytyka (Lekcja 25) |
| Tool-call-based supervision | "Supervisor bez biblioteki" | Zaimplementuj supervisor jako bezpośrednie wywołania narzędzi dla kontroli kontekstu |
| Crew | "Autonomiczny zespół" | Tryb współpracy opartej na rolach CrewAI |
| Flow | "Deterministyczny przepływ pracy" | Tryb produkcyjny CrewAI sterowany zdarzeniami |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — pięć wzorców + agent vs przepływ pracy
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchiczny
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — wzór debaty