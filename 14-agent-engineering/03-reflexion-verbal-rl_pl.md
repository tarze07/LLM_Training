# Reflexion: Werbalne uczenie przez wzmacnianie

> Gradientowe RL potrzebuje tysięcy prób i klastra GPU, aby naprawić tryb awarii. Reflexion (Shinn et al., NeurIPS 2023) robi to w języku naturalnym: po każdej nieudanej próbie agent pisze refleksję, przechowuje ją w pamięci epizodycznej i warunkuje następną próbę na tej pamięci. To jest wzorzec stojący za Letta sleep-time compute, CLAUDE.md w Claude Code i learn-rule w pro-workflow.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## Learning Objectives

- Wymień trzy komponenty Reflexion (Aktor, Ewaluator, Samo-reflektor) i rolę pamięci epizodycznej.
- Zaimplementuj pętlę Reflexion w stdlib z binarnym ewaluatorem, buforem refleksji i świeżymi ponownymi próbami.
- Wybierz między skalarnymi, heurystycznymi i samooceniającymi źródłami informacji zwrotnej dla danego zadania.
- Wyjaśnij, dlaczego werbalne wzmacnianie łapie błędy, które gradientowe RL wymagałoby tysięcy prób do naprawienia.

## The Problem

Agent nie wykonuje zadania. W standardowym RL uruchomiłbyś tysiące dodatkowych prób, obliczył gradienty, zaktualizował wagi. Drogie, wolne, a większość produkcyjnych agentów nie ma budżetu treningowego na każdą porażkę.

Reflexion (Shinn et al., arXiv:2303.11366) zadaje inne pytanie: co jeśli agent po prostu pomyślałby, dlaczego zawiódł, i spróbował ponownie z tą myślą w swoim prompcie? Żadnych aktualizacji wag. Żadnego gradientu. Po prostu język naturalny przechowywany między próbami.

Wynik: na ALFWorld bije ReAct i inne niedostrojone linie bazowe. Na HotpotQA poprawia względem ReAct. Na generowaniu kodu (HumanEval/MBPP) ustanawia ówczesny stan sztuki. Wszystko bez ani jednego kroku gradientowego.

## The Concept

### Trzy komponenty

```
Actor         : generuje trajektorię (pętla w stylu ReAct)
Evaluator     : ocenia trajektorię — binarnie, heurystycznie lub samoocena
Self-Reflector: pisze refleksję w języku naturalnym o niepowodzeniu
```

Plus jedna struktura danych:

```
Episodic memory: lista wcześniejszych refleksji, dołączana na początku promptu następnej próby
```

Jedna próba uruchamia Aktora. Ewaluator ją ocenia. Jeśli wynik jest niski, Samo-reflektor tworzy refleksję ("Wybrałem złe narzędzie, bo źle przeczytałem pytanie jako pytanie o X, gdy chodziło o Y"). Refleksja trafia do pamięci epizodycznej. Następna próba zaczyna od nowa, ale widzi refleksję.

### Trzy typy ewaluatora

1. **Skalarny** — zewnętrzny sygnał binarny. ALFWorld udaje się lub nie. Testy HumanEval przechodzą lub nie. Najprostszy, najsilniejszy sygnał.
2. **Heurystyczny** — predefiniowane sygnatury awarii. "Jeśli agent wyprodukował tę samą akcję dwa razy z rzędu, oznacz jako zablokowanego." "Jeśli trajektoria przekracza 50 kroków, oznacz jako nieefektywną."
3. **Samooceniający** — LLM ocenia własną trajektorię. Potrzebne, gdy nie ma dostępnej prawdy podstawowej. Słabszy sygnał; dobrze łączy się z weryfikacją opartą na narzędziach (Lekcja 05 — CRITIC).

Domyślne ustawienie 2026 to mieszanka: skalarny gdy dostępny, samoocena gdy nie, heurystyki jako bariery bezpieczeństwa.

### Dlaczego to się uogólnia

Reflexion to nie tyle nowy algorytm, co nazwany wzorzec. Prawie każdy produkcyjny agent "samonaprawiający" uruchamia jakąś odmianę:

- Letta sleep-time compute (Lekcja 08): oddzielny agent analizuje przeszłe rozmowy i zapisuje do bloków pamięci.
- `CLAUDE.md` / "save memory" w Claude Code: refleksje przechwycone jako wnioski, dołączane do przyszłych sesji.
- Polecenie `/learn-rule` w pro-workflow: poprawki przechwycone jako jawne reguły.
- Węzły refleksji w LangGraph: węzeł, który ocenia wyjście i kieruje do poprawy, jeśli potrzeba.

Wszystkie wywodzą się z tego samego spostrzeżenia: język naturalny jest wystarczająco bogatym medium, by przenosić "czego nauczyłem się z porażki" między uruchomieniami.

### Kiedy działa, a kiedy nie

Reflexion działa, gdy:

- Jest jasny sygnał porażki (nieudany test, błąd narzędzia, błędna odpowiedź).
- Klasa zadania jest powtarzalna (ten sam typ pytania może być zadany ponownie).
- Refleksja ma przestrzeń, by poprawić trajektorię (wystarczający budżet akcji).

Reflexion nie pomaga, gdy:

- Agent już odnosi sukces za pierwszym razem.
- Porażka jest zewnętrzna (sieć nie działa, narzędzie zepsute) — refleksja "sieć nie działała" nie pomaga w przyszłych uruchomieniach.
- Refleksja zamienia się w przesąd — przechowywanie narracji o jednorazowym, niestabilnym uruchomieniu.

Pułapka 2026: gnicie pamięci. Refleksje się kumulują; niektóre są nieaktualne lub błędne; ponowne uruchomienia zwalniają w miarę wzrostu bufora epizodycznego. Łagodzenie: okresowa kompakcja (Lekcja 06), TTL na refleksjach lub oddzielny agent czyszczący w czasie bezczynności (Letta).

```figure
react-trace
```

## Build It

`code/main.py` implementuje Reflexion na zabawkowej łamigłówce: wyprodukuj 3-elementową listę, której suma jest równa celowi. Aktor emituje listy kandydatów; Ewaluator sprawdza sumę; Samo-reflektor pisze linię o tym, co poszło nie tak. Refleksja trafia do pamięci epizodycznej na następną próbę.

Komponenty:

- `Actor` — skryptowana polityka, która poprawia się, gdy widzi refleksje.
- `Evaluator.binary()` — zaliczenie/niezaliczenie docelowej sumy.
- `SelfReflector` — generuje jednoliniową diagnozę porażki.
- `EpisodicMemory` — ograniczona lista z semantyką TTL.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje trzy próby. Próba 1 kończy się niepowodzeniem, refleksja jest przechowywana, próba 2 widzi refleksję i poprawia się, ale wciąż zawodzi, próba 3 odnosi sukces. Porównaj z bazowym uruchomieniem (bez refleksji) — pozostaje ono zablokowane na odpowiedzi z próby 1.

## Use It

LangGraph dostarcza refleksję jako wzorzec węzła. Polecenie `/memory` w Claude Code i `/learn-rule` w pro-workflow eksternalizują bufor epizodyczny jako plik markdown. Letta sleep-time compute uruchamia Samo-reflektor w czasie bezczynności, aby główny agent pozostał związany opóźnieniem. OpenAI Agents SDK nie dostarcza Reflexion bezpośrednio; budujesz je z niestandardowym Guardrailem, który odrzuca trajektorie według wyniku, i sesją `Session`, która utrzymuje się między uruchomieniami.

## Ship It

`outputs/skill-reflexion-buffer.md` tworzy i utrzymuje bufor epizodyczny z przechwytywaniem refleksji, TTL i deduplikacją. Mając klasę zadania i porażkę, emituje refleksję, która faktycznie pomaga następnej próbie (a nie ogólne "bądź bardziej ostrożny").

## Exercises

1. Przełącz z binarnego na skalarny ewaluator, który zwraca metrykę odległości (jak daleko od celu). Czy zbiega się szybciej?
2. Dodaj TTL wynoszący 10 prób do refleksji. Czy starsze refleksje szkodzą czy pomagają po tym punkcie?
3. Zaimplementuj heurystyczny ewaluator: oznacz próbę jako zablokowaną, jeśli ta sama akcja się powtarza. Jak to oddziałuje z Samo-reflektorem?
4. Uruchom Reflexion z adwersarialnym Aktorem, który ignoruje refleksje. Jaka jest minimalna inżynieria promptu refleksji, która zmusza Aktora do ich zauważenia?
5. Przeczytaj Sekcję 4 pracy Reflexion o AlfWorld. Odtwórz koncepcyjnie 130% poprawę wskaźnika sukcesu: jaka jest kluczowa różnica względem waniliowego ReAct?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|------|----------------|------------------------|
| Reflexion | "Samokorekta" | Shinn et al. 2023 — Aktor, Ewaluator, Samo-reflektor plus pamięć epizodyczna |
| Werbalne wzmacnianie | "Uczenie bez gradientów" | Refleksja w języku naturalnym dołączana na początku promptu następnej próby |
| Pamięć epizodyczna | "Refleksje per-zadanie" | Ograniczony bufor wcześniejszych refleksji dla jednej klasy zadania |
| Ewaluator skalarny | "Binarny sygnał sukcesu" | Zaliczenie/niezaliczenie lub numeryczny wynik z prawdy podstawowej |
| Ewaluator heurystyczny | "Detektor oparty na wzorcach" | Predefiniowane sygnatury awarii (np. zablokowana pętla, zbyt wiele kroków) |
| Samoewaluator | "LLM jako sędzia własnego śladu" | Słabszy sygnał awaryjny, gdy brak prawdy podstawowej — łącz z weryfikacją opartą na narzędziach |
| Gnicie pamięci | "Nieaktualne refleksje" | Bufor epizodyczny wypełnia się nieaktualnymi wpisami; napraw przez kompakcję/TTL |
| Refleksja w czasie bezczynności | "Asynchroniczna autorefleksja" | Uruchom Samo-reflektor poza ścieżką krytyczną, aby główny agent pozostał szybki |

## Further Reading

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — kanoniczna praca
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — asynchroniczna refleksja w produkcji
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — zarządzanie buforem epizodycznym jako częścią kontekstu
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — wzorzec węzła refleksji

(End of file - total 135 lines)