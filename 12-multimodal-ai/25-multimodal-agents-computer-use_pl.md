# Agenci Multimodalni i Użytkowanie Komputera (Capstone)

> Granicznym produktem 2026 roku jest multimodalny agent, który czyta zrzuty ekranu, klika przyciski, nawiguje po interfejsach WWW, wypełnia formularze i realizuje przepływy pracy end-to-end. SeeClick i CogAgent (2024) udowodniły prymityw ugruntowania GUI. Ferret-UI dodał urządzenia mobilne. ChartAgent wprowadził wizualne użycie narzędzi dla wykresów. VisualWebArena i AgentVista (2026) to benchmarki, za którymi goni granica — a nawet Gemini 3 Pro i Claude Opus 4.7 osiągają ~30% w trudnych zadaniach AgentVista. Ten capstone łączy wszystkie wątki Fazy 12: percepcję (wysokorozdzielczy VLM), rozumowanie (LLM z użyciem narzędzi), ugruntowanie (wyjście współrzędnych), pamięć długiego horyzontu i ewaluację.

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 minutes

## Learning Objectives

- Zaprojektować pętlę agenta multimodalnego: postrzegaj → rozumuj → działaj → obserwuj → powtórz.
- Zbudować schemat wyjścia ugruntowania GUI (współrzędne kliknięcia, wpisz tekst, przewiń, przeciągnij), który VLM może emitować jako JSON.
- Porównać agentów opartych tylko na zrzutach ekranu, agentów opartych na drzewie dostępności i agentów hybrydowych.
- Skonfigurować ewaluację benchmarku agenta multimodalnego na małym wycinku VisualWebArena.

## The Problem

Przepływ pracy na stronie rezerwacji: "znajdź mi lot do Tokio na 15 kwietnia, miejsce przy przejściu poniżej 800 USD, zarezerwuj."

Agent multimodalny musi:

1. Zrobić zrzut ekranu przeglądarki.
2. Sparsować zrzut ekranu + URL + cel w plan.
3. Wyemitować strukturalną akcję: kliknij (w x, y), wpisz "Tokio" (w elemencie E), przewiń w dół, wybierz (przycisk radiowy).
4. Zastosować akcję w przeglądarce.
5. Obserwować nowy stan (następny zrzut ekranu).
6. Powtarzać, aż zadanie zostanie wykonane.

Każdy krok to multimodalne wywołanie VLM. Wyjście VLM musi być parsowalnym JSON-em. Błędy kumulują się między krokami, więc odzyskiwanie ma znaczenie.

## The Concept

### Ugruntowanie GUI — prymityw

Ugruntowanie GUI to: mając zrzut ekranu i instrukcję w języku naturalnym, wyprowadź współrzędne (x, y) do kliknięcia (lub inną akcję).

SeeClick (arXiv:2401.10935) był pierwszym otwartym wynikiem na skalę: dostrój VLM na syntetycznych + rzeczywistych danych GUI, wyprowadź współrzędne jako zwykłe tokeny tekstowe. Działa.

CogAgent (arXiv:2312.08914) dodał kodowanie wysokiej rozdzielczości 1120x1120 dla gęstych interfejsów. Wynik: ~84% w nawigacji internetowej.

Ferret-UI (arXiv:2404.05719) skupia się na mobilnych interfejsach, integruje się z danymi dostępności iOS.

Format wyjścia to zwykle JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` pomaga w odzyskiwaniu: jeśli współrzędne dryfują między zrzutami ekranu, semantyczna podpowiedź pozwala systemowi ponownie się ugruntować.

### Schematy akcji

Typowy schemat akcji ma 6–10 typów akcji:

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent emituje jedną akcję na krok. Opakowanie przeglądarki wykonuje i zwraca nowy stan.

### Tylko zrzut ekranu vs drzewo dostępności

Dwa tryby wejścia:

- Tylko zrzut ekranu: pełny obraz, bez informacji strukturalnych. Najbardziej ogólny; działa na każdej aplikacji.
- Drzewo dostępności: strukturalny DOM / informacje dostępności iOS. Znacznie bardziej niezawodne dla ugruntowania; działa tam, gdzie drzewo jest dostępne.
- Hybrydowy: oba, z drzewem jako niezawodnym gruntem dla atomowych akcji i zrzutem ekranu dla kontekstu semantycznego.

Agenci produkcyjni używają hybrydy, gdy to możliwe. Automatyzacja przeglądarki (Selenium + dostępność) zawsze ma drzewo; aplikacje desktopowe czasami.

### Pamięć długiego horyzontu

20-krokowy przepływ pracy generuje 20 zrzutów ekranu. Kontekst VLM szybko się wypełnia. Trzy strategie kompresji:

- Łańcuch podsumowań: co 5 kroków podsumuj, co się wydarzyło, odrzuć stare zrzuty ekranu.
- Pomijanie klatek: zachowaj pierwszy, ostatni i co 3. zrzut ekranu.
- Log rejestrowany przez narzędzie: wykonuj akcje, przechowuj tekstowy log tego, co zostało zrobione; nie przeglądaj ponownie starych zrzutów ekranu.

API używania komputera Claude'a używa wzorca logu. Prostsze, bardziej niezawodne.

### Wizualne użycie narzędzi

ChartAgent (arXiv:2510.04514) wprowadza wizualne użycie narzędzi dla rozumienia wykresów: przycięcie, powiększenie, OCR, wywołanie zewnętrznej detekcji. Agent może wyprowadzić "przytnij do obszaru (100, 200, 300, 400), a następnie wywołaj OCR" jako wywołanie narzędzia. Narzędzie zwraca tekst; VLM kontynuuje rozumowanie.

Ten wzorzec się uogólnia: promptowanie set-of-mark, adnotacje regionów i zewnętrzne narzędzia detekcji wszystkie mieszczą się w tym samym schemacie "wyślij wywołanie narzędzia, otrzymaj strukturalną odpowiedź."

### Benchmarki 2026

- ScreenSpot-Pro. Ugruntowanie GUI na ~1k zrzutów ekranu stron WWW. Otwarty SOTA Qwen2.5-VL-72B ~85%. Granica ~90%.
- VisualWebArena. Zadania internetowe end-to-end (sklep, forum, ogłoszenia). Otwarty SOTA ~20%. Gemini 3 Pro ~27%.
- AgentVista (arXiv:2602.23166). Najtrudniejszy benchmark 2026. Realistyczne przepływy pracy w 12 domenach. Modele graniczne osiągają 27–40%; otwarte modele 10–20%.
- WebArena / WebShop. Starsze benchmarki; nasycone przez modele graniczne.

### Dlaczego wciąż jest trudne

Wąskie gardła wydajności agentów:

1. Ugruntowanie wizualne w drobnej skali. "Kliknij mały X" często zawodzi w rozdzielczości mobilnej.
2. Planowanie długiego horyzontu. Po 10 akcjach agent oddala się od celu.
3. Odzyskiwanie po błędach. Gdy kliknięcie zawodzi (zły przycisk), wykrycie + odzyskanie rzadko występuje w danych treningowych.
4. Kontekst między stronami. Skakanie między kartami lub długie formularze tracą stan.

Kierunki badań: architektury pamięci, jawne przeplanowanie, multimodalna weryfikacja (dopasowanie zrzutu ekranu do powodzenia akcji).

### Zadanie capstone

Zadanie capstone: zbuduj agenta używania komputera, który:

1. Czyta HTML + zrzut ekranu pozorowanej strony rezerwacji.
2. Planuje wieloetapową sekwencję: szukaj → wybierz → wypełnij formularz → wyślij.
3. Emituje akcje JSON zgodne ze schematem akcji.
4. Ewaluuje na ustalonym wycinku 10 zadań.

Lekcja dostarcza kod szkieletowy, który jest łatwy do rozszerzenia w prawdziwą przeglądarkę.

## Use It

`code/main.py` to szkielet capstone:

- Definicja schematu akcji JSON (10 akcji).
- Pozorowany stan przeglądarki jako słownik.
- Szkielet pętli agenta: otrzymaj stan, wyemituj akcję, zastosuj, powtórz.
- Minibenchmark 10 zadań (syntetyczne strony) do pomiaru wskaźnika sukcesu end-to-end.
- Hak odzyskiwania po błędach dla przypadku, gdy akcja zawodzi.

## Ship It

Ta lekcja produkuje `outputs/skill-multimodal-agent-designer.md`. Dla produktu używania komputera (domena, zestaw akcji, cel ewaluacji) projektuje pełną pętlę agenta, strategię pamięci, tryb ugruntowania i oczekiwany wynik benchmarku.

## Exercises

1. Rozszerz schemat akcji o narzędzie `screenshot_region` (przycięcie + powiększenie). Jakie zadania zyskują?

2. Przeczytaj AgentVista (arXiv:2602.23166). Opisz najtrudniejszą kategorię zadań i dlaczego modele graniczne wciąż zawodzą.

3. Kompresja pamięci długiego horyzontu: zaprojektuj łańcuch podsumowań z ≤4 przechowywanymi aktywnymi zrzutami ekranu, dowolną liczbą zalogowanych.

4. Zbuduj hak odzyskiwania po błędach: po niepowodzeniu akcji (przycisk nieznaleziony), co robi agent dalej?

5. Porównaj Claude 4.7 tylko na zrzutach ekranu z hybrydowym zrzut ekranu + drzewo dostępności Qwen2.5-VL na 10 zadaniach internetowych. Który wygrywa w których zadaniach?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GUI grounding | "Click coordinates" | Model wyprowadza (x,y) dla celu instrukcji na zrzucie ekranu |
| Action schema | "Tool definitions" | Opis JSON poprawnych akcji (click, type, scroll, drag) |
| Accessibility tree | "Structured DOM" | Hierarchia interfejsu czytelna maszynowo z API przeglądarki/iOS |
| Hybrid agent | "Screenshot + tree" | Używa zarówno obrazu, jak i informacji strukturalnych; bardziej niezawodny niż każdy z osobna |
| Visual tool use | "Zoom/crop/detect" | Agent wywołuje zewnętrzne narzędzia wizyjne (OCR, detekcja) w trakcie planu |
| Summary-chain | "Memory compression" | Okresowe tekstowe podsumowania zastępują długą historię zrzutów ekranu |
| VisualWebArena | "E2E web bench" | Benchmark z 2024 roku dla internetowych zadań end-to-end |
| AgentVista | "2026 hard bench" | Realistyczne przepływy pracy w 12 domenach; nawet Gemini 3 Pro osiąga ~30% |

## Further Reading

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)