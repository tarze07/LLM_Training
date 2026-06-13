# Wnioskowanie asynchroniczne i Hogwild!

> Dekodowanie spekulatywne (Faza 10 · 15) parallelizuje tokeny w obrębie jednej sekwencji. Frameworki wieloagentowe parallelizują całe sekwencje, ale wymuszają jawną koordynację (głosowanie, dzielenie podzadań). Hogwild! Inference (Rodionov i in., arXiv:2504.06261) robi coś innego: uruchamia N instancji tego samego LLM równolegle na WSPÓŁDZIELONEJ pamięci podręcznej klucz-wartość. Każdy pracownik natychmiast widzi tokeny wygenerowane przez każdego innego pracownika. Nowoczesne modele rozumujące — QwQ, DeepSeek-R1 — mogą samodzielnie koordynować się poprzez tę współdzieloną pamięć podręczną bez żadnego dostrajania. Podejście jest eksperymentalne, ale otwiera zupełnie nową oś równoległości wnioskowania, która jest ortogonalna do dekodowania spekulatywnego. Ta lekcja implementuje symulator Hogwild! z dwoma pracownikami w stdlib Python i wyjaśnia, dlaczego współpraca poprzez współdzieloną pamięć podręczną wyłania się z istniejących zdolności rozumowania modelu.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (optymalizacja wnioskowania), Phase 10 · 15 (dekodowanie spekulatywne)
**Time:** ~60 minut

## Cele nauczania

- Opisać trzy powszechne topologie równoległych LLM (głosowanie, podzadanie, Hogwild!) i wskazać, które problemy każda z nich rozwiązuje.
- Przedstawić podstawową konfigurację Hogwild!: wielu pracowników, jedna współdzielona pamięć podręczna KV, wyłaniająca się koordynacja poprzez samopodpowiadanie.
- Obliczyć przyspieszenie ścienne Hogwild! jako funkcję liczby pracowników `N`, równoległości na poziomie zadania `p` i narzutu koordynacji `c`.
- Zaimplementować symulator Hogwild! z dwoma pracownikami na zabawkowym problemie i zaobserwować wyłaniający się podział zadań.

## Problem

Nowoczesne LLM rozwiązują trudne problemy, produkując długie łańcuchy rozumowania — 5000 tokenów krok po kroku to norma, dziesiątki tysięcy tokenów zdarzają się przy głębokich problemach matematycznych. Przy 35 tokenach/s dekodowania na modelu 70B, 50k tokenów to 24 minuty. Model nie jest interaktywny.

Dekodowanie spekulatywne (Faza 10 · 15) daje 3-5-krotne przyspieszenie poprzez parallelizację w obrębie jednej sekwencji. Poza tym sekwencyjna zależność dekodowania autoregresyjnego jest twardym sufitem. Każdy nowy token zależy od każdego poprzedniego tokena.

Oczywiste pytanie: czy możemy parallelizować w poprzek sekwencji? Uruchomić wiele kopii tego samego modelu na tym samym problemie, pozwolić im współpracować, podzielić pracę?

Wcześniejsze prace: zespoły głosujące (uruchom N modeli, wybierz odpowiedź większościową), drzewo myśli (rozgałęziaj ścieżki rozumowania i łącz), frameworki wieloagentowe (przypisz każdemu agentowi podzadanie, użyj koordynatora). Wszystkie one pomagają w określonych domenach zadań. Wszystkie wprowadzają również jawną maszynerię koordynacji — reguły głosowania, logikę rozgałęziaj-i-przycinaj, protokoły komunikacji agent-agent.

Hogwild! Inference przyjmuje inne podejście. N pracowników współdzieli pojedynczą pamięć podręczną KV. Każdy pracownik natychmiast widzi tokeny wygenerowane przez każdego innego pracownika, tak jakby były jego własnym kontekstem. Pracownicy — bez żadnego trenowania ani dostrajania — sami ustalają, jak podzielić pracę. Nowoczesne modele rozumujące (QwQ, DeepSeek-R1, tryb rozumowania rodziny Claude) potrafią czytać współdzieloną pamięć podręczną i mówić rzeczy w stylu „widzę, że pracownik 2 już obsłużył przypadek podstawowy, więc zajmę się krokiem indukcyjnym."

Przyspieszenie zależy od obciążenia i jest eksperymentalne według stanu na kwiecień 2026. Ale pomysł jest wart poznania, ponieważ otwiera nową oś równoległości wnioskowania.

## Koncepcja

### Konfiguracja

Zainicjuj N procesów pracowniczych, wszystkie uruchamiające ten sam LLM. Zamiast pamięci podręcznych KV na pracownika, utrzymuj JEDNĄ współdzieloną pamięć podręczną. Gdy pracownik `i` generuje token `t_j`, token jest zapisywany do współdzielonej pamięci podręcznej na następnej pozycji. Gdy pracownik `k` wykonuje swój następny krok, odczytuje bieżący stan pamięci podręcznej (który zawiera wszystko, co wszyscy N pracownicy wygenerowali do tej pory).

W każdym kroku pracownicy ścigają się, aby zapisać tokeny. Nie ma indeksu pozycji na pracownika — pamięć podręczna to pojedyncza rosnąca sekwencja. Kolejność jest określana przez czas przybycia zapisu.

### Dlaczego koordynacja się wyłania

Pracownicy współdzielą podpowiedź (prompt). Zazwyczaj coś w stylu „Jesteś jedną z N instancji pracujących razem nad tym problemem. Każda instancja czyta współdzieloną pamięć i może zobaczyć, co napisały inne instancje. Unikaj zbędnej pracy." Podpowiedź plus współdzielona pamięć podręczna wystarczają. Modele rozumujące czytają pamięć podręczną, zauważają, które części problemu zostały już podjęte, i (często, ale nie zawsze) przechodzą do niezbadanych części.

Artykuł Hogwild! (Rodionov i in., 2025) raportuje obserwacje takie jak:

- Pracownicy formułują plany i komunikują je innym pracownikom poprzez pamięć podręczną.
- Pracownicy zauważają błędy w rozumowaniu innych pracowników i je wytykają.
- Pracownicy dostosowują się, gdy plan zawodzi, i proponują alternatywy.
- Gdy poproszeni o sprawdzenie nadmiarowości, pracownicy ją wykrywają i zmieniają kierunek.

Żadne z tego nie wymaga dostrajania. Wyłaniające się zachowanie pochodzi ze zdolności rozumowania, które model już posiada.

### Nazewnictwo

Nazwa artykułu nawiązuje do Hogwild! SGD (Recht i in., 2011), optymalizatora z aktualizacjami asynchronicznymi. Analogia: asynchroniczni pracownicy SGD zapisują do współdzielonego wektora parametrów; pracownicy Hogwild! Inference zapisują do współdzielonej pamięci podręcznej KV. Oba polegają na empirycznej zbieżności, a nie na gwarancjach synchronizacji.

### RoPE czyni to wykonalnym

Rotary Position Embeddings (RoPE, Su i in. 2021) kodują informację o pozycji poprzez obrót w wektorach Q i K. Ponieważ pozycje są obrotami, a nie wbudowanymi przesunięciami, pozycja tokena może się zmienić bez przeliczania wpisu w pamięci podręcznej KV. Gdy pracownik `i` zapisuje do współdzielonej pamięci podręcznej na pozycji `p`, inni pracownicy czytający tę pozycję mogą użyć zapisanego wpisu bezpośrednio — nie jest potrzebny ponowny obrót.

W modelu z uczoną pozycją lub absolutną pozycją, Hogwild! wymagałby unieważnienia pamięci podręcznej przy każdym równoczesnym zapisie. RoPE pozwala pamięci podręcznej pozostać stabilną.

### Matematyka czasu ściennego

Niech `T_serial` będzie czasem dla jednego pracownika na samodzielne rozwiązanie problemu. Niech `p` będzie ułamkiem zadania możliwym do zrównoleglenia. Niech `c` będzie narzutem koordynacji na krok (czytanie rozszerzonej pamięci podręcznej, decydowanie, co napisać).

Czas jednego pracownika: `T_serial`.
Czas N pracowników Hogwild!, jeśli koordynacja jest darmowa: `T_serial * ((1 - p) + p / N)`. Klasyczne Amdahl.
Z narzutem koordynacji: `T_serial * ((1 - p) + p / N) + c * steps_per_worker`.

Aby pracownik był produktywny, `c` musi być małe w stosunku do czasu dekodowania na krok. Na modelach rozumujących produkujących 5k+ tokenów, pracownicy mogą sobie pozwolić na setki tokenów narzutu koordynacji i wciąż wychodzić na plus. Przy krótkich zadaniach czatowych koordynacja dominuje i Hogwild! jest gorszy niż sekwencyjny.

### Konkretny przykład

Problem rozumowania: 10k tokenów łańcucha myśli. Załóżmy, że problem ma `p = 0.7` treści możliwej do zrównoleglenia (różne strategie dowodowe, różne analizy przypadków) i `c = 200` tokenów narzutu koordynacji na pracownika. Przy `N = 4` pracownikach:

- Czas sekwencyjny: 10000 kroków dekodowania.
- Czas Hogwild!: 10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 kroków dekodowania.
- Przyspieszenie: 10000 / 5550 = 1.8x.

To skromne. Ale przy dłuższych problemach rozumowania (50k tokenów) narzut koordynacji się amortyzuje, a przyspieszenie sięga 2.5-3x. Hogwild! jest odpowiednikiem równoległości na poziomie wątków w języku, który pozwala naturalnie pisać kod wielowątkowy.

### Kiedy sięgać po Hogwild!

- Długie problemy rozumowania (tysiące tokenów), gdzie zadanie można zrównoleglić na niezależne podcele.
- Modele rozumujące, które zostały wytrenowane do myślenia krok po kroku. Modele nierozumujące nie koordynują się dobrze.
- Wdrożenia na pojedynczym węźle z wystarczającą ilością VRAM, aby pomieścić współdzieloną pamięć podręczną plus N procesów pracowniczych. Pamięć podręczna jest współdzielona, ale każdy pracownik ma własną pamięć aktywacji.

### Kiedy nie

- Krótki interaktywny czat. Narzut koordynacji dominuje.
- Zadania, które się nie parallelizują (pojedynczy dowód liniowy, pojedyncza kompilacja). N=1 to maksimum.
- Modele nierozumujące. Żadna koordynacja się nie wyłania.
- Wdrożenia na wielu węzłach. Współdzielona pamięć podręczna wymaga bardzo szybkiej synchronizacji między pracownikami. Wewnątrz węzła jest w porządku; między węzłami to katastrofa opóźnień.

### Status eksperymentalny

Według stanu na kwiecień 2026, Hogwild! to metoda badawcza z otwartą implementacją PyTorch. Produkcyjna adopcja nie nastąpiła. Trzy blokady:

1. Zarządzanie współdzieloną pamięcią podręczną KV w równoczesnych procesach to nietrywialna inżynieria.
2. Wyłaniająca się koordynacja zależy od zadania; benchmarki są wciąż budowane.
3. Przyspieszenia są skromne w porównaniu do tego, co już dostarcza dekodowanie spekulatywne, a obie techniki można łączyć, ale połączona inżynieria to kolejna warstwa.

Warte poznania. Warte eksperymentowania. Jeszcze nie warte stawiania produktu.

```figure
continuous-batching
```

## Zbuduj

`code/main.py` implementuje zabawkowy symulator Hogwild!:

- Dwa procesy pracownicze, każdy deterministyczny „LLM" produkujący jedną z kilku kategorii tokenów (token-pracy, token-obserwacji, token-koordynacji) ze znanymi prawdopodobieństwami.
- Współdzielona pamięć podręczna (po prostu lista tokenów), którą obaj pracownicy czytają i do której zapisują.
- Prosta logika koordynacji: gdy pracownik widzi, że drugi wyprodukował już wystarczająco dużo tokenów pracy w danej kategorii, wybiera inną kategorię.

Symulator działa przez ustalony budżet kroków i raportuje:

- Łączną liczbę wyprodukowanych tokenów pracy.
- Łączny czas ścienny (liczba kroków pracownika).
- Efektywne przyspieszenie w stosunku do pojedynczego pracownika.
- Ślad tego, który pracownik napisał który token.

### Krok 1: współdzielona pamięć podręczna

Lista, do której obaj pracownicy dołączają. Proste blokowanie (`threading.Lock`) w rzeczywistej implementacji; symulujemy za pomocą licznika.

### Krok 2: pętla pracownika

Każdy pracownik w każdym kroku:

- Czyta bieżącą współdzieloną pamięć podręczną.
- Decyduje, jaką kategorię tokena napisać na podstawie tego, co już tam jest.
- Zapisuje jeden token.

### Krok 3: heurystyka koordynacji

Jeśli kategoria X ma już K tokenów w pamięci podręcznej, a zamierzona kategoria pracownika to X, pracownik przełącza się na kategorię Y. To zabawkowy substytut zachowania modelu rozumującego „zauważ, że to już jest obsłużone, zrób coś innego."

### Krok 4: zmierzone przyspieszenie

Uruchom symulator z N=1 pracownikiem i z N=2 pracownikami, ten sam całkowity budżet kroków. Policz wyprodukowane tokeny pracy. N=2 powinno wyprodukować około 1.5-1.8x więcej tokenów pracy z powodu podziału zadań napędzanego koordynacją.

### Krok 5: obciąż koordynację

Zmniejsz czułość heurystyki koordynacji. Uruchom ponownie. Zaobserwuj, że bez dobrej koordynacji N=2 redundantly produkuje te same tokeny, a przyspieszenie spada poniżej 1. To odpowiada obserwacji z artykułu: sztuczka działa tylko wtedy, gdy pracownicy mają zdolność rozumowania do samokoordynacji.

## Użyj

Integracja Hogwild! w produkcji według stanu na kwiecień 2026 jest na poziomie badawczym. Implementacja referencyjna z Yandex/HSE/IST jest oparta na PyTorch i celuje w konfiguracje wieloprocesowe na jednym węźle na modelach DeepSeek-R1 i QwQ.

Pragmatyczna ścieżka adopcji:

1. Profiluj swoje obciążenie zadaniami rozumowania. Zmierz ułamek tokenów, które są eksploracyjne (wiele strategii, analizy przypadków, przeszukiwanie) vs liniowe.
2. Jeśli eksploracja dominuje, uruchom eksperyment Hogwild! z dwoma pracownikami. Zmierz poprawę czasu ściennego.
3. Jeśli poprawa jest poniżej 1.3x, jesteś w reżimie zdominowanym przez koordynację. Wróć do jednego pracownika.
4. Jeśli poprawa jest powyżej 1.5x, przejdź do N=4 i zmierz ponownie. Malejące zyski zwykle pojawiają się w okolicy N=4-8.

Połącz z dekodowaniem spekulatywnym: każdy pracownik Hogwild! może niezależnie używać dekodowania spekulatywnego. Oba przyspieszenia mnożą się (z grubsza), dając 3x dekodowanie spekulatywne i 1.8x Hogwild! do efektywnego 5.4x w stosunku do naiwnego dekodowania z jednym pracownikiem.

## Dostarcz

Ta lekcja produkuje `outputs/skill-parallel-inference-router.md`. Dla profilu obciążenia rozumowania (budżet tokenów, profil równoległości zadań, rodzina modeli, cel wdrożenia) kieruje między strategiami głosowania, drzewa myśli, wieloagentową, Hogwild! i dekodowania spekulatywnego.

## Ćwiczenia

1. Uruchom `code/main.py` z domyślnymi ustawieniami. Potwierdź, że konfiguracja Hogwild! N=2 produkuje więcej tokenów pracy niż linia bazowa N=1 w tym samym czasie ściennym.

2. Zmniejsz siłę heurystyki koordynacji (ustaw `coordination_weight=0.1`). Uruchom ponownie. Pokaż, że przyspieszenie załamuje się. Wyjaśnij dlaczego: pracownicy dublują wysiłek, gdy nie mogą się koordynować.

3. Oblicz oczekiwane przyspieszenie Hogwild! dla zadania rozumowania 50k tokenów z `p=0.8, c=500` i N=4 pracownikami. Zrób to samo dla zadania czatowego 1k tokenów z `p=0.3, c=200` i N=4. Dlaczego jeden to zysk, a drugi strata?

4. Przeczytaj Sekcję 4 artykułu Hogwild! (wstępna ewaluacja). Zidentyfikuj dwa tryby awarii, które autorzy raportują. Opisz, jak lepsza podpowiedź koordynacyjna mogłaby złagodzić każdy z nich.

5. Połącz Hogwild! z dekodowaniem spekulatywnym w zabawce: każdy pracownik używa wewnętrznie dekodowania spekulatywnego 2-tokenowego. Raportuj mnożnikowe przyspieszenie. Jaki problem księgowy pojawia się, gdy dwóch pracowników chce rozszerzyć ten sam prefiks współdzielonej pamięci podręcznej?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Hogwild! | „Równolegli pracownicy, współdzielona pamięć podręczna" | N instancji tego samego LLM działających równolegle z jedną współdzieloną pamięcią podręczną KV; wyłaniająca się koordynacja poprzez samopodpowiadanie |
| Współdzielona pamięć podręczna KV | „Medium koordynacji" | Pojedynczy rosnący bufor KV, który wszyscy pracownicy czytają i do którego zapisują; umożliwia natychmiastową widoczność tokenów między pracownikami |
| Wyłaniająca się koordynacja | „Bez potrzeby trenowania" | LLM zdolne do rozumowania mogą czytać współdzieloną pamięć podręczną i dzielić pracę bez żadnego dostrajania ani jawnego protokołu |
| Narzut koordynacji (c) | „Tokeny wydane na orientację" | Koszt na pracownika czytania rozszerzonej pamięci podręcznej i decydowania, co robić; musi pozostać mały w stosunku do całkowitego czasu dekodowania |
| Ułamek możliwy do zrównoleglenia (p) | „Co może działać równolegle" | Równoległość na poziomie zadania: ułamek całkowitej pracy, która nie jest z natury sekwencyjna |
| RoPE umożliwia Hogwild! | „Pozycje obrotowe są niezmienne względem przesunięcia" | Ponieważ pozycje są obrotami, zapis do współdzielonej pamięci podręcznej nie wymaga przeliczania poprzednich tokenów |
| Zespół głosujący | „Uruchom N, wybierz większość" | Najprostsza topologia równoległego wnioskowania; przydatna do klasyfikacji, mniej do długich form rozumowania |
| Drzewo myśli | „Rozgałęziaj i przycinaj" | Strategia rozumowania, która eksploruje wiele gałęzi i przycina; jawna logika koordynacji |
| Framework wieloagentowy | „Przypisz podzadania" | Każdy agent dostaje rolę; koordynator orkiestruje; duży narzut protokołu |

## Dalsze czytanie

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — artykuł Hogwild!, wstępna ewaluacja na QwQ i DeepSeek-R1
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — oryginalny Hogwild!, źródło nazwy
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, właściwość, która czyni wnioskowanie ze współdzieloną pamięcią podręczną wykonalnym
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — strategia rozumowania drzewa myśli, do której Hogwild! jest ortogonalny
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — dekodowanie spekulatywne, równoległość wewnątrz-sekwencyjna, z którą Hogwild! się łączy
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — jedyne źródło prawdy dla eksperymentów z artykułu