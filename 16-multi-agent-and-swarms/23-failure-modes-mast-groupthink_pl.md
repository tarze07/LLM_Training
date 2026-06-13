# Tryby Awarii — MAST, Myślenie Grupowe, Monokultura, Kaskadowe Błędy

> Referencyjna taksonomia na 2026 to **MAST** (Cemri et al., NeurIPS 2025, arXiv:2503.13657), wyprowadzona z 1642 śladów wykonania w 7 najnowocześniejszych otwartoźródłowych MAS, pokazująca **41–86.7% wskaźnik awaryjności**. Trzy główne kategorie: **Problemy Specyfikacji** (41.77%) — niejednoznaczność ról, niejasne definicje zadań; **Awarie Koordynacji** (36.94%) — załamania komunikacji, desynchronizacja stanu; **Luki Weryfikacyjne** (21.30%) — brak walidacji, brak kontroli jakości. Rodzina **Myślenia Grupowego** (arXiv:2508.05687) dodaje: załamanie monokultury (ten sam model bazowy → skorelowane błędy), tendencyjność konformizmu (agenci wzmacniają nawzajem swoje błędy), deficyt teorii umysłu, dynamikę mieszanych motywacji, kaskadowe awarie niezawodności. Przykład kaskady: burze ponowień, gdzie nieudana płatność wyzwala ponowienia zamówień, które wyzwalają ponowienia zapasów, które przytłaczają usługę zapasów (10x obciążenia w sekundach — potrzebne wyłączniki obwodowe). Zatrucie pamięci: halucynacja jednego agenta trafia do współdzielonej pamięci, agenci niższego szczebla traktują ją jako fakt; dokładność spada stopniowo, co utrudnia diagnozę przyczyn źródłowych. **STRATUS** (NeurIPS 2025) raportuje 1.5x poprawę skuteczności łagodzenia dzięki wyspecjalizowanym agentom detekcji / diagnozy / walidacji. Ta lekcja traktuje tryby awarii jako pierwszorzędne cele inżynieryjne.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**Time:** ~75 minutes

## Problem

Systemy wieloagentowe zawodzą w 41-86.7% przypadków w rzeczywistych zadaniach (Cemri et al. 2025 zmierzyli to w 7 otwartoźródłowych MAS). Nie da się tego naprawić przez "po prostu dodaj więcej agentów." Awarie mają strukturalne przyczyny. Taksonomia MAST daje ci kategorie. Ta lekcja mapuje każdą kategorię na konkretny wzorzec detekcji, diagnozy i łagodzenia, aby liczby przestały wyglądać arbitralnie.

Praktyka produkcyjna z 2026 polega na traktowaniu trybów awarii jako danych wejściowych do projektu. Twoja architektura nie jest "wystarczająco dobra", dopóki nie możesz wskazać na każdą kategorię MAST i nazwać zastosowanego środka zaradczego.

## Koncepcja

### Kategorie MAST

**Problemy Specyfikacji (41.77% awarii).** Zadanie agenta nie zostało zdefiniowane wystarczająco precyzyjnie. Przykłady:

- Niejednoznaczność ról: dwóch agentów myśli, że to oni są recenzentem.
- Niedookreślone zadanie: "podsumuj to", gdy użytkownik chciał konkretnej perspektywy.
- Niejawna kryteria sukcesu: agent nie może stwierdzić, czy mu się udało.

Środki zaradcze:
- Pisz wyraźne kontrakty ról. Prompt każdego agenta określa, co robi *i czego nie robi*.
- Testy akceptacyjne na zadanie. Zanim agent zacznie, zdefiniuj "zrobione wygląda jak X."
- Kontrola specyfikacji przed uruchomieniem: oddzielny agent przegląda definicję zadania przed wysłaniem.

**Awarie Koordynacji (36.94%).** Załamania komunikacji lub stanu.

Przykłady:
- Dwóch agentów aktualizuje współdzielony stan bez synchronizacji.
- Wiadomość zagubiona między agentami (awaria kolejki, limit czasu).
- Dryf stanu: agent A myśli, że zadanie jest zrobione; agent B wciąż je wykonuje.

Środki zaradcze:
- Wersjonowany współdzielony stan z optymistyczną współbieżnością.
- Wyraźne potwierdzenie dla krytycznych wiadomości (ponawiaj aż do potwierdzenia).
- Okresowe punkty kontrolne synchronizacji stanu; wykrywaj dryf wcześnie.

**Luki Weryfikacyjne (21.30%).** Brak niezależnej kontroli wyników.

Przykłady:
- Jeden agent ogłasza sukces; nikt nie weryfikuje.
- Łańcuch agentów, z których każdy ufa wynikom poprzednika.
- Brak pokrycia testami emergentnego, złożonego zachowania.

Środki zaradcze:
- Niezależny agent weryfikator (Lekcja 13). Tylko do odczytu, niezależne źródło dostępu.
- Wyraźny kontrakt przekazania: "wynik A musi przejść przez kontroler C, zanim B zacznie."
- Rejestrowanie wyników do analizy post-hoc.

### Rodzina Myślenia Grupowego (arXiv:2508.05687)

Pięć powiązanych awarii, gdy agenci homogenizują się lub naśladują:

**Załamanie monokultury.** Ten sam model bazowy lub dane treningowe → skorelowane błędy. Gdy trzej agenci dzielą LLM, dzielą jego halucynacje.

**Tendencyjność konformizmu.** Agenci dostosowują się do najgłośniejszego lub najbardziej pewnego siebie rówieśnika, nawet gdy ten się myli.

**Deficyt ToM.** Agenci nie potrafią modelować przekonań innych; koordynacja się załamuje (Lekcja 18).

**Dynamika mieszanych motywacji.** Agenci z częściowo zgodnymi zachętami dryfują w kierunku kompromisowego środka, który nie satysfakcjonuje nikogo.

**Kaskadowe awarie niezawodności.** Wzorzec błędu jednego komponentu wyzwala wzorce błędów w zależnych komponentach.

### Przykład kaskady — burza ponowień

Klasyczny wzorzec incydentu z 2026:

```
usługa płatności zawodzi w 10% żądań
   ↓
agent zamówienia ponawia płatność (wykładnicze opóźnienie, ale naiwne)
   ↓
każde ponowienie to nowe sprawdzenie zamówienia-zapasów
   ↓
usługa zapasów widzi 2x normalne obciążenie
   ↓
usługa zapasów zaczyna przekraczać limity czasu
   ↓
każde zamówienie ponawia sprawdzenie zapasów
   ↓
usługa zapasów widzi 10x normalne obciążenie
   ↓
klaster ulega awarii
```

Rozwiązanie jest klasyczne: **wyłączniki obwodowe**. Gdy wskaźnik błędów niższego szczebla przekroczy próg, zewrzyj z wynikami z pamięci podręcznej lub domyślnymi. Plus ograniczone budżety ponowień na żądanie.

Wyłączniki obwodowe są jednym z niewielu środków zaradczych dla awarii wieloagentowych, które zapożyczasz wprost z systemów rozproszonych bez modyfikacji.

### Zatrucie pamięci (ponownie)

Z Lekcji 13: halucynacja jednego agenta staje się faktem we współdzielonej pamięci; agenci niższego szczebla rozumują na zatrutym fakcie. W terminologii MAST, jest to luka weryfikacyjna na warstwie współdzielonej pamięci.

Stopniowy spadek dokładności jest objawem. Nie dostajesz awarii; dostajesz powolny dryf, który jest trudny do zdiagnozowania.

Środek zaradczy: dziennik tylko do dopisywania, pochodzenie, niezapisywalny weryfikator. Już omówione w Lekcji 13.

### STRATUS — wyspecjalizowani agenci do wykrywania awarii

STRATUS (NeurIPS 2025) raportuje 1.5x poprawę skuteczności łagodzenia przy wdrożeniu:

- **Agent detekcji.** Obserwuje wzorce objawów (wysoki poziom niezgodności, skoki ponowień, dryf dokładności).
- **Agent diagnozy.** Na podstawie objawów wnioskuje prawdopodobną przyczynę źródłową z taksonomii MAST.
- **Agent walidacji.** Po zastosowaniu środka zaradczego sprawdza, czy objawy ustąpiły.

To jest styl reagowania na incydenty SRE, zastosowany do systemów agentowych. Trzy role mogą być agentami LLM z wyspecjalizowanymi promptami.

### Audyt trybów awarii

Najlepsza praktyka z 2026 to coroczny (lub na każde główne wydanie) audyt trybów awarii:

1. **Próbka śladów.** Zbierz ~1000 rzeczywistych śladów wykonania.
2. **Kategoryzuj.** Dla każdej awarii w śladzie, mapuj na kategorie MAST + Myślenie Grupowe.
3. **Oblicz wskaźnik awarii według kategorii.** Które kategorie dominują w twoim systemie?
4. **Uszereguj środki zaradcze.** Która naprawa wyeliminowałaby najwięcej awarii?
5. **Wybierz 2-3 środki zaradcze.** Zaimplementuj; przeprowadź ponowny audyt w następnym kwartale.

Dyscyplina jest ważniejsza niż konkretne wybory. Bez audytów awarie zlewają się w szum i nigdy nie są systematycznie adresowane.

### Gdy systemy zawodzą po cichu

Najbardziej niebezpieczną kategorią awarii jest cicha awaria poprawności. System, który zawodzi głośno (awaria, wyjątek, alert), może być monitorowany. System, który produkuje prawdopodobne, ale błędne wyniki, nie może być wykryty przez dzienniki wyjątków. Dlatego luki weryfikacyjne są najdroższą kategorią na awarię, mimo że stanowią tylko 21.30% pod względem liczby.

Inwestuj w:
- Próbkowy przegląd ludzki.
- Testy regresyjne na złotym zestawie danych.
- Wzajemne sprawdzanie między agentami dla ważnych wyników.

### Awaria vs powolna awaria

Niektóre awarie są natychmiastowe; niektóre są powolne. Natychmiastowe awarie (limit czasu, niezgodność schematu, błąd uwierzytelniania) są tanie do wykrycia. Powolne awarie (zatrucie pamięci, dryf monokultury, niejednoznaczność ról) są kosztowne do wykrycia i zapobiegania.

Posunięcie inżynieryjne z 2026: instrumentuj proxy powolnych awarii, abyś mógł złapać dryf, zanim stanie się widocznym błędem. Wskaźnik zgodności, wskaźnik ponowień, rozkład długości wyników i odległość edycyjna między kolejnymi wersjami agenta — wszystko to przydatne proxy.

## Build It

`code/main.py` implementuje:

- `FailureTaxonomy` — kategoryzuje symulowane incydenty na kategorie MAST + Myślenie Grupowe.
- `CircuitBreaker` — klasyczny wzorzec; otwiera się, gdy wskaźnik błędów przekroczy próg.
- `RetryStormSimulator` — pokazuje kaskadową awarię; przełącza wyłącznik obwodowy włączony / wyłączony.
- `DetectionAgent` — skryptowany dopasowywacz objawów w stylu STRATUS.

Uruchom:

```
python3 code/main.py
```

Oczekiwane wyniki:
- burza ponowień bez wyłącznika obwodowego: błędy zapasów eksplodują (symulowane).
- z wyłącznikiem obwodowym: ograniczenie do progu; odpowiedzi w trybie zdegradowanym.
- agent detekcji flaguje wzorzec i nazywa kategorię MAST.

## Use It

`outputs/skill-mast-auditor.md` przeprowadza audyt trybów awarii w stylu MAST w systemie wieloagentowym. Ślady → kategoryzacja → ranking środków zaradczych.

## Ship It

Dyscyplina trybów awarii w produkcji:

- **Audyt MAST co kwartał.** Nie coroczny. Kategorie zmieniają się, gdy twój system rośnie.
- **Wyłączniki obwodowe wszędzie.** Każde wywołanie wychodzące do dowolnej zależnej usługi. Domyślny próg otwarcia przy 5-10% wskaźniku błędów.
- **Złote zestawy danych.** Małe, wysokiej jakości, ręcznie audytowane. Testy regresyjne co tydzień.
- **Triada STRATUS.** Agenci detekcji + diagnozy + walidacji monitorujący produkcję. Zacznij tylko od agenta detekcji; dodaj diagnozę, gdy objawy są hałaśliwe.
- **Budżet awarii.** Wyraźny SLO dla wskaźnika awarii według kategorii. Przekroczenie budżetu wyzwala rozmowę o wstrzymaniu wydawania.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że wyłącznik obwodowy ogranicza burzę ponowień. Zmień próg awarii i zaobserwuj kompromis.
2. Zaimplementuj **proxy powolnej awarii**: wskaźnik zgodności między 3 równoległymi agentami. Gdy gwałtownie spadnie, wyzwól alert. Zasymuluj dryf monokultury, stopniowo korelując wyniki agentów.
3. Przeczytaj Cemri et al. (arXiv:2503.13657). Wybierz jeden z ich 7 systemów MAS i zmapuj jego 3 najważniejsze kategorie awarii. Jak wypadają w porównaniu z tym, co przewiduje MAST?
4. Przeczytaj artykuł o Myśleniu Grupowym (arXiv:2508.05687). Zidentyfikuj, który z pięciu wzorców jest najtrudniejszy do wykrycia w produkcji. Zaproponuj metrykę proxy.
5. Zaprojektuj triadę w stylu STRATUS (detekcja-diagnoza-walidacja) dla konkretnego systemu wieloagentowego, który znasz. Na jakie objawy czuwa detekcja? Jakie środki zaradcze zaleca diagnoza? Jak walidacja potwierdza, że działają?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|----------------|------------------------|
| MAST | "Taksonomia z 2026" | Cemri 2025; 3 główne kategorie + 14 podtypów awarii. |
| Problem Specyfikacji | "Niejednoznaczność ról" | Zadanie lub rola niedookreślone; agenci nie wiedzą, co robić. |
| Awarie Koordynacji | "Dryf stanu" | Załamanie komunikacji lub synchronizacji między agentami. |
| Luka Weryfikacyjna | "Nikt nie sprawdził" | Wyniki przyjęte bez niezależnej walidacji. |
| Rodzina Myślenia Grupowego | "Awarie homogeniczności" | Monokultura, konformizm, deficyt ToM, mieszane motywacje, kaskadowość. |
| Załamanie monokultury | "Ten sam model, te same halucynacje" | Skorelowane błędy z tego samego modelu bazowego lub danych treningowych. |
| Burza ponowień | "Kaskadowe wzmocnienie błędu" | Jedna awaria wyzwala ponowienia, które wzmacniają obciążenie w dół strumienia. |
| Wyłącznik obwodowy | "Szybkie odrzucanie przy wskaźniku błędów" | Otwiera się, gdy wskaźnik błędów przekroczy próg; zwieranie z domyślną wartością. |
| STRATUS | "Triada reagowania na incydenty" | Agenci detekcji + diagnozy + walidacji. 1.5x skuteczność łagodzenia. |
| Zatrucie pamięci | "Halucynacje się propagują" | Fakt we współdzielonej pamięci skażony; agenci niższego szczebla rozumują na truciźnie. |

## Dalsza lektura

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — taksonomia MAST, NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — monokultura, konformizm i pięciorodzinna taksonomia
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — wpis w materiałach NeurIPS 2025 (detekcja + diagnoza + walidacja)
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — kanoniczna referencja wyłącznika obwodowego
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — notatki o produkcyjnych trybach awarii