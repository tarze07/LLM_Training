# Negocjacje i Targowanie

> Agenci negocjują zasoby, ceny, przydziały zadań i warunki. Zestaw benchmarków na 2026 rok jest jasny: NegotiationArena (arXiv:2402.05863) pokazuje, że LLMy mogą poprawić wypłaty o ~20% poprzez manipulację personą („desperacja"); „Measuring Bargaining Abilities" (arXiv:2402.15813) pokazuje, że kupujący ma trudniejsze zadanie niż sprzedający, a skala nie pomaga — ich **OG-Narrator** (deterministyczny generator ofert + narrator LLM) podniósł wskaźnik zawierania umów z 26.67% do 88.88%; Wielkoskalowy Autonomiczny Konkurs Negocjacyjny (arXiv:2503.06416) przeprowadził ~180 tys. negocjacji i odkrył, że agenci z **ukrywaniem łańcucha myśli** wygrywają poprzez ukrywanie rozumowania przed przeciwnikami; Bhattacharya i in. 2025 w oparciu o metryki Harvard Negotiation Project sklasyfikowali Llama-3 jako najbardziej efektywnego, Claude-3 jako agresywnego, GPT-4 jako najuczciwszego. Ta lekcja implementuje Contract Net Protocol (protokół przodka FIPA, Lekcja 02), łączy kupującego/sprzedającego w stylu LLM, uruchamia dekompozycję w stylu OG-Narrator i mierzy, jak wskaźnik zawierania umów zmienia się wraz z każdym wyborem strukturalnym.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

Dwóch agentów musi uzgodnić cenę. Pozostawieni samym sobie z czystymi promptami językowymi, LLMy z lat 2024-2026 zawierają umowy z zaskakująco niską częstotliwością (~27% w ściśle parametryzowanych targach w arXiv:2402.15813). Skala tego nie naprawia: GPT-4 nie jest strukturalnie lepszy w targowaniu niż GPT-3.5; jest lepszy w *języku* targowania.

Główny problem polega na tym, że LLMy łączą dwie role — podejmowanie decyzji o ofercie i narrację oferty. OG-Narrator rozdzielił te funkcje: deterministyczny generator ofert oblicza ruchy numeryczne; LLM tylko narratywnie je oprawia. Wskaźnik zawierania umów skacze do ~89%.

Odzwierciedla to klasyczne odkrycie w systemach wieloagentowych: oddzielenie mechanizmu od warstwy komunikacyjnej wygrywa. Contract Net Protocol (FIPA, 1996; Smith, 1980) jest referencyjnym mechanizmem rynku zadań. Podłącz LLM do roli narratora i otrzymujesz nowoczesny rynek zadań napędzany przez LLM.

## Koncepcja

### Contract Net w jednym akapicie

Contract Net Protocol Smitha z 1980 roku: **menedżer** rozgłasza **zapytanie o oferty (cfp)**; **oferenci** odpowiadają wiadomościami **propose** zawierającymi ich oferty; menedżer wybiera zwycięzcę i wysyła **accept-proposal** do zwycięzcy oraz **reject-proposal** do przegranych. Zwycięzca wykonuje pracę. Opcjonalna wiadomość: **refuse** (oferent odmawia złożenia oferty). FIPA skodyfikowała to jako protokół interakcji `fipa-contract-net`.

### Dlaczego OG-Narrator wygrywa

„Measuring Bargaining Abilities of Language Models" (arXiv:2402.15813) zaobserwował, że:

- LLMy często łamią zasady targowania (oferują po absurdalnych cenach, ignorują ZOPA drugiej strony).
- Źle kotwiczą (akceptują złe pierwsze oferty; składają kontroferty na kwoty symboliczne, a nie strategiczne).
- Sama skala tego nie naprawia. Większe modele produkują bardziej prawdopodobny język z podobnymi błędami strategicznymi.

Dekompozycja OG-Narrator:

```
           ┌──────────────────┐        ┌──────────────────┐
  stan  → │ generator ofert  │ cena → │  narrator LLM    │ → wiadomość
           │  (deterministyczny)      │  (pisze          │
           │                  │        │   oprawę         │
           └──────────────────┘        │   w ludzkim      │
                                        │   stylu)         │
                                        └──────────────────┘
```

Generator ofert to klasyczna strategia negocjacyjna: model targowania Rubinsteina, strategia Zeuthena lub proste tit-for-tat na cenie. LLM narratywnie oprawia. Wiadomość zawiera deterministyczną cenę i naturalnojęzykową ramkę.

Wskaźnik zawierania umów skacze, ponieważ:
- Ceny pozostają w strefie targowania.
- Kotwice są strategiczne, a nie emocjonalne.
- LLM robi to, w czym jest dobry: pisanie.

### Odkrycia NegotiationArena

arXiv:2402.05863 dostarcza kanoniczny benchmark. Główne odkrycia:

- LLMy mogą poprawić wypłaty o ~20% poprzez przyjmowanie person („Muszę to sprzedać do piątku, jestem zdesperowany") — manipulacja personą to realna taktyka.
- Uczciwi/kooperatywni agenci są wykorzystywani przez adversarialnych; obrona wymaga jawnego przeciwstawnego pozycjonowania.
- Symetryczne pary zbiegają się do nieuczciwych wyników w około 40% scenariuszy benchmarkowych.

To nie znaczy, że „LLMy są złymi negocjatorami." To znaczy, że „LLMy negocjują zbyt podobnie do ludzi, łącznie z częściami podatnymi na wykorzystanie."

### Ukrywanie łańcucha myśli

Wielkoskalowy Autonomiczny Konkurs Negocjacyjny (arXiv:2503.06416) przeprowadził ~180 tys. negocjacji z wieloma strategiami LLM. Zwycięzcy ukrywali swoje rozumowanie przed przeciwnikami:

- Jeśli agent wypisze „zejdę tylko do 75 zł; moja cena rezerwacyjna to 70 zł" do publicznie widocznego scratchpada, przeciwnik to czyta.
- Zwycięzcy obliczają strategię prywatnie; kanał wyjściowy zawiera tylko ofertę i niezbędne minimum narracji.

Jest to echo klasycznej teorii gier (Aumann 1976 o racjonalności i informacji): ujawnienie swojej prywatnej wyceny kosztuje wypłatę. LLMy nie intuicyjnie tego rozumieją i chętnie wpisują swoje rezerwacje w śladach rozumowania, które stają się widoczne dla przeciwnika.

Wniosek inżynieryjny: oddziel prywatny kontekst scratchpada od publicznego kontekstu wiadomości. To nie jest opcjonalne.

### Bhattacharya i in. 2025 — rankingi modeli

W oparciu o metryki Harvard Negotiation Project (negocjacje oparte na zasadach, szacunek dla BATNA, wzajemność interesów):

- **Llama-3** była najbardziej efektywna w zawieraniu umów (wskaźnik + wypłata).
- **Claude-3** był najbardziej agresywnym negocjatorem (wysokie kotwice, późne ustępstwa).
- **GPT-4** był najuczciwszy (najmniejsza wariancja wypłaty w parach).

To migawka z 2025 roku. Chodzi nie o to, który model wygrywa w kwietniu 2026 — chodzi o to, że różne modele bazowe mają trwałe style negocjacyjne. Heterogeniczne zespoły (Lekcja 15) uwzględniają to jako źródło różnorodności.

### Przydział zadań przez Contract Net + LLM

Nowoczesne wykorzystanie Contract Net dla wieloagentowych LLM:

1. Agent menedżer dekomponuje zadanie na jednostki.
2. Rozgłasza `cfp` z opisem zadania do agentów pracowników.
3. Każdy pracownik zwraca ofertę: `(cena, eta, pewność)`, gdzie cena może być w tokenach, jednostkach obliczeniowych lub dolarach.
4. Menedżer wybiera zwycięzców (jednego lub wielu, w zależności od zadania) i przyznaje zadanie.
5. Odrzuceni pracownicy mogą składać oferty na inne zadania.

Skaluje się dobrze powyżej 100 pracowników, ponieważ koordynacja to rozgłaszanie i odpowiedź, a nie synchroniczny czat. Używane w produkcji: wzorce orkiestracyjne Microsoft Agent Framework, niektóre implementacje LangGraph.

### Interaktywne Negocjacje Interesariuszy LLM

NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) wprowadza wielostronne gry punktowane z **tajnymi wynikami** i **minimalnymi progami akceptacji**. Każdy interesariusz ma prywatne użyteczności; LLM musi je wywnioskować z wiadomości. Jest to uogólnienie dwustronnego targowania na tworzenie koalicji N-stronnych. Istotne dla produkcyjnych rynków zadań z heterogenicznymi możliwościami pracowników.

### Reguła narracja-kontra-mechanizm

We wszystkich benchmarkach negocjacyjnych z lat 2024-2026 spójna reguła inżynieryjna brzmi:

> Pozwól LLM-owi narratywnie oprawiać. Nie pozwól LLM-owi obliczać oferty.

Jeśli oferta musi być liczbą (cena, ETA, ilość), wygeneruj ją deterministycznie na podstawie stanu negocjacji i pozwól LLM-owi wyprodukować oprawę. Jeśli oferta musi być strukturą propozycji (dekompozycja zadania, przypisanie ról), pozwól LLM-owi ją naszkicować, ale zweryfikuj ją względem schematu i sprawdź ograniczenia przed wysłaniem.

## Build It

`code/main.py` implementuje:

- `ContractNetManager`, `ContractNetTask`, `Bid` — menedżer + oferenci, rozgłaszanie cfp, zbieranie propozycji, przyznawanie.
- `og_narrator_bargain(state, rng)` — kupujący OG-Narrator: deterministyczne ustępstwo w stylu Zeuthena w kierunku punktu środkowego.
- `seller_response(state, rng)` — deterministyczna polityka kontroferty sprzedającego (prawda strukturalna dla obu stylów).
- `naive_llm_bargain(state, rng)` — symuluje negocjatora w pełni LLM: wybiera ceny z wysoką wariancją, często poza ZOPA.
- Pomiar: wskaźnik zawierania umów w 1000 prób z nowymi cenami rezerwacyjnymi próbkowanymi w każdej próbie.

Uruchomienie:

```
python3 code/main.py
```

Oczekiwane wyjście: wskaźnik zawierania umów dla naiwnego LLM ~65-75%; wskaźnik dla OG-Narrator ~85-95%; różnica 15-25 punktów procentowych to strukturalna przewaga dekompozycji generowania oferty od narracji. Plus przykład alokacji zadań na rynku Contract Net z trzema oferentami i jednym zadaniem.

## Use It

`outputs/skill-bargainer-designer.md` projektuje protokół negocjacyjny: kto generuje oferty (deterministycznie czy LLM), kto narratywnie oprawia, jak prywatne scratchpady są oddzielone od publicznych wiadomości i jak monitorowany jest wskaźnik zawierania umów.

## Ship It

Lista kontrolna dla produkcyjnych negocjacji:

- **Oddzielny scratchpad.** Prywatny stan nigdy nie trafia do kontekstu przeciwnika. To nie podlega negocjacjom.
- **Deterministyczne generowanie ofert.** Ceny, ilości, ETA: obliczaj, nie promptuj.
- **Waliduj wszystkie przychodzące oferty** względem schematu. Odrzucaj oferty poza ZOPA na granicy protokołu.
- **Ogranicz rundy.** Maksymalnie 3-5 rund; eskalacja do mediatora w przypadku impasu.
- **Mierz wskaźnik zawierania umów i wariancję wypłaty** w sposób ciągły. Spadający wskaźnik to symptom — często dryft promptu lub atak ze strony przeciwnika.
- **Loguj wszystkie odrzucone propozycje** z deterministycznym uzasadnieniem. Dla menedżerów Contract Net, przegrani oferenci muszą rozumieć dlaczego.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że OG-Narrator bije naiwnego LLM pod względem wskaźnika zawierania umów. O ile?
2. Zaimplementuj **poprawę wypłaty opartą na personie** (arXiv:2402.05863) — kupujący przyjmuje personę „zdesperowany, by kupić w tym tygodniu" tylko w narracji, generator ofert bez zmian. Czy wskaźnik zawierania umów lub wypłata się zmienia?
3. Zaimplementuj **ukrywanie** łańcucha myśli: utrzymuj prywatny scratchpad, który nie jest przekazywany przeciwnikowi. Co się stanie, jeśli przypadkowo go wyciekniesz (symuluj przez zamianę kanałów)?
4. Rozszerz Contract Net na aukcję z N oferentów z ceną rezerwową. Gdy wszystkie oferty przekraczają cenę rezerwową, jak menedżer decyduje między najniższą ceną a najwyższą jakością? Którą regułę przyznawania wybierasz i dlaczego?
5. Przeczytaj Bhattacharya i in. 2025 o metrykach Harvard Negotiation Project. Zaimplementuj dwóch negocjatorów z różnymi stylami (agresywny vs. uczciwy). Zmierz wariancję wypłaty w parach symetrycznych i asymetrycznych.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Contract Net | „Rynek zadań" | Smith 1980, FIPA 1996. cfp + propose + accept/reject. Kanoniczny rynek zadań. |
| ZOPA | „Strefa możliwego porozumienia" | Nakładanie się maksimum kupującego i minimum sprzedającego. Oferty poza nią nie mogą zostać zawarte. |
| BATNA | „Najlepsza alternatywa dla negocjowanego porozumienia" | Twoja opcja zapasowa, jeśli ta umowa upadnie. Ustala twoją cenę rezerwacyjną. |
| OG-Narrator | „Generator ofert + narrator" | Dekompozycja: deterministyczna oferta, narracja LLM. |
| Strategia Zeuthena | „Ustępstwo minimalizujące ryzyko" | Klasyczny generator ofert, który ustępuje w oparciu o limity ryzyka. |
| Targowanie Rubinsteina | „Równowaga ofert naprzemiennych" | Model teorii gier dla targowania w nieskończonym horyzoncie z dyskontowaniem. |
| CoT concealment | „Ukryj swoje rozumowanie" | Zwycięzcy w arXiv:2503.06416 utrzymywali prywatne scratchpady; publiczny kanał pokazuje tylko ofertę. |
| Manipulacja personą | „Pozycjonowanie emocjonalne" | arXiv:2402.05863: ~20% wzrost wypłaty z person desperacji/pilności. |

## Dalsza Literatura

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — benchmark; odkrycia dotyczące manipulacji personą i wykorzystywania
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator i wynik, że kupujący ma trudniej niż sprzedający
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — ~180 tys. negocjacji; ukrywanie łańcucha myśli wygrywa
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — wielostronne gry punktowane z tajnymi użytecznościami
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — klasyczny mechanizm, IEEE Transactions on Computers