# AI Scientist v2 — Autonomiczne Badania na Poziomie Warsztatów

> AI Scientist v2 Sakany (Yamada i in., arXiv:2504.08066) uruchamia pełną pętlę badawczą: hipoteza, kod, eksperymenty, ryciny, opis, zgłoszenie. Jest pierwszym systemem, którego wygenerowany artykuł przeszedł recenzję na warsztatach ICLR 2025. Niezależna ocena (Beel i in.) wykazała, że 42% eksperymentów zawiodło z powodu błędów kodowania, a przegląd literatury często błędnie oznaczał ugruntowane koncepcje jako nowatorskie. Własne dokumenty Sakany ostrzegają, że kod wykonuje kod napisany przez LLM i zalecają izolację Docker. Obie połowy tego obrazu są sednem sprawy.

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Problem

Badania to zadanie otwarto-końcowe. W przeciwieństwie do przeszukiwania algorytmicznego AlphaEvolve czy samomodyfikacji ograniczonej benchmarkiem DGM, wynik badawczy nie ma maszynowo-sprawdzalnego kryterium poprawności. Artykuł jest oceniany przez recenzentów, a nie testy jednostkowe. To czyni pętlę trudniejszą do zamknięcia — i bardziej wartościową, jeśli jest zamknięta, ponieważ badania to miejsce, gdzie żyje kumulatywny postęp.

AI Scientist v1 (Sakana, 2024) zamykał pętlę, zaczynając od szablonów napisanych przez człowieka. LLM wypełniał eksperymenty w ramach stałego scaffoldingu. AI Scientist v2 (Yamada i in., 2025) usuwa wymóg szablonu, używając agencyjnego przeszukiwania drzewa z pętlą krytyki modelu wizyjno-językowego. System generuje pomysły, implementuje eksperymenty, produkuje ryciny, pisze artykuł i iteruje na opinii recenzentów.

Werdykt recenzji: jeden artykuł wygenerowany przez v2 został zaakceptowany na warsztatach ICLR 2025 (z ujawnieniem). Werdykt niezależnej oceny: system jest daleki od niezawodności. Oba są prawdziwe.

## Koncepcja

### Architektura

1. **Generowanie pomysłów.** LLM proponuje pomysły badawcze warunkowane tematem i wcześniejszą literaturą. v1 używał szablonów; v2 używa agencyjnego przeszukiwania przestrzeni hipotez.
2. **Sprawdzenie nowości.** Krok wyszukiwania literatury sprawdza, czy pomysł został opublikowany. To na tym kroku ocena Beela i in. znalazła błędne oznaczanie — ugruntowane metody często klasyfikowane jako nowatorskie.
3. **Plan eksperymentu.** Agent sporządza protokół eksperymentalny i pisze kod.
4. **Wykonanie.** Kod uruchamia się w piaskownicy. Błędy są zwracane do pętli ponawiania. W pomiarach Beela i in., 42% eksperymentów zawiodło z powodu błędów kodowania na tym etapie.
5. **Generowanie rycin.** Model wizyjno-językowy czyta wygenerowane ryciny i przepisuje je dla przejrzystości. To był kluczowy dodatek techniczny v2.
6. **Opis.** LLM pisze artykuł, iteruje z wewnętrznym recenzentem.
7. **Opcjonalnie: zgłoszenie.** Artykuł jest zgłaszany do wydarzenia.

### Co oznacza wynik akceptacji na warsztatach

Jeden artykuł wygenerowany przez v2 przeszedł recenzję na warsztatach ICLR 2025. Autorzy ujawnili pochodzenie artykułu komitetowi programowemu. Akceptacja jest punktem danych; nie jest licencją na twierdzenie, że system "prowadzi badania."

Ważny kontekst: artykuły warsztatowe to niższy próg niż artykuły na głównej konferencji. Recenzja jest głośna; mała frakcja zgłoszeń jest akceptowana każdego dnia. Jeden sukces to dowód koncepcji, nie twierdzenie o niezawodności. Artykuł z Nature 2026 dokumentuje pętlę end-to-end i sam był współautorstwa ludzkich badaczy; nie jest to "system napisał artykuł do Nature."

### Co znalazła niezależna ocena

Beel i in. (arXiv:2502.14297) przeprowadzili zewnętrzną ocenę. Główne ustalenia:

- **Błędy eksperymentów.** 42% eksperymentów zawiodło z powodu błędów kodowania (złe importy, niedopasowania kształtów, niezdefiniowane zmienne). Pętla ponawiania złapała niektóre, ale nie wszystkie.
- **Błędne oznaczanie nowości.** Krok wyszukiwania literatury często oznaczał ugruntowane koncepcje jako nowatorskie. To jest odpowiednik halucynacji w badaniach.
- **Luka w jakości prezentacji.** Krytyka rycin przez model wizyjno-językowy produkowała grafikę na poziomie publikacji, maskując słabości eksperymentalne.

Ostatnie ustalenie jest ważne dla tej fazy. System, który produkuje przekonujące wyniki bez prowadzenia przekonujących badań, jest bardziej niebezpieczny, a nie bezpieczniejszy, niż ten, który zawodzi oczywiście. Ocena musi sięgać do leżących u podstaw twierdzeń, a nie zatrzymywać się na rycinie.

### Obawa o ucieczkę z piaskownicy

Własny plik README repozytorium Sakany ostrzega:

> Ze względu na naturę tego oprogramowania, które wykonuje kod wygenerowany przez LLM, nie możemy zagwarantować bezpieczeństwa. Istnieje ryzyko niebezpiecznych pakietów, niekontrolowanego dostępu do sieci i tworzenia niezamierzonych procesów. Używasz na własne ryzyko i rozważ izolację Docker.

To jest operacyjny kształt autonomii w niezweryfikowanej dziedzinie. LLM pisze kod; kod się uruchamia; kod może zrobić wszystko, co proces ma uprawnienia zrobić. Bez piaskownicy, która twardo ogranicza system plików, sieć i działania procesów, każdy samodzielny agent badawczy może eksfiltrować dane, spalać moc obliczeniową lub przepisać samego siebie.

Historia piaskownicy AlphaEvolve jest łatwiejsza, ponieważ jego ewaluator jest szczelny. Pętla AI Scientist v2 uruchamia otwarto-końcowy kod z otwarto-końcowymi celami. Dlatego potrzebuje silniejszej izolacji (minimum Docker; preferowane seccomp / gVisor) i ręcznego przeglądu każdego zgłoszenia przed opuszczeniem systemu.

### Gdzie v2 siedzi w stosie technologicznym

| System | Cel | Rodzaj wyniku | Ewaluator | Znana awaria |
|---|---|---|---|---|
| AlphaEvolve | algorytmy | kod | jednostkowy + benchmark | ograniczona rygo rem ewaluatora |
| DGM | scaffolding agenta | kod | SWE-bench | hakowanie nagrody |
| AI Scientist v2 | artykuły badawcze | tekst + kod + ryciny | recenzja (słaba) | błędy eksperymentów, błędne oznaczanie, polerowanie maskujące słabości |

v2 ma najsłabszy automatyczny ewaluator z trzech, najszerszą powierzchnię wyjściową i najkrótszą ścieżkę do publicznych artefaktów. Kontrole operacyjne (piaskownica, przegląd, ujawnienie) wykonują większość pracy bezpieczeństwa.

## Użyj

`code/main.py` symuluje pętlę v2 jako maszynę stanów: pomysł → sprawdzenie nowości → eksperyment → rycina → opis → recenzja → akceptacja-lub-iteracja. Każdy stan ma konfigurowalne prawdopodobieństwo awarii zaczerpnięte z ustaleń Beela i in. Uruchom symulator dla N pętli i policz:

- Ile pomysłów dociera do zgłoszenia.
- Ile zgłoszeń miałoby krytyczną wadę eksperymentalną, którą wypolerowany artykuł ukrywa.
- Jak budżety ponawiania równoważą jakość vs wydajność.

## Dostarcz

`outputs/skill-ai-scientist-sandbox-review.md` to dwubramkowa lista kontrolna przeglądu dla wszystkiego, co zostało wyprodukowane przez pętlę badawczą, zanim opuści piaskownicę.

## Ćwiczenia

1. Uruchom `code/main.py` z domyślnymi parametrami. Jaka frakcja przebiegów pętli produkuje "czysty" artykuł? Jaka frakcja produkuje artykuł z wadą błędu eksperymentalnego, którą krytyka rycin wypolerowała?

2. Wartości domyślne już używają 42% / 25% Beela i in. Uruchom ponownie z `--experiment-failure 0.20 --novelty-mislabel 0.10`, a następnie z `--experiment-failure 0.60 --novelty-mislabel 0.40`. Jak zmienia się udział wypolerowanych, ale wadliwych między tymi dwoma uruchomieniami?

3. Przeczytaj plik README repozytorium AI Scientist v2 Sakany na temat wymagań piaskownicy. Wymień dwa dodatkowe ograniczenia (poza Dockerem), które byś zastosował dla wielodniowego autonomicznego przebiegu.

4. Przeczytaj Sekcję 4 Beela i in. o luce w jakości prezentacji. Zaprojektuj jeden dodatkowy ewaluator, który złapałby wypolerowane, ale eksperymentalnie wadliwe artykuły.

5. Zaproponuj protokół recenzji ludzkiej dla wyników agenta badawczego, który skaluje się lepiej niż "doktorant czyta każdy artykuł." Zidentyfikuj wąskie gardło i zaprojektuj wokół niego.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|---|---|---|
| AI Scientist v1 | "Szablonowy agent badawczy Sakany" | Wypełniał eksperymenty w stałym szkielecie |
| AI Scientist v2 | "Agent badawczy bez szablonu" | Agencyjne przeszukiwanie drzewa z krytyką rycin VLM |
| Agencyjne przeszukiwanie drzewa | "Rozgałęziający agent badawczy" | Rozwija wiele planów eksperymentalnych równolegle; przycina przez wewnętrznego krytyka |
| Krytyka wizyjno-językowa | "Polerowanie rycin przez VLM" | Model multimodalny czyta ryciny i przepisuje je dla przejrzystości |
| Wyszukiwanie literatury | "Sprawdzenie nowości" | Przeszukuje wcześniejsze prace, aby potwierdzić nowość pomysłu — udokumentowano błędne oznaczanie |
| Maskujące polerowanie | "Ładny artykuł, złamane badania" | Jakość prezentacji przewyższa jakość eksperymentalną; ukrywa słabości |
| Ucieczka z piaskownicy | "Kod LLM ucieka" | Kod wykonany przez agenta robi rzeczy, których projektant pętli nie zamierzał |

## Dalsza lektura

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — artykuł.
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — podsumowanie autora z kontekstem recenzji.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — liczby z zewnętrznej oceny.
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — szablonowy poprzednik.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — szersze ramy dla otwarto-końcowych agentów badawczych.