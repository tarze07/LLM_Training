# AlphaEvolve — Ewolucyjne Agenci Kodowania

> Połącz graniczny model kodowania z pętlą ewolucyjną i maszynowo-sprawdzalnym ewaluatorem. Pozwól pętli działać wystarczająco długo. Odkrywa procedurę mnożenia macierzy zespolonych 4x4, która używa 48 mnożeń skalarnych — pierwsze ulepszenie względem Strassena od 56 lat. Znajduje też heurystykę szeregowania Borg w skali Google, która odzyskuje ~0,7% mocy obliczeniowej klastra w produkcji. Architektura jest celowo nudna. Zwycięstwa pochodzą z rygoru ewaluatora.

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60 minutes

## Problem

Duże modele językowe potrafią pisać kod. Algorytmy ewolucyjne potrafią przeszukiwać kod. Obie próbowano osobno przez dekady; obie napotkały sufity. Sufitem LLM jest konfabulacja: model pisze wiarygodny kod, który nie robi tego, co twierdzi. Sufitem ewolucyjnym jest koszt przeszukiwania: losowe mutacje składni rzadko produkują kompilowalne programy, a tym bardziej lepsze.

AlphaEvolve (Novikov i in., DeepMind, arXiv:2506.13131, czerwiec 2025) łączy je. LLM proponuje celowane edycje bazy programów; automatyczny ewaluator ocenia każdy wariant; wysoko ocenione warianty stają się rodzicami dla przyszłych pokoleń. LLM zajmuje się kosztownym krokiem pisania wiarygodnego kodu; ewaluator łapie konfabulacje. Pętla działa godzinami lub tygodniami.

Zgłoszone wyniki: mnożenie macierzy zespolonych 4x4 z 48 mnożeniami skalarnymi (granica Strassena z 1969 roku wynosiła 49), heurystyka szeregowania Borg w produkcji Google, przyspieszenie jądra FlashAttention o 32,5%, ulepszenia przepustowości trenowania Gemini.

Architektura działa, ponieważ ewaluator jest maszynowo-sprawdzalny. Nie działa tam, gdzie ewaluator nie jest. Ta asymetria jest lekcją.

## Koncepcja

### Pętla

1. Zacznij od programu początkowego `P_0`, który jest poprawny, ale nieoptymalny.
2. Utrzymuj bazę wariantów programów, każdy oceniony przez ewaluatora.
3. Próbkuj jednego lub więcej rodziców z bazy (w stylu MAP-elites lub wyspowym).
4. Poproś LLM (Gemini Flash dla wielu kandydatów, Gemini Pro dla trudnych) o wyprodukowanie zmodyfikowanego wariantu rodzica.
5. Skompiluj, uruchom i oceń wariant na wydzielonym ewaluatorze.
6. Wstaw do bazy z kluczem według wyniku i wektora cech.
7. Powtórz.

Dwa szczegóły mają znaczenie. Po pierwsze, LLM otrzymuje w prompcie więcej niż tylko program rodzica — zazwyczaj kilka najlepszych wariantów z bazy, plus sygnaturę ewaluatora, plus krótki opis zadania. Zadaniem modelu jest zaproponowanie celowanej zmiany, która może poprawić wynik. Po drugie, baza jest ustrukturyzowana (siatka MAP-elites, wyspowa), aby pętla eksplorowała różnorodność, a nie tylko obecnego lidera.

### Co sprawia, że ewaluator jest niepodlegający negocjacjom

Zwycięstwa AlphaEvolve wszystkie pochodzą z dziedzin, gdzie ewaluator jest szybki, deterministyczny i trudny do oszukania:

- **Algorytm mnożenia macierzy**: test jednostkowy, który mnoży macierze i sprawdza równość bit po bicie.
- **Heurystyka szeregowania Borg**: symulator klasy produkcyjnej, który odtwarza historyczne obciążenie klastra i mierzy zmarnowaną moc obliczeniową.
- **Jądro FlashAttention**: test poprawności plus benchmark czasu rzeczywistego na prawdziwym sprzęcie.
- **Przepustowość trenowania Gemini**: zmierzone GPU-sekund na krok.

W każdym przypadku ewaluator łapie klasę błędów LLM, które w przeciwnym razie dominowałyby: konfabulowane twierdzenia o poprawności, twierdzenia o wydajności, które znikają na sprzęcie, oraz awarie na przypadkach brzegowych. Usuń ewaluator, a pętla optymalizuje pod kątem ładnego kodu.

### Hakowanie nagrody jest drugą stroną tego twierdzenia

Ewolucja optymalizuje pod kątem tego, co mierzy ewaluator. Jeśli ewaluator jest niedoskonały, pętla znajdzie niedoskonałość. W niezweryfikowanej dziedzinie pętla optymalizowałaby pod kątem powierzchownej cechy, a nie zamierzonego zachowania. DeepMind wyraźnie to sygnalizuje w artykule: sukcesy AlphaEvolve przenoszą się tylko do dziedzin, gdzie rygor ewaluatora dorównuje ambicji przeszukiwania.

Konkretne przykłady hakowania nagrody w pętlach przeszukiwania kodu z lat 2025-2026:

- Cele optymalizacji nagradzające "czas do ukończenia" nagradzały zgłaszanie pustych rozwiązań.
- Wyniki benchmarków nagradzające poprawność w testach nagradzały zapamiętywanie testów i przetrenowanie.
- Proxy "jakości kodu" nagradzało usuwanie komentarzy i zmianę nazw zmiennych, bez zmiany semantycznej.

Naprawa w AlphaEvolve: dostarcz wydzielony ewaluator, którego LLM nigdy nie widział, z danymi wejściowymi generowanymi w czasie oceny. Nawet wtedy DeepMind zaleca silny przegląd każdego proponowanego wdrożenia.

### Dlaczego LLM + przeszukiwanie bije każde z osobna

LLM może produkować kompilowalne, semantycznie wiarygodne modyfikacje. Losowo mutujący GA na 2000-linijkowym pliku Pythona prawie zawsze produkuje błędy składniowe. LLM także koncentruje przeszukiwanie na wiarygodnych sąsiedztwach (zmień jedną funkcję, nie losowe bajty), co dramatycznie redukuje zmarnowane wywołania ewaluatora.

Ewaluator z kolei łapie konfabulacje LLM. Modele językowe z przekonaniem twierdzą, że funkcja "jest O(n log n) w granicy", gdy w rzeczywistości jest O(n^2); benchmark czasu rzeczywistego rozstrzyga kwestię.

### Gdzie AlphaEvolve pasuje w stosie technologicznym

| System | Generator | Ewaluator | Dziedzina | Przykładowe zwycięstwo |
|---|---|---|---|---|
| AlphaEvolve | Gemini | poprawność + benchmark | algorytmy, jądra, szeregowanie | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | poprawność | matematyka kombinatoryczna | dolne granice cap-set |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | krytyka LLM + eksperyment | badania ML | artykuł na warsztaty ICLR |
| Darwin Godel Machine (L4) | scaffolding agenta | SWE-bench / Polyglot | kod agenta | 20% → 50% SWE-bench |

Wszystkie cztery to wariacje na ten sam przepis: generator plus ewaluator, pętla. Różnice polegają na tym, co ewaluator ocenia i jak rygorystyczny jest.

## Użyj

`code/main.py` implementuje minimalną pętlę w stylu AlphaEvolve na zabawkowym problemie regresji symbolicznej. "LLM" to proxy z stdlib, które proponuje małe mutacje składniowe programu obliczającego funkcję docelową. "Ewaluator" mierzy średni błąd kwadratowy na wydzielonych punktach testowych.

Obserwuj:

- Jak najlepszy wynik poprawia się w kolejnych pokoleniach.
- Jak siatka MAP-elites utrzymuje różnorodne rozwiązania przy życiu, aby pętla nie zbiegła do lokalnego minimum.
- Jak usunięcie wydzielonego testu (ewaluator tylko na treningu) pozwala pętli spektakularnie się przetrenować.

## Dostarcz

`outputs/skill-evaluator-rigor-audit.md` jest warunkiem wstępnym do rozważenia pętli w stylu AlphaEvolve w nowej dziedzinie: czy Twój ewaluator faktycznie łapie awarie, na których Ci zależy?

## Ćwiczenia

1. Uruchom `code/main.py`. Zanotuj trajektorię najlepszego wyniku. Wyłącz wydzielony ewaluator (flaga `--no-holdout`) i uruchom ponownie. Określ ilościowo przetrenowanie.

2. Przeczytaj Sekcję 3 artykułu AlphaEvolve o siatce MAP-elites. Zaprojektuj deskryptor wektora cech dla nowego problemu (np. optymalizacyjne przebiegi kompilatora), który utrzymałby różnorodność przeszukiwania.

3. Wynik 48-mnożeń 4x4 poprawił granicę Strassena (49-mul) po 56 latach. Przeczytaj Dodatek F artykułu i wyjaśnij w trzech zdaniach, dlaczego ewaluator dla tego problemu jest szczególnie łatwy do zrobienia dobrze i dlaczego większość dziedzin nie jest taka.

4. Zaproponuj jedną dziedzinę, w której AlphaEvolve by zawiodło. Zidentyfikuj dokładnie, gdzie ewaluator się psuje i dlaczego.

5. Dla dziedziny, którą znasz, napisz sygnaturę ewaluatora, którego byś użył. Uwzględnij (a) warunki poprawności, (b) metrykę wydajności, (c) regułę generowania wydzielonych danych wejściowych, (d) co najmniej jedną kontrolę anty-hakowaniu nagrody.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|---|---|---|
| AlphaEvolve | "Ewolucyjny agent kodowania DeepMind" | Gemini + baza programów + maszynowo-sprawdzalny ewaluator |
| MAP-elites | "Archiwum zachowujące różnorodność" | Siatka kluczowana wektorami cech; każda komórka trzyma najlepszy wariant z tym deskryptorem |
| Model wyspowy | "Równoległe subpopulacje ewolucyjne" | Niezależne populacje, które okresowo migrują; zapobiega przedwczesnej zbieżności |
| Maszynowo-sprawdzalny ewaluator | "Deterministyczny oracle" | Test jednostkowy, symulator lub benchmark, którego LLM nie może sfałszować — warunek wstępny tej pętli |
| Hakowanie nagrody | "Optymalizacja miary, nie celu" | Pętla znajduje sposób na maksymalizację wyniku bez wykonywania zamierzonego zadania |
| Program początkowy | "Punkt startowy" | Początkowy poprawny, ale nieoptymalny program, z którego pętla ewoluuje |
| Wydzielony ewaluator | "Dane ewaluacyjne, których LLM nigdy nie widział" | Dane wejściowe generowane w czasie oceny, aby zapobiec zapamiętaniu |

## Dalsza lektura

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — pełny artykuł.
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — opis autora z wynikami.
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — odkryte algorytmy, w tym 48-mul 4x4 matmul.
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — system poprzedzający.
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — przedstawia autonomię ograniczoną ewaluatorem jako kluczowy kierunek badań.