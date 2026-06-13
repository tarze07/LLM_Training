# STaR, V-STaR, Quiet-STaR — Samouczone Rozumowanie

> Najmniejsza możliwa pętla samodoskonalenia siedzi wewnątrz uzasadnienia. Model generuje łańcuch myśli, zachowuje te, które trafiają na poprawne odpowiedzi, i dostraja się na nich. To jest STaR. V-STaR dodaje weryfikator, aby selekcja w czasie wnioskowania była lepsza. Quiet-STaR spycha uzasadnienie do każdego tokena. Wszystkie trzy działają. Żaden z nich nie jest magią — pętla zachowuje każdy skrót, który przypadkiem doprowadził do poprawnej odpowiedzi.

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time:** ~60 minutes

## Problem

Prostym sposobem nauczenia modelu rozumowania jest zbieranie ludzkich śladów rozumowania. To jest drogie, powolne i ograniczone tym, ile wysokiej jakości łańcuchów myśli ludzie są skłonni napisać.

STaR (Self-Taught Reasoner, Zelikman i in., 2022) pyta: co jeśli model sam pisze swoje uzasadnienia i ocenia je względem znanych odpowiedzi? Pętla wygląda następująco:

1. Próbkuj ślad rozumowania plus odpowiedź.
2. Jeśli ostateczna odpowiedź jest poprawna, zachowaj ślad.
3. Dostrój na zachowanych śladach.
4. Powtórz.

To działa. GSM8K i CommonsenseQA oba uległy poprawie bez nowej adnotacji ludzkiej. Ale pętla ma wbudowane obciążenie: każde uzasadnienie, które dało poprawną odpowiedź, jest zachowywane, niezależnie od tego, czy samo rozumowanie było poprawne. V-STaR (Hosseini i in., 2024) łata to za pomocą uczonego weryfikatora; Quiet-STaR (Zelikman i in., 2024) uogólnia pomysł na wewnętrzne uzasadnienia na poziomie każdego tokena.

## Koncepcja

### STaR: bootstrap na tym, co działało

Zacznij od modelu bazowego z pewną słabą zdolnością rozumowania. Dla każdego problemu treningowego próbkuj uzasadnienie plus odpowiedź. Jeśli odpowiedź zgadza się z etykietą, zachowaj trójkę (problem, uzasadnienie, odpowiedź). Dostrój model na zachowanym zbiorze. Powtórz.

Jeden szczegół ma znaczenie. Jeśli model nigdy nie może poprawnie rozwiązać problemu, pętla nie może się na nim uczyć. STaR dodaje **racjonalizację**: dla problemów, które model oblał, wstrzyknij poprawną odpowiedź jako podpowiedź i ponownie poproś model o wygenerowanie uzasadnienia, które do niej prowadzi. Zracjonalizowane uzasadnienia są dodawane do zbioru treningowego.

Wynik w oryginalnej publikacji (Zelikman i in., 2022): model bazowy GPT-J poprawił się na GSM8K z 5,8% do 10,7% poprzez powtarzane rundy STaR z racjonalizacją — około 5 punktów procentowych bezwzględnie. Na CommonsenseQA, STaR-wytrenowany GPT-J 6B osiągnął 72,5%, porównywalnie do dostrojonego GPT-3 175B (~73%) — modelu ~30 razy większego, trenowanego na ręcznie adnotowanych uzasadnieniach.

### V-STaR: trenuj weryfikator za pomocą DPO

STaR odrzuca niepoprawne uzasadnienia. Hosseini i in. (2024) zauważyli, że one również są danymi: każda para (uzasadnienie, "czy to poprawne") może trenować weryfikator. Używają Direct Preference Optimization na zarówno poprawnych, jak i niepoprawnych rozwiązaniach, aby zbudować rankinger. W czasie wnioskowania próbkuj N uzasadnień i wybierz najlepszy wybór weryfikatora.

Zgłoszona delta: +4 do +17 punktów procentowych ponad poprzednie baseline'y samodoskonalenia na GSM8K i MATH, przy czym większość zysku pochodzi z użycia weryfikatora do selekcji w czasie wnioskowania, a nie z dodatkowego dostrajania generatora.

### Quiet-STaR: wewnętrzne uzasadnienia na każdy token

Zelikman i in. (2024) zapytali: co jeśli model nauczy się generować krótkie wewnętrzne uzasadnienie na każdej pozycji tokena, nie tylko między problemem a odpowiedzią? Quiet-STaR trenuje model do emitowania ukrytej "myśli" przed każdym przewidywanym tokenem, a następnie miesza predykcję uwzględniającą myśl z baseline'ową predykcją za pomocą uczonej wagi.

Wynik: Mistral 7B zyskał absolutne ulepszenia zerowego strzału na GSM8K z 5,9% do 10,9% i na CommonsenseQA z 36,3% do 47,2% bez dostrajania specyficznego dla zadania. Model nauczył się "kiedy myśleć" — trudne tokeny dostają dłuższe wewnętrzne uzasadnienia; łatwe prawie żadne.

### Dlaczego wszystkie trzy dzielą problem bezpieczeństwa

Wszystkie trzy metody używają ostatecznej odpowiedzi jako sygnału gradientu. Uzasadnienie, które osiąga poprawną odpowiedź poprzez błędne rozumowanie — wykorzystując skrót, zgadując lub używając nieuogólniającego wzorca — dostaje pozytywne wzmocnienie. Na problemach wewnątrz dystrybucji skrót działa. Na problemach poza dystrybucją psuje się po cichu.

Weryfikator V-STaR łagodzi to, ucząc się rankingu uzasadnień, ale weryfikator jest trenowany na tym samym zbiorze etykiet. Może nauczyć się preferować ładnie sformatowane błędne rozumowanie nad uczciwą niepewnością. Bezpieczniejszym projektem jest łączenie danych w stylu STaR z (a) modelami nagrody nadzorowanymi procesowo (nagradzającymi pośrednie kroki, nie tylko odpowiedzi) oraz (b) wydzieloną oceną OOD, która łamie proste skróty.

### Porównanie

| Metoda | Sygnał treningowy | Koszt wnioskowania | Strata danych | Znany tryb awarii |
|---|---|---|---|---|
| STaR | zachowaj (uzasadnienie, odpowiedź) jeśli poprawne | 1x | odrzuca wszystkie niepoprawne uzasadnienia | skrótowe uzasadnienia |
| STaR + racjonalizacja | jak wyżej + ponowne próby z podpowiedzią poprawnej odpowiedzi | 1x | mniej | zracjonalizowane uzasadnienia mogą być nieprawdopodobne |
| V-STaR | STaR + weryfikator DPO z obu klas | Nx (best-of-N) | minimalne | weryfikator może wzmacniać pewne błędne przekonania |
| Quiet-STaR | uzasadnienie na token + waga mieszania | 1,5-3x | minimalne | wciąż gradient warunkowany odpowiedzią |

### Gdzie to siedzi w stosie technologicznym 2026

STaR jest stary. Ale wzór pojawia się wszędzie w latach 2025-2026. RL na weryfikowalnych problemach matematycznych (DeepSeek-R1, Kimi-k1.5, o1) to sygnał gradientu warunkowany odpowiedzią STaR, w skali. Modele nagrody procesowej (Lightman i in., 2023; OpenAI "Let's verify step by step") to alternatywa nadzorowana procesowo. AlphaEvolve (Lekcja 3) to STaR dla kodu, z ewaluatorem programu zamiast etykiety. Darwin Godel Machine (Lekcja 4) to STaR dla samego scaffolding agenta.

Zrozumienie STaR sprawia, że wszystkie one stają się jasne. To minimalnie wykonalna pętla samodoskonalenia.

```figure
reflection-loop
```

## Użyj

`code/main.py` uruchamia symulowaną pętlę STaR na zabawkowym zadaniu arytmetycznym. Możesz obserwować:

- Jak dokładność rośnie w kolejnych rundach bootstrapu.
- Jak skróty się wkradają: symulator zawiera "leniwą" klasę uzasadnień, która dostaje poprawną odpowiedź w 40% przypadków, ale generalizuje źle. Obserwuj, czy STaR je zachowuje.
- Jak weryfikator (w stylu V-STaR) pomaga na wnioskowaniu, ale nie może w pełni wyeliminować skrótów wprowadzonych podczas treningu.

## Dostarcz

`outputs/skill-star-loop-reviewer.md` pomaga audytować proponowany pipeline samouczonego rozumowania, zanim zaczniesz na nim trenować.

## Ćwiczenia

1. Uruchom symulator. Ustaw częstotliwość skrótów na zero, a następnie na 0,4. Jak bardzo różni się końcowa dokładność między tymi dwoma uruchomieniami, mimo że oba osiągają >90% na dystrybucji treningowej?

2. Dodaj do symulatora wydzielony test OOD. Losuj problemy z innej dystrybucji i oceń model bootstrapowany zarówno na zbiorze wewnątrz dystrybucji, jak i OOD. Określ ilościowo lukę.

3. Przeczytaj artykuł Quiet-STaR (arXiv:2403.09629) Sekcja 3. Wyjaśnij token "end-of-thought" i głowę wagi mieszania w trzech zdaniach każde.

4. Porównaj filtr STaR "zachowaj jeśli poprawne" z alternatywą nadzorowaną procesowo, która nagradza każdy krok uzasadnienia niezależnie. Zidentyfikuj różnicę w koszcie etykietowania i prawdopodobną różnicę w jakości.

5. Zaprojektuj jedną ewaluację, która złapałaby skrótowe uzasadnienia we wdrożonym modelu. Nie musi być doskonała — musi łamać najprostsze skróty, które pętla STaR by wzmocniła.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|---|---|---|
| STaR | "Self-Taught Reasoner" | Dostrajaj na uzasadnieniach wygenerowanych przez model, które trafiają poprawne odpowiedzi; powtórz |
| Racjonalizacja | "Ponowna próba z podpowiedzią" | Wstrzyknij poprawną odpowiedź i poproś ponownie o uzasadnienie dla problemów, które model bazowy oblał |
| V-STaR | "Verifier STaR" | DPO-trenuj weryfikator na zarówno poprawnych, jak i niepoprawnych uzasadnieniach; użyj do selekcji w czasie wnioskowania |
| Quiet-STaR | "Uzasadnienia na token" | Generuj ukryte myśli na każdej pozycji tokena; mieszaj z baseline'ową predykcją |
| Gradient warunkowany odpowiedzią | "Sygnał oparty na wyniku" | Pętla treningowa nagradza ostateczne odpowiedzi, nie kroki rozumowania |
| Model nagrody procesowej | "Weryfikator na poziomie kroku" | Model nagrody trenowany na poprawności każdego kroku, nie na wyniku — kontrastuje z STaR |
| Skrótowe uzasadnienie | "Poprawna odpowiedź, błędne rozumowanie" | Uzasadnienie, które osiąga etykietę przez nieuogólniający wzorzec; STaR je zachowuje |

## Dalsza lektura

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — oryginalna publikacja.
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — dodaje weryfikator DPO do selekcji w czasie wnioskowania.
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — wewnętrzne uzasadnienia na token.
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — modele nagrody procesowej, alternatywny sygnał gradientu.
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — RL na weryfikowalnych zadaniach, STaR w skali trenowania modeli granicznych.