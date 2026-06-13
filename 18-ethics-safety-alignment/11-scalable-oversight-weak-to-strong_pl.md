# Skalowalny Nadzór i Generalizacja Od Słabego Do Silnego

> Burns et al. (OpenAI Superalignment, „Weak-to-Strong Generalization", 2023) zaproponowali proxy dla problemu superalignmentu: dostrój silny model przy użyciu etykiet wyprodukowanych przez słabszy model. Jeśli silny model generalizuje poprawnie z niedoskonałego słabego nadzoru, obecne metody alignmentu w skali ludzkiej mogą rozciągnąć się na superludzkie systemy. Skalowalny nadzór i W2SG są uzupełniające. Skalowalny nadzór (debata, rekurencyjne modelowanie nagrody, dekompozycja zadań) zwiększa efektywne capability nadzorcy, aby mógł nadążyć za modelem pod nadzorem. W2SG zapewnia, że silny model generalizuje poprawnie z jakiegokolwiek niedoskonałego nadzoru dostarczonego przez nadzorcę. Debata Pomaga W2SG (arXiv:2501.13124, styczeń 2025) łączy je.

**Type:** Learn
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time:** ~60 minutes

## Learning Objectives

- Zdefiniuj skalowalny nadzór i generalizację od słabego do silnego oraz wyjaśnij, jak się uzupełniają.
- Opisz konfigurację eksperymentalną Burns et al. 2023: dostrój GPT-4 przy użyciu etykiet z GPT-2.
- Wyjaśnij metrykę odzysku luki wydajnościowej (PGR) i co mierzy.
- Wymień trzy główne mechanizmy skalowalnego nadzoru (debata, rekurencyjne modelowanie nagrody, dekompozycja zadań) i jedną zaletę każdego.

## The Problem

Każda technika alignmentu w Fazie 18 do tej pory zakłada, że nadzorca może ocenić zachowanie modelu. Gdy model jest superludzki, nadzorca jest słabym ogniwem. Pytanie superalignmentu: czy słabszy nadzorca może niezawodnie wyprodukować silniejszy, wyrównany model?

Burns et al. redukują to do operacjonalizowanej konfiguracji empirycznej: nadzoruj silnego słabym, zmierz, ile capability silnego modelu przetrwa słaby nadzór. To nie jest rozwiązanie superalignmentu — to proxy, które pozwala nam mierzyć postęp.

## The Concept

### W2SG: the Burns et al. setup

- Słaby model: klasy GPT-2.
- Silny model: klasy GPT-4.
- Cel: sufitowy wynik silnego GPT-4 w zadaniu.

Procedura:
1. Pobierz przewidywania zero-shot słabego modelu w zadaniu.
2. Dostrój silny model na danych oznaczonych przez słaby.
3. Zmierz dokładność dostrojonego silnego modelu.

Linie bazowe: własna dokładność słabego modelu, sufitowy wynik silnego modelu na złotych etykietach.

Metryka luki: Performance Gap Recovered (PGR) = (dostrojony - słaby) / (sufit - słaby). PGR 1,0 oznacza, że słaby nadzór całkowicie zamyka lukę; PGR 0 oznacza, że słaby nadzór nie pomógł.

### Burns et al. empirical findings

W zadaniach NLP, szachowych zagadkach i modelowania nagrody PGR był konsekwentnie dodatni: silny model generalizował poza błędami słabego nadzorcy. PGR specyficzny dla zadania wahał się od ~20% do ~80%. Silny model „rozumie" zamierzone zadanie z niedoskonałych etykiet, uzupełniając swoimi wstępnie wytrenowanymi priorytetami.

Ograniczenia zauważone przez Burns et al.:
- Luka słaby-silny dotyczy tu capability, a nie alignmentu. Słaby model jest wyrównany; prawdziwie superludzki model może być wyrównany przez człowieka, ale z dużo większą luką capability.
- Generalizacja silnego modelu może dotyczyć bardziej priorytetów zadaniowych niż odzyskiwania prawdziwego zamiaru.
- Wydobycie latentnej wiedzy silnego modelu to prawdziwy problem; PGR mierzy konkretną operacjonalizację.

### Scalable oversight: three mechanisms

- **Debata (Irving et al. 2018).** Dwie instancje U argumentują przeciwne strony; słaby sędzia decyduje. Założenie: znalezienie przekonującej prawdziwej odpowiedzi jest łatwiejsze niż znalezienie przekonującej fałszywej odpowiedzi. Niedawna praca z 2024-2025 (Khan et al., Michael et al.) pokazuje, że debata czasami pomaga, a czasami nie, i czy pomaga, zależy od struktury zadania.
- **Rekurencyjne Modelowanie Nagrody (Leike et al. 2018).** U pomaga człowiekowi trenować model nagrody dla U+1. Efektywne capability nadzorcy rośnie wraz z capability modelu.
- **Dekompozycja Zadań (Christiano, Shlegeris, Amodei 2018).** Rozłóż trudne zadanie na podzadania, które człowiek może sprawdzić, rekurencyjnie. Zakłada dekomponowalność.

Każdy mechanizm zakłada coś o strukturze zadania lub alignmentie pośrednich komponentów.

### Why scalable oversight and W2SG are complementary

Skalowalny nadzór zwiększa efektywną jakość sygnału nadzorcy.
W2SG zamyka lukę z jakiegokolwiek niedoskonałego sygnału, który nadzorca może dostarczyć.

Lang et al. — Debata Pomaga Generalizacji Od Słabego Do Silnego (arXiv:2501.13124) łączy je: protokół debaty dostarcza lepsze słabe etykiety, a silny model jest trenowany na tych etykietach. Zgłoszone zyski PGR w zadaniach NLP.

### The organizational drama

Zespół Superalignmentu OpenAI został rozwiązany w maju 2024 po odejściu Jana Leike do Anthropic. Program (skalowalny nadzór, W2SG, zautomatyzowane badania alignmentu) kontynuował się w Anthropic i w laboratoriach akademickich — MATS (Lekcja 28), Redwood (Lekcja 10), Apollo (Lekcja 8), METR (Lekcja 28). Struktura organizacyjna się zmieniła; pytania badawcze nie.

### Where this fits in Phase 18

Lekcje 6-10 opisują zagrożenie i paradygmat obronny przy założeniu, że U jest niewiarygodny. Lekcja 11 to paradygmat ofensywny: uczyń nadzorcę wystarczająco silnym, aby zweryfikować alignment U. Lekcje 12-16 przechodzą następnie do praktycznych narzędzi ewaluacji kontradyktoryjnej.

## Use It

`code/main.py` symuluje dostrajanie W2SG na syntetycznym zadaniu. Słaby etykieter ma 70% dokładności z ustrukturyzowanymi błędami; silny model ma 95% sufitowej dokładności na złotych etykietach. Dostrajasz silny model na słabych etykietach, mierzysz PGR i porównujesz do silny-na-złocie i słaby-sam.

## Ship It

Ta lekcja produkuje `outputs/skill-w2sg-pgr.md`. Otrzymując opis konfiguracji nadzoru, identyfikuje słabego nadzorcę, silny model, jakość nadzoru i oblicza (lub żąda) PGR. Flaguje, czy twierdzenie brzmi „słaby może nadzorować silnego" czy „słaby + mechanizm nadzoru może nadzorować silnego."

## Exercises

1. Uruchom `code/main.py`. Zgłoś PGR dla weak_accuracy = 0,60, 0,70, 0,80. Wyjaśnij kształt krzywej PGR.

2. Zmodyfikuj słabego etykietera, aby miał ustrukturyzowany błąd (np. zawsze błędny na konkretnej klasie wejść). Czy PGR wzrasta, maleje, czy pozostaje taki sam? Wyjaśnij.

3. Przeczytaj Burns et al. 2023 Sekcję 4.3 (zadania NLP). Odtwórz intuicję „pomocniczej straty ufności": gdy silny model jest bardziej pewny niż słabe etykiety, kto wygrywa?

4. Zaprojektuj protokół skalowalnego nadzoru, który łączy debatę i dekompozycję zadań dla zadania inżynierii oprogramowania. Wymień jeden tryb awarii każdego komponentu i wyjaśnij, jak kombinacja rozwiązuje lub nie rozwiązuje każdy.

5. Przedstaw, co sfalsyfikowałoby twierdzenie „generalizacja od słabego do silnego jest realną ścieżką do superalignmentu." Bądź konkretny co do empirycznej sygnatury, którą musiałbyś zobaczyć.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Scalable oversight | „uczynienie nadzorcy silniejszym" | Mechanizmy zwiększające zdolność nadzorcy do oceny bardziej zdolnego modelu |
| W2SG | „słaby nadzoruje silnego" | Dostrajanie silnego modelu na słabych etykietach i mierzenie odzyskanego capability |
| PGR | „odzysk luki wydajnościowej" | (dostrojony - słaby) / (sufit - słaby); 1,0 = całkowicie zamknięta, 0 = brak pomocy |
| Debate | „dwie instancje U argumentują" | Mechanizm skalowalnego nadzoru, w którym słaby sędzia wybiera między dwoma obrońcami U |
| RRM | „rekurencyjne modelowanie nagrody" | U pomaga trenować model nagrody dla U+1; capability nadzorcy śledzi U |
| Task decomposition | „podzadania, które człowiek sprawdza" | Podziel trudne zadanie na podzadania, które człowiek może zweryfikować, rekurencyjnie |
| Superalignment | „alignment superludzkiej SI" | Program badawczy dotyczący alignmentu modeli, których człowiek nie może bezpośrednio ocenić |

## Further Reading

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — artykuł W2SG
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — mechanizm debaty
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — rekurencyjne modelowanie nagrody
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — badanie empiryczne debaty z silniejszymi debatującymi z 2024 roku
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — kombinacja debaty + W2SG z 2025 roku