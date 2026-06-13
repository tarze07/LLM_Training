# Człowiek w Pętli: Proponuj, a Potem Zatwierdź

> Konsensus z 2026 roku w sprawie HITL jest konkretny. To nie jest "agent pyta, użytkownik klika Zatwierdź". To proponuj, a potem zatwierdź: proponowana akcja jest utrwalana w trwałym magazynie z kluczem idempotentności; prezentowana recenzentowi z intencją, pochodzeniem danych, dotkniętymi uprawnieniami, promieniem rażenia i planem wycofania; zatwierdzana dopiero po pozytywnym potwierdzeniu; weryfikowana po wykonaniu, aby potwierdzić, że efekt uboczny rzeczywiście wystąpił. LangGraph `interrupt()` z punktami kontrolnymi PostgreSQL, Microsoft Agent Framework `RequestInfoEvent` i Cloudflare `waitForApproval()` wszystkie implementują ten sam kształt. Kanonicznym trybem awarii jest zatwierdzenie na oślep: przycisk "Zatwierdzić?" jest klikany bez przeglądu. Udokumentowanym środkiem zaradczym jest wyzwanie-i-odpowiedź z wyraźną listą kontrolną.

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 minutes

## Problem

Agent wykonuje akcję. Użytkownik musi zdecydować: zatwierdzić czy nie. Jeśli decyzja jest błyskawiczna, prawdopodobnie nie jest to przegląd. Jeśli decyzja jest ustrukturyzowana, jest powolna, ale godna zaufania. Pytanie inżynieryjne brzmi: jak sprawić, by ustrukturyzowany przegląd był ścieżką najmniejszego oporu.

Wzorzec HITL z epoki 2023 to synchroniczny prompt: "Agent chce wysłać e-mail do X z treścią Y — zatwierdzić?" Użytkownik klika Zatwierdź. Wszyscy czują, że system jest bezpieczny. W praktyce ta powierzchnia jest mocno klikana na oślep: użytkownicy zatwierdzają szybko, zatwierdzenia niewiele przewidują, a gdy agent popełni błąd, ślad audytowy pokazuje długą historię zatwierdzeń, których użytkownik nie pamięta.

Wzorzec z 2026 roku — proponuj, a potem zatwierdź — przenosi HITL na trwały substrat, dołącza ustrukturyzowane metadane i wymaga pozytywnego zatwierdzenia. Każdy zarządzany SDK agenta dostarcza wersję: LangGraph `interrupt()`, Microsoft Agent Framework `RequestInfoEvent`, Cloudflare `waitForApproval()`. Nazwy API różnią się; kształt nie.

## Koncepcja

### Maszyna stanów proponuj, a potem zatwierdź

1. **Zaproponuj.** Agent tworzy proponowaną akcję. Utrwalana w trwałym magazynie (PostgreSQL, Redis, Durable Object). Zawiera:
   - intencję (dlaczego agent to robi)
   - pochodzenie danych (jakie źródło doprowadziło do tej propozycji)
   - dotknięte uprawnienia (które zakresy / pliki / endpointy)
   - promień rażenia (jaki jest najgorszy przypadek)
   - plan wycofania (jeśli zatwierdzone, jak to cofamy)
   - klucz idempotentności (unikalny na propozycję; ponowne przesłanie zwraca ten sam rekord)
2. **Przedstaw.** Recenzent widzi propozycję ze wszystkimi metadanymi. Recenzent to osoba (nie agent recenzujący siebie).
3. **Zatwierdź.** Pozytywne potwierdzenie. Akcja jest wykonywana.
4. **Zweryfikuj.** Po wykonaniu efekt uboczny jest odczytywany i potwierdzany. Jeśli krok weryfikacji się nie powiedzie, system znajduje się w znanym złym stanie i włącza się alert.

### Klucz idempotentności

Bez klucza idempotentności ponowienie po przejściowym błędzie może podwójnie wykonać zatwierdzoną akcję. Konkretny przykład: użytkownik zatwierdza "prześlij 100 USD z A do B." Sieć się zacina. Przepływ pracy jest ponawiany. Użytkownik zatwierdził raz, ale przelew wykonuje się dwa razy. Klucz idempotentności wiąże zatwierdzenie z pojedynczym, unikalnym efektem ubocznym; drugie wykonanie jest bezoperacyjne.

To ten sam wzorzec idempotentności, którego używają API Stripe i AWS. Jego ponowne użycie dla zatwierdzeń agenta jest wyraźnie opisane w dokumentacji Microsoft Agent Framework.

### Trwałość: dlaczego zatwierdzenia przeżywają procesy

Poczekalnia zatwierdzeń to kawałek stanu, którego agent nie posiada. Przepływ pracy jest wstrzymany (Lekcja 12). Gdy nadejdzie zatwierdzenie, przepływ pracy wznawia się dokładnie od tego punktu. Dlatego LangGraph łączy `interrupt()` z punktami kontrolnymi PostgreSQL, a nie tylko stanem w pamięci — zatwierdzenie dwa dni później wciąż znajduje przepływ pracy nienaruszonym.

### Zatwierdzenia na oślep i środek zaradczy wyzwania-i-odpowiedzi

Domyślny interfejs HITL (przyciski "Zatwierdź" / "Odrzuć") generuje szybkie zatwierdzenia bez rzeczywistego przeglądu. Udokumentowany środek zaradczy: lista kontrolna wyzwania-i-odpowiedzi, która wymaga pozytywnych odpowiedzi na konkretne pytania, zanim przycisk Zatwierdź zostanie włączony. Konkretny kształt:

- "Czy rozumiesz, jakiego zasobu to dotyczy? [ ]"
- "Czy zweryfikowałeś, że promień rażenia jest akceptowalny? [ ]"
- "Czy masz plan wycofania, jeśli to się nie powiedzie? [ ]"

To nie biurokracja dla samej biurokracji — to funkcja wymuszająca. Recenzent, który nie może zaznaczyć pól, albo prosi o wyjaśnienie (eskalacja), albo odrzuca (bezpieczna domyślna). Badania Anthropic nad bezpieczeństwem agentów wyraźnie cytują HITL oparty na liście kontrolnej jako środek zaradczy dla wzorców zatwierdzeń na oślep.

### Co kwalifikuje się jako znaczące

Nie każda akcja potrzebuje mechanizmu proponuj, a potem zatwierdź. Wytyczne z 2026 roku:

- **Akcje znaczące** (zawsze HITL): nieodwracalne zapisy, transakcje finansowe, komunikacja wychodząca, zmiany w produkcyjnej bazie danych, destrukcyjne operacje na systemie plików.
- **Akcje odwracalne** (czasami HITL): edycje lokalnych plików, zmiany w środowisku stagingowym, odwracalne zapisy z jasnym planem wycofania.
- **Odczyty i inspekcje** (nigdy HITL): czytanie pliku, listowanie zasobów, wywoływanie API tylko do odczytu.

### Weryfikacja po akcji

"Zatwierdzenie zostało wykonane" to nie to samo co "efekt uboczny wystąpił." Partycjonowanie sieci i warunki wyścigu mogą wyprodukować przepływ pracy, który myśli, że się udał, podczas gdy backend nie utrwalił. Krok weryfikacji ponownie odczytuje docelowy zasób po zatwierdzeniu, aby potwierdzić. To ten sam wzorzec co transakcje baz danych z klauzulami `RETURNING` lub AWS `GetObject` po `PutObject`.

### EU AI Act Artykuł 14

Artykuł 14 nakazuje skuteczny nadzór człowieka dla systemów AI wysokiego ryzyka w UE. "Skuteczny" nie jest dekoracyjny. Język regulacyjny wyraźnie wyklucza wzorce zatwierdzeń na oślep. Proponuj, a potem zatwierdź z wyzwaniem-i-odpowiedzią to kształt, który przechodzi kontrolę Artykułu 14 w dokumentach zgodności Microsoft Agent Governance Toolkit.

## Użyj Tego

`code/main.py` implementuje maszynę stanów proponuj, a potem zatwierdź w stdlib Python. Trwałym magazynem jest plik JSON. Klucz idempotentności to hash z (thread_id, action_signature). Sterownik symuluje trzy przypadki: czysty przepływ zatwierdzenia, ponowienie po przejściowym błędzie (które nie może podwójnie wykonać) oraz domyślne zatwierdzenie na oślep w porównaniu z przepływem wyzwania-i-odpowiedzi.

## Dostarcz To

`outputs/skill-hitl-design.md` przegląda proponowany przepływ pracy HITL pod kątem kształtu proponuj, a potem zatwierdź i wskazuje brakujące metadane, idempotentność, weryfikację lub warstwy wyzwania-i-odpowiedzi.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że ponowienie zatwierdzonej propozycji używa trwałego rekordu i nie wykonuje się ponownie. Teraz zmień klucz idempotentności, aby zawierał znacznik czasu i pokaż, że ponowienie wykonuje się podwójnie.

2. Rozszerz rekord propozycji o pole `rollback`. Zasymuluj wykonanie, którego krok weryfikacji się nie powiódł. Pokaż, że wycofanie zadziała automatycznie.

3. Przeczytaj dokumentację Microsoft Agent Framework `RequestInfoEvent`. Zidentyfikuj jedno pole metadanych, które API zawiera, a którego brakuje w zabawkowym silniku. Dodaj je i wyjaśnij, przed czym chroni.

4. Zaprojektuj listę kontrolną wyzwania-i-odpowiedzi dla konkretnej akcji (np. "opublikuj na publicznym koncie Twitter"). Jakie trzy pytania musi odpowiedzieć recenzent? Dlaczego właśnie te trzy?

5. Wybierz jeden przypadek, gdzie synchroniczny prompt "Zatwierdzić?" byłby wystarczający (bez potrzeby trwałego magazynu). Wyjaśnij dlaczego i nazwij klasę ryzyka, którą akceptujesz.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|---|---|---|
| Proponuj, a potem zatwierdź | "Dwufazowe zatwierdzenie" | Utrwalona propozycja + pozytywne zatwierdzenie + weryfikacja |
| Klucz idempotentności | "Bezpieczny token ponawiania" | Unikalny na propozycję; drugie wykonanie jest bezoperacyjne |
| Pochodzenie danych | "Skąd to przyszło" | Konkretna treść źródłowa, która doprowadziła do propozycji |
| Promień rażenia | "Najgorszy przypadek" | Zakres efektu, jeśli akcja pójdzie źle |
| Zatwierdzenie na oślep | "Szybkie zatwierdzenie" | "Zatwierdź" kliknięte bez rzeczywistego przeglądu |
| Wyzwanie-i-odpowiedź | "Wymuszająca lista kontrolna" | Recenzent musi pozytywnie potwierdzić konkretne pytania |
| RequestInfoEvent | "Prymityw MS Agent Framework" | Trwałe żądanie HITL z ustrukturyzowanymi metadanymi |
| `interrupt()` / `waitForApproval()` | "Prymitywy frameworków" | LangGraph / Cloudflare odpowiedniki tego samego kształtu |

## Dalsza Lektura

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, trwałe zatwierdzenia.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` i Durable Objects.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL jako środek zaradczy dla ryzyka długoterminowego.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulacyjna linia bazowa dla systemów wysokiego ryzyka.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — ramy konstytucyjne wokół nadzoru.