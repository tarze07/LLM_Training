# Capstone 17 — Osobisty Korepetytor AI (Adaptacyjny, Multimodalny, z Pamięcią)

> Khanmigo (Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat i Synthesis Tutor wszystkie dostarczyły adaptacyjne multimodalne korepetycje na skalę w 2026 roku. Wspólny kształt to sokratejska polityka (nigdy nie wyrzucaj odpowiedzi), model ucznia aktualizowany po każdej interakcji (w stylu bayesowskiego śledzenia wiedzy), wejście głosowe + tekstowe + foto-math, wyszukiwanie w grafie programu nauczania, harmonogram powtórek w odstępach i twarde filtry bezpieczeństwa dla treści odpowiednich do wieku. Capstone polega na dostarczeniu korepetytora specyficznego dla przedmiotu (algebra K-12 lub wstęp do Pythona), przeprowadzeniu dwutygodniowego badania skuteczności z 10 uczniami i przejściu audytu bezpieczeństwa treści.

**Type:** Capstone
**Languages:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Prerequisites:** Phase 5 (NLP), Phase 6 (speech), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30 hours

## Problem

Adaptacyjne korepetycje były kiedyś niszą badawczą ed-tech. Do 2026 roku stały się produktem konsumenckim. Khanmigo jest wdrożone w większości amerykańskich okręgów szkolnych. Duolingo Max osiągnęło dziesiątki milionów MAU. Google LearnLM / Gemini for Education zasila korepetycje w Google Classroom. Quizlet Q-Chat siedzi obok fiszek. Synthesis Tutor osiągnęło wiralowość jako korepetytor dla ciekawskich dzieci. Wspólne elementy: multimodalne wejście (pisz, mów, fotografuj równania), sokratejska pedagogika (najpierw zapytaj, wyjaśnij później), model ucznia aktualizowany po każdej interakcji i ścisłe bezpieczeństwo odpowiednie do wieku.

Zbudujesz jeden z tych systemów dla konkretnej grupy. Miara pomiarowa to rzeczywiste badanie skuteczności: wyniki pre-testu i post-testu przez dwa tygodnie z 10 uczniami. Pętla głosowa musi być naturalna (pod-stos capstone 03). Pamięć musi szanować prywatność. Filtr bezpieczeństwa musi przejść red-team świadomy COPPA dla K-12.

## Koncepcja

Cztery komponenty. **Polityka korepetytora** to sokratejska pętla: gdy uczeń prosi o odpowiedź, polityka zadaje pytanie naprowadzające; gdy odpowie poprawnie, przechodzi do następnego konceptu; gdy utknie, oferuje rusztowanie podpowiedzi. **Model ucznia** to bayesowskie śledzenie wiedzy (lub prosty wariant), które aktualizuje prawdopodobieństwo opanowania na węzeł programu nauczania po każdej interakcji. **Graf programu nauczania** to Neo4j konceptów z krawędziami wymagań wstępnych; polityka przechodzi graf, aby wybrać następny koncept. **Pamięć** to magazyn epizodyczny + semantyczny (w stylu agentmemory) przechowujący przeszłe interakcje, błędy i preferencje.

UX jest multimodalny. Wejście tekstowe dla pisemnych odpowiedzi. Wejście głosowe przez LiveKit + Whisper (wykorzystaj ponownie capstone 03). Wejście zdjęciowe dla problemów matematycznych przez dots.ocr lub PaliGemma 2. Wyjście głosowe przez Cartesia Sonic-2. Bezpieczeństwo używa Llama Guard 4 plus filtra odpowiedniego do wieku (blokuje treści dla dorosłych, przemoc, samookaleczenie) i polityki przechowywania pamięci zgodnej z COPPA.

Badanie skuteczności jest rezultatem. 10 uczniów, pre-test i post-test, dwa tygodnie. Raportuj przyrost uczenia się i przedział ufności. Porównaj z nieadaptacyjnym baseline (ta sama treść dostarczona liniowo bez polityki korepetytora).

## Architektura

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## Stack

- Wybór przedmiotu: algebra K-12 lub wstęp do Pythona (wybierz jeden dla głębi)
- Polityka korepetytora: LangGraph na Claude Sonnet 4.7 (z prompt caching)
- Model ucznia: Bayesian knowledge tracing (klasyczny) lub FSRS do odstępów
- Graf programu nauczania: Neo4j konceptów + krawędzi wymagań wstępnych + treści OER
- Pamięć: trwały magazyn wektorowy + epizodyczny + semantyczny w stylu agentmemory
- Głos: LiveKit Agents 1.0 + Cartesia Sonic-2 (wykorzystaj ponownie pod-stos capstone 03)
- Foto-math: dots.ocr lub PaliGemma 2 do rozpoznawania równań
- Bezpieczeństwo: Llama Guard 4 + niestandardowy filtr odpowiedni do wieku
- Ewaluacja: generowanie pytań na poziomie Blooma, harness pre/post test, narzędzia do badania skuteczności

## Build It

1. **Graf programu nauczania.** Zbuduj Neo4j z 50-150 węzłami konceptów (np. algebra K-12 od "osi liczbowej" do "wzoru kwadratowego") z krawędziami wymagań wstępnych. Dołącz treści OER na węzeł (Open Textbook, OpenStax).

2. **Model ucznia.** Zainicjuj Bayesian knowledge tracing z priorami: zgadnij, poślizgnij się, tempo uczenia. Aktualizuj opanowanie na koncept po każdej interakcji. Zapisz na ucznia.

3. **Polityka korepetytora.** LangGraph z węzłami: `read_signal` (czy odpowiedź ucznia była poprawna / częściowa / utknięta?), `select_concept` (przejdź graf programu nauczania, wybierając koncept o najwyższym priorytecie), `scaffold` (sokratejski prompt), `update_mastery`.

4. **Pamięć.** Każda interakcja zapisuje się do magazynu epizodycznego. Błędy i preferencje promują do pamięci semantycznej. Polityka przechowywania zgodna z COPPA: auto-usunięcie po 1 roku, dostępne dla rodzica.

5. **Ścieżka głosowa.** Worker LiveKit Agents podłączony do polityki korepetytora. ASR przez Whisper-v3-turbo. TTS przez Cartesia Sonic-2. Obsługa wtargnięcia (wykorzystaj ponownie mechanikę capstone 03).

6. **Ścieżka foto-math.** Prześlij lub zrób zdjęcie; uruchom dots.ocr lub PaliGemma 2, aby rozpoznać równanie; przekaż do korepetytora jako strukturalne wejście.

7. **Bezpieczeństwo.** Każde wyjście modelu przechodzi przez Llama Guard 4 + filtr odpowiedni do wieku (blokuje samookaleczenie, treści dla dorosłych, przemoc). Dostęp do pamięci zakresowany identyfikatorem ucznia; powierzchnia dostępu dla rodzica do usuwania.

8. **Badanie skuteczności.** 10 uczniów, pre-test (standardowy 30-pytaniowy baseline), dwa tygodnie interakcji z korepetytorem (3 sesje/tydzień), post-test. Porównaj z nieadaptacyjną grupą baseline 10 uczniów na tej samej treści.

9. **Tygodniowe raporty postępu.** Na ucznia, automatycznie generuj podsumowanie PDF tematów zbadanych, trajektorii opanowania i zalecanych następnych kroków.

## Use It

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## Ship It

`outputs/skill-ai-tutor.md` jest rezultatem. Adaptacyjny korepetytor specyficzny dla przedmiotu z multimodalnym wejściem, modelem ucznia, pamięcią, bezpieczeństwem i zmierzoną skutecznością.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Różnica przyrostu uczenia | Różnica pre/post-test w 10-uczniowym dwutygodniowym badaniu |
| 20 | Wierność sokratejska | Punktacja rubrykowa na próbkach transkryptów |
| 20 | UX multimodalny | Spójność głos + zdjęcie + tekst od końca do końca |
| 20 | Postawa bezpieczeństwa + prywatności | Wskaźnik przepustowości Llama Guard 4 + przechowywanie zgodne z COPPA |
| 15 | Szerokość programu nauczania i jakość grafu | Pokrycie konceptów + spójność grafu wymagań wstępnych |
| **100** | | |

## Ćwiczenia

1. Przeprowadź badanie skuteczności z i bez adaptacyjnego modelu ucznia (losowa kolejność konceptów). Raportuj różnicę. Spodziewaj się, że adaptacyjny wygra, ale rozmiar jest interesującą liczbą.

2. Dodaj multimodalną próbkę: to samo pytanie koncepcyjne dostarczone jako tekst, głos i zdjęcie. Zmierz, czy uczniowie szybciej się zbiegają z preferowaną modalnością.

3. Zbuduj kokpit rodzica: tematy przećwiczone, trajektorie opanowania, nadchodzące koncepty, zdarzenia bezpieczeństwa (trafienia guardraili). Zgodne z COPPA.

4. Dodaj tryb zmiany języka: korepetytor akceptuje wejście po hiszpańsku i uczy po hiszpańsku. Zmierz pokrycie X-Guard.

5. Przetestuj prywatność pamięci: zweryfikuj, że uczeń A nie może zobaczyć danych ucznia B nawet przez atak ponownego wstrzyknięcia klipu głosowego. Zaloguj próbę dostępu i alarmuj.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Socratic policy | "Pytaj, nie wyrzucaj" | Korepetytor zadaje pytanie naprowadzające zamiast podawać odpowiedź |
| Bayesian knowledge tracing | "BKT" | Klasyczne równania modelu ucznia dla prawdopodobieństwa opanowania na koncept |
| FSRS | "Free Spaced Repetition Scheduler" | Harmonogram powtórek w odstępach 2024, lepszy niż SM-2 |
| Curriculum graph | "DAG konceptów" | Neo4j konceptów z krawędziami wymagań wstępnych |
| Episodic memory | "Log na interakcję" | Każda interakcja przechowywana do późniejszego odzyskania |
| Semantic memory | "Magazyn wyuczonych wzorców" | Skompaktowane błędy i preferencje promowane z epizodycznych |
| COPPA | "Ustawa o prywatności dzieci" | Amerykańskie prawo ograniczające zbieranie danych od dzieci poniżej 13 lat |

## Dalsza lektura

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — reference consumer K-12 tutor
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — reference language-learning tutor
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — hosted reference model
- [Quizlet Q-Chat](https://quizlet.com) — alternate reference
- [Synthesis Tutor](https://www.synthesis.com) — startup reference
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — spaced-repetition scheduler
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — learner-model classic
- [LiveKit Agents](https://github.com/livekit/agents) — voice stack