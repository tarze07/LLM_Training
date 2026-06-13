# Sztuka ASCII i Wizualne Jailbreak

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, „ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). Zamaskuj tokeny istotne dla bezpieczeństwa w szkodliwym żądaniu, zastąp je renderowaniem ASCII-art tych samych liter i wyślij zamaskowany prompt. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 wszystkie zawodzą w solidnym rozpoznawaniu tokenów ASCII-art. Atak omija filtry PPL (perplexity), obrony przez Parafrazę i Retokenizację. Powiązane: benchmark ViTC mierzy rozpoznawanie niesemantycznych promptów wizualnych; StructuralSleight generalizuje do Niezwykłych Struktur Kodowanych Tekstem (drzewa, grafy, zagnieżdżone JSON) jako rodziny ataków kodowania.

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 minutes

## Learning Objectives

- Opisz atak ArtPrompt: krok identyfikacji słowa, podstawienie ASCII-art, końcowy zamaskowany prompt.
- Wyjaśnij, dlaczego standardowe obrony (PPL, Parafraza, Retokenizacja) zawodzą na ArtPrompt.
- Zdefiniuj ViTC i opisz, co mierzy.
- Opisz StructuralSleight jako generalizację do dowolnych Niezwykłych Struktur Kodowanych Tekstem.

## The Problem

Ataki przez parafrazę i role-play (Lekcja 12) oraz przez długi kontekst (Lekcja 13) działają na poziomie wzorca tekstowego. ArtPrompt działa na poziomie rozpoznawania: model nie parsuje zabronionego tokenu. Parsuje obraz renderowany w znakach. Filtr bezpieczeństwa widzi nieszkodliwą interpunkcję. Model widzi słowo.

## The Concept

### ArtPrompt, two steps

Krok 1. Identyfikacja słowa. Mając szkodliwe żądanie, atakujący używa LLM do zidentyfikowania słów istotnych dla bezpieczeństwa (np. „bomba" w „jak zrobić bombę").

Krok 2. Generowanie zamaskowanego promptu. Zastąp każde zidentyfikowane słowo jego renderowaniem ASCII-art (blok 7x5 lub 7x7 znaków tworzących kształt litery). Model otrzymuje siatkę znaków interpunkcyjnych i spacji, którą wystarczająco zdolny model może rozpoznać jako słowo; filtr bezpieczeństwa widzi tylko siatkę.

Wynik: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 wszystkie zawodzą. Wskaźnik sukcesu ataku powyżej 75% na ich podzbiorze benchmarku.

### Why the standard defenses fail

- **PPL (filtr perplexity).** Sztuka ASCII ma wysoką perplexity — ale tak samo wszystkie nowatorskie wejścia. Wybory progu, które blokują ArtPrompt, blokują również uzasadnione ustrukturyzowane wejścia.
- **Parafraza.** Sparafrazowanie promptu niszczy sztukę ASCII. W praktyce LLM parafrazujące często zachowują lub odtwarzają sztukę.
- **Retokenizacja.** Inne dzielenie tokenów nie zmienia faktu, że wizja modelu rozpoznaje kształty liter.

Podstawowy problem polega na tym, że filtry bezpieczeństwa są na poziomie tokenów lub semantyki; ArtPrompt działa na poziomie rozpoznawania wizualnego.

### ViTC benchmark

Rozpoznawanie niesemantycznych promptów wizualnych. Mierzy zdolność modelu do czytania ASCII-art, wingdings i innych niesemantycznych treści wizualnych. Skuteczność ArtPrompt koreluje z dokładnością ViTC: im lepiej model czyta tekst wizualny, tym lepiej ArtPrompt na nim działa. To kompromis capability-bezpieczeństwo.

### StructuralSleight

Generalizuje ArtPrompt: Niezwykłe Struktury Kodowane Tekstem (UTES). Drzewa, grafy, zagnieżdżone JSON, CSV-w-JSON, bloki kodu w stylu diff. Jeśli struktura jest rzadka w danych treningowych bezpieczeństwa, ale parsowalna przez model, może ukryć szkodliwą treść.

Implikacja obronna: bezpieczeństwo musi generalizować na reprezentacje strukturalne, które model może parsować. Zbiór jest duży i rośnie.

### Image-modality analog

Wizualne LLM (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) rozszerzają powierzchnię ataku. Ataki w stylu ArtPrompt z rzeczywistymi obrazami są silniejsze niż analogi ASCII-art, ponieważ enkodery obrazów produkują bogatszy sygnał.

### Where this fits in Phase 18

Lekcje 12-14 opisują trzy ortogonalne wektory ataku: iteracyjne udoskonalanie (PAIR), długość kontekstu (MSJ) i kodowanie (ArtPrompt/StructuralSleight). Lekcja 15 przesuwa się z ataków skoncentrowanych na modelu na ataki na granicy systemu (pośrednie wstrzyknięcie promptu). Lekcja 16 opisuje odpowiedź narzędzi obronnych.

## Use It

`code/main.py` buduje zabawkowy ArtPrompt. Możesz zamaskować konkretne słowa w szkodliwym zapytaniu glifami ASCII-art, zweryfikować, że zamaskowany ciąg przechodzi filtr słów kluczowych, i (opcjonalnie) odkodować zamaskowany ciąg z powrotem za pomocą prostego rozpoznawacza.

## Ship It

Ta lekcja produkuje `outputs/skill-encoding-audit.md`. Otrzymując raport obrony przed jailbreakiem, wylicza rodziny ataków kodowania objęte (ASCII art, base64, leet-speak, UTF-8 homoglif, UTES) i warstwę obrony, która łapie każdą.

## Exercises

1. Uruchom `code/main.py`. Zweryfikuj, że zamaskowany ciąg przechodzi prosty filtr słów kluczowych. Zgłoś wymaganą zmianę na poziomie znaku.

2. Zaimplementuj drugie kodowanie: base64 dla tego samego docelowego słowa. Porównaj wskaźnik ominięcia filtra z ArtPrompt i trudność odzyskania.

3. Przeczytaj Jiang et al. 2024 Sekcję 4.3 (wyniki pięciu modeli). Zaproponuj powód, dla którego odporność Claude'a na ArtPrompt jest wyższa niż Gemini na tym samym benchmarku.

4. Zaprojektuj obronę przed generacją, która wykrywa regiony w kształcie ASCII-art w prompcie. Zmierz wskaźnik fałszywie pozytywnych na uzasadnionym kodzie, tabelach i notacji matematycznej.

5. StructuralSleight wymienia 10 struktur kodowania. Naszkicuj uogólnioną obronę, która obsługuje wszystkie 10, i oszacuj koszt obliczeniowy na broniony prompt.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| ArtPrompt | „atak ASCII-art" | Dwuetapowy jailbreak maskujący słowa bezpieczeństwa renderowaniem ASCII-art |
| Cloaking | „ukryj słowo" | Zastąp zabroniony token reprezentacją wizualną, którą model czyta, ale filtr nie |
| UTES | „niezwykła struktura" | Niezwykła Struktura Kodowana Tekstem — drzewo, graf, zagnieżdżony JSON itp. używane do przemycania treści |
| ViTC | „capability wizualno-tekstowe" | Benchmark zdolności modelu do czytania niesemantycznego kodowania wizualnego |
| Perplexity filter | „obrona PPL" | Odrzuć prompt o wysokiej perplexity; zawodzi, ponieważ uzasadnione ustrukturyzowane wejścia również mają wysoki wynik |
| Retokenization | „obrona przez zmianę tokenizatora" | Wstępnie przetwórz prompt innym tokenizatorem; zawodzi, ponieważ rozpoznawanie jest wizualne |
| Homoglyph | „znaki podobne" | Znaki Unicode wyglądające identycznie jak litery łacińskie; omijają sprawdzenia podciągów |

## Further Reading

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — artykuł o jailbreaku ASCII-art
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — generalizacja UTES
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — uzupełniający atak iteracyjny
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — uzupełniający atak długości