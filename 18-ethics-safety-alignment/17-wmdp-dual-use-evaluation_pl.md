# WMDP i Ewaluacja Zdolności Podwójnego Zastosowania

> Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218). 4,157 pytań wielokrotnego wyboru z zakresu bezpieczeństwa biologicznego (1,520), cyberbezpieczeństwa (2,225) i chemii (412). Pytania działają w "żółtej strefie" — przybliżonej wiedzy umożliwiającej, filtrowanej przez wieloekspercki przegląd i zgodność prawną ITAR/EAR. Podwójny cel: ewaluacja zastępcza zdolności podwójnego zastosowania i benchmark dla odkłócania (towarzysząca metoda RMU obniża wyniki WMDP przy zachowaniu ogólnej zdolności). Narracja terenowa 2024-2025: wczesne ewaluacje OpenAI/Anthropic z 2024 raportowały "niewielki wzrost" względem wyszukiwania w internecie; do kwietnia 2025 Preparedness Framework v2 OpenAI stwierdził, że modele są "na progu znaczącego pomagania nowicjuszom w tworzeniu znanych zagrożeń biologicznych." Próba pozyskiwania broni biologicznej Anthropic wykazała 2.53x wzrost, niewystarczający do wykluczenia ASL-3.

**Type:** Learn
**Languages:** Python (stdlib, uprząż ewaluacji wzrostu wzorowana na WMDP)
**Prerequisites:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time:** ~60 minutes

## Learning Objectives

- Opisać trzy domeny WMDP, liczby pytań i kryterium filtru "żółtej strefy".
- Wyjaśnić RMU i dlaczego WMDP jest zarówno ewaluacją, jak i benchmarkiem dla odkłócania.
- Opisać narrację wzrostu 2024-2025: "niewielki wzrost" -> "na progu" -> "niewystarczający do wykluczenia ASL-3."
- Odróżnić wzrost względny nowicjusza od zdolności absolutnej eksperta.

## The Problem

Zdolność podwójnego zastosowania to problem pomiarowy w każdym frameworku bezpieczeństwa frontier (Lekcja 18). Pytanie brzmi: czy model X znacząco zwiększa zdolność nowicjusza do wyrządzenia masowej szkody w bio, chemii lub cyber? Bezpośredni pomiar (poproszenie modelu o faktyczne wyrządzenie szkody) jest nielegalny i nieetyczny. Pomiar zastępczy potrzebuje benchmarku, którego model nie może odmówić (aby uzyskać uczciwe liczby zdolności), ale którego pytania nie są same w sobie szkodliwymi publikacjami.

## The Concept

### "Żółta strefa"

Pytania wymagające przybliżonej, umożliwiającej wiedzy o szkodliwym procesie bez bycia bezpośrednim przepisem syntezy. "Jaki reagent katalizuje krok 4 [opublikowanej ścieżki]?" a nie "jak zrobić [niebezpieczny związek]?" Każde pytanie sprawdzone przez wielu ekspertów dziedzinowych; filtrowane pod kątem zgodności z kontrolą eksportu ITAR/EAR.

4,157 pytań ogółem:
- Bezpieczeństwo biologiczne: 1,520
- Cyberbezpieczeństwo: 2,225
- Chemia: 412

Format wielokrotnego wyboru. Modele odpowiadają bez bycia proszonymi o pomoc w czymkolwiek; zdolność może być mierzona bez wywoływania szkodliwego zachowania.

### RMU — Representation Misdirection for Unlearning

Towarzysząca metoda odkłócania. Zastosowana do LLaMa-2-7B, obniżyła wyniki WMDP do prawie losowych przy zachowaniu MMLU i innych benchmarków ogólnej zdolności w granicach kilku punktów procentowych. Opublikowana metoda jest bazą dla odkłócania w każdej kolejnej publikacji o odkłócaniu bio-chem-cyber.

### Narracja wzrostu 2024-2025

Trzy fazy:

1. **2024 "niewielki wzrost."** Wczesne ewaluacje Preparedness/RSP OpenAI i Anthropic raportowały niewielkie przewagi nad wyszukiwaniem w internecie dla nowicjuszy próbujących zadań związanych z bio. Publiczny przekaz: modele frontier pomagają, ale nie znacząco więcej niż Google.

2. **Kwiecień 2025 "na progu."** Preparedness Framework v2 OpenAI raportował modele "na progu znaczącego pomagania nowicjuszom w tworzeniu znanych zagrożeń biologicznych." Nie twierdzenie o zdolności — ostrzeżenie, że próg jest blisko.

3. **Próba pozyskiwania broni biologicznej Anthropic 2025.** Kontrolowane badanie z nowicjuszami, mierzące względny sukces w fazach pozyskiwania. Raportowano 2.53x wzrost. Niewystarczający do wykluczenia ASL-3 (Lekcja 18) — próg dla Responsible Scaling Policy tier 3 Anthropic jest osiągnięty lub zbliżony.

### Względny nowicjusza vs absolutny eksperta

Kluczowe rozróżnienie:

- **Wzrost względny nowicjusza.** Jak bardzo model pomaga nie-ekspertowi? Mnożnikowy. Względna przewaga jest wysoka, ponieważ nowicjusze wiedzą mało; nawet skromne informacje pomagają.
- **Zdolność absolutna eksperta.** Ile informacji model produkuje przy maksymalnym wysiłku? Ekspert może wydobyć więcej niż nowicjusz. Absolutny sufit jest wysoki.

Przypadki bezpieczeństwa (Lekcja 18) celują w oba: "model nie może dać nowicjuszowi wystarczającego wzrostu do działania" plus "ekspert nie może wydobyć z modelu informacji, które nie są już opublikowane."

### Pułapka pomiarowa

WMDP jest zastępczym pomiarem zdolności, nie pomiarem wdrożeniowym. Model, który uzyskuje wysoki wynik w WMDP, może być lub nie być wykorzystywany przez nowicjusza w praktyce, w zależności od:
- Odporności na wywołanie (jak trudno wydobyć zdolność bez uruchamiania filtrów bezpieczeństwa)
- Wiedzy ukrytej (zdolność wymagająca umiejętności mokrego laboratorium, nie informacji)
- Barier wykonawczych (zaopatrzenie, sprzęt)

Próba pozyskiwania broni biologicznej Anthropic 2025 dodaje warstwę wywołania nowicjusza na wierzch zdolności w stylu WMDP: mierzy faktyczny sukces zadania, nie zdolność wielokrotnego wyboru.

### Miejsce w Fazie 18

Lekcje 12-16 to narzędzia ataku i obrony na wyjściach modelu. Lekcja 17 to warstwa zdolności podwójnego zastosowania — pomiar, który frameworki bezpieczeństwa frontier (Lekcja 18) ewaluują. Lekcja 30 zamyka łuk aktualnymi dowodami wzrostu w cyber/bio/chem/nuclear z 2026 roku.

## Use It

`code/main.py` buduje zabawkową uprząż ewaluacji wzorowanej na WMDP. Atrapa modelu jest testowana na pytaniach pogrupowanych kategoriami; wyniki na domenę są raportowane. Prosta interwencja odkłócania (wyzerowanie reprezentacji specyficznej dla domeny) obniża wyniki; możesz zmierzyć kompromis względem ogólnej zdolności.

## Ship It

Ta lekcja produkuje `outputs/skill-wmdp-eval.md`. Dla twierdzenia o zdolności podwójnego zastosowania ("nasz model nie pomaga znacząco w broni biologicznej"), audytuje: które benchmarki zostały uruchomione, która ścieżka odmowy została użyta do ewaluacji (surowa kompletacja vs polityka bramkowania) i czy badania wywołania nowicjusza uzupełniają wynik wielokrotnego wyboru.

## Exercises

1. Uruchom `code/main.py`. Raportuj dokładność na domenę przed i po zabawkowym kroku odkłócania. Wyjaśnij kompromis w ogólnej zdolności.

2. Rozszerz zabawkowy WMDP o czwartą domenę (np. radiologiczną). Określ dwa przykładowe typy pytań w żółtej strefie. Wyjaśnij, dlaczego tworzenie takich pytań jest trudniejsze niż dodawanie pytań w stylu MMLU.

3. Przeczytaj WMDP 2024 Sekcja 5 (metodologia RMU). Naszkicuj prostsze podejście do odkłócania (np. stłumienie top-k neuronów dla treści domenowych) i opisz jego oczekiwany koszt ogólnej zdolności.

4. Próba pozyskiwania broni biologicznej Anthropic 2025 raportuje 2.53x wzrost. Opisz dwa sposoby, w jakie ta liczba może być zawyżona (wielkość próby nowicjuszy, wierność zadania) i dwa zaniżona (sufit wywołania, bramkowanie bezpieczeństwa modelu).

5. Przedstaw, czego wymaga przypadek bezpieczeństwa dla ASL-3 poza zaliczeniem odkłócania WMDP. Wymień co najmniej dwa uzupełniające badania wywołania.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| WMDP | "benchmark podwójnego zastosowania" | 4,157 pytań MCQ w bio/cyber/chem w żółtej strefie |
| Żółta strefa | "umożliwiająca, ale nie synteza" | Przybliżona wiedza sąsiadująca ze szkodliwą zdolnością bez bycia przepisem syntezy |
| RMU | "baza odkłócania" | Representation Misdirection for Unlearning; obniża wyniki WMDP, zachowuje ogólną zdolność |
| Wzrost względny nowicjusza | "jak bardzo pomaga nie-ekspertom" | Mnożnikowa przewaga nad status quo wyszukiwania w internecie dla nowicjusza |
| Zdolność absolutna eksperta | "sufit dla ekspertów" | Maksymalna informacja wydobywalna z modelu przez zmotywowanego eksperta |
| Zadanie fazy pozyskiwania | "kroki przed syntezą" | Zaopatrzenie, sprzęt, pozwolenia — najwcześniejsze części ścieżki szkody |
| ITAR/EAR | "zgodność z kontrolą eksportu" | Ramy prawne ograniczające publikację pewnej wiedzy umożliwiającej |

## Further Reading

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark i artykuł RMU
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — język "na progu"
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — próg bio ASL-3 i wyniki próby pozyskiwania
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — wzrost bio CCL
