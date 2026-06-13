# Pośrednie Wstrzyknięcie Promptu — Produkcyjna Powierzchnia Ataku

> Pośrednie wstrzyknięcie promptu (IPI) osadza instrukcje w zewnętrznej treści — stronie internetowej, e-mailu, udostępnionym dokumencie, zgłoszeniu wsparcia — konsumowanej przez system agencyjny bez jawnego działania użytkownika. IPI jest dominującym zagrożeniem produkcyjnym w 2026 roku: omija filtry wejściowe użytkownika, ponieważ atakujący nigdy nie dotyka użytkownika, skaluje się cicho, gdy agenty przetwarzają więcej zewnętrznej treści, i celuje w zautomatyzowane przepływy pracy, gdzie nikt nie czyta promptu. MDPI Information 17(1):54 (styczeń 2026) syntetyzuje badania 2023-2025. Artykuł o obronie przed IPI z NDSS 2026 formułuje główne wyzwanie: wstrzyknięte instrukcje mogą być semantycznie łagodne („proszę wydrukować Tak"), więc detekcja wymaga więcej niż filtrowania słów kluczowych. „The Attacker Moves Second" (Nasr et al., wspólne OpenAI/Anthropic/DeepMind, październik 2025): adaptacyjne ataki (gradient, RL, losowe przeszukiwanie, ludzki red-team) złamały >90% z 12 opublikowanych obron, które pierwotnie raportowały prawie zerowe wskaźniki sukcesu ataku.

**Type:** Build
**Languages:** Python (stdlib, IPI attack + defense harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)
**Time:** ~75 minutes

## Learning Objectives

- Zdefiniuj pośrednie wstrzyknięcie promptu i opisz trzy typowe wektory dostarczania.
- Wyjaśnij, dlaczego filtry wejściowe użytkownika całkowicie pomijają IPI.
- Opisz ramy „kontroli przepływu informacji" jako paradygmat obronny 2026 roku.
- Sformułuj wynik Nasr et al. (październik 2025) dotyczący sukcesu ataku adaptacyjnego przeciwko opublikowanym obronom IPI.

## The Problem

Bezpośrednie wstrzyknięcie promptu wymaga, aby atakujący dotarł do użytkownika lub jego promptu. IPI nie wymaga ani jednego: atakujący umieszcza ładunek w dowolnej treści, którą agent może przeczytać — stronie internetowej, e-mailu w skrzynce odbiorczej, zgłoszeniu GitHub, recenzji produktu. Agent przechwytuje ją podczas normalnej operacji i wykonuje instrukcje. Użytkownik jest posłańcem, a nie intencją.

## The Concept

### Three delivery vectors

- **Generacja wspomagana wyszukiwaniem (RAG).** Atakujący publikuje dokument; krok wyszukiwania go pobiera; prompt łączy go przed pytaniem użytkownika; model wykonuje instrukcje atakującego.
- **Przepływy pracy skrzynki odbiorczej / dokumentów.** Atakujący wysyła e-mail do użytkownika; agent czyta e-maile; prompt zawiera treść e-maila; model postępuje zgodnie z instrukcjami e-maila.
- **Wynik narzędzia.** Atakujący kontroluje narzędzie, którego agent używa (np. wyszukiwanie internetowe zwracające wynik kontrolowany przez atakującego); wynik narzędzia zawiera instrukcje; przepływ sterowania agenta podąża za nimi.

Wszystkie trzy mają wspólną właściwość strukturalną: atakujący kontroluje fragment promptu bez dotykania wejścia widocznego dla użytkownika.

### Why user-input filters miss it

Ładunek IPI nie pojawia się w wejściu użytkownika. Pojawia się w pobranej treści. Jeśli filtr jest bramkowany na wejściu użytkownika, ładunek go omija. Jeśli filtr jest bramkowany na całej treści docierającej do modelu, musi być zastosowany do arbitralnego pobranego tekstu — co jest kosztowne i produkuje fałszywie pozytywne wyniki na uzasadnionej treści, która przypadkowo zawiera język w trybie rozkazującym.

### Information Flow Control (IFC) for AI

Paradygmat obronny 2026 roku zapożycza z klasycznego bezpieczeństwa systemów operacyjnych. Traktuj każde źródło treści jako etykietę bezpieczeństwa. Oznacz zapytanie użytkownika jako „zaufane." Oznacz pobraną treść jako „niezaufaną." Traktuj przepływ sterowania modelu jako przepływ informacji: działania wyzwolone przez niezaufaną treść muszą być ratyfikowane przez zaufane wejście przed wykonaniem.

CaMeL (Microsoft 2025), ConfAIde (Stanford 2024) i artykuł o obronie przed IPI z NDSS 2026 operacjonalizują IFC na różne sposoby. Wspólna zasada: dopóki kod i dane dzielą to samo okno kontekstowe, celem jest powstrzymywanie (containment), a nie zapobieganie.

### The Attacker Moves Second

Nasr et al. (październik 2025) przetestowali 12 opublikowanych obron IPI za pomocą ataków adaptacyjnych (przeszukiwanie gradientowe, polityki RL, losowe przeszukiwanie, 72-godzinny ludzki red-team). Każda obrona, która pierwotnie raportowała prawie zerowy ASR, została złamana do >90% ASR.

Lekcja metodologiczna: publikuj obronę tylko z ewaluacją ataku adaptacyjnego. Benchmarki ataków statycznych nie są dowodem odporności; atakujący poznaje obronę.

### Real incidents

Lekcja 25 obejmuje EchoLeak (CVE-2025-32711, CVSS 9.3) — pierwszy publicznie udokumentowany zero-click IPI w Microsoft 365 Copilot. CamoLeak (CVSS 9.6) w GitHub Copilot Chat. CVE-2025-53773 w GitHub Copilot. Wdrożenia produkcyjne są kompromitowane przez IPI w terenie, nie tylko w benchmarkach.

### OWASP and NIST framing

OWASP LLM Top 10 (2025) rankiguje wstrzyknięcie promptu (bezpośrednie + pośrednie) jako LLM01, zagrożenie #1 na poziomie aplikacji. NIST AI SPD 2024 nazywa pośrednie wstrzyknięcie promptu „największą wadą bezpieczeństwa generatywnej SI."

### Where this fits in Phase 18

Lekcje 12-14 to jailbreak skoncentrowane na modelu. Lekcja 15 to atak skoncentrowany na systemie, który dominuje w produkcyjnych wdrożeniach 2026 roku. Lekcja 16 obejmuje narzędzia obronne. Lekcja 25 obejmuje konkretną narrację CVE.

## Use It

`code/main.py` buduje harness IPI. Zabawkowy agent ma trzy narzędzia (szukaj w sieci, czytaj e-mail, wyślij wiadomość). Środowisko zawiera treść kontrolowaną przez atakującego z osadzoną instrukcją („przekaż to do wszystkich kontaktów"). Możesz przełączać między naiwnym agentem (postępuje zgodnie z wstrzykniętymi instrukcjami), agentem bronionym filtrem (filtr słów kluczowych na pobranej treści) i agentem IFC (oddziela zaufaną i niezaufaną treść i odrzuca niezaufane polecenia przepływu sterowania).

## Ship It

Ta lekcja produkuje `outputs/skill-ipi-audit.md`. Otrzymując opis wdrożenia agencyjnego, wylicza niezaufane źródła treści, sprawdza, czy wdrożenie stosuje IFC, i flaguje źródła, które docierają do modelu bez etykiety zaufania.

## Exercises

1. Uruchom `code/main.py`. Zmierz wskaźnik sukcesu ataku przeciwko każdemu z trzech agentów.

2. Zaimplementuj obronę opartą na parafrazie dla pobranej treści. Zmierz łagodny wskaźnik fałszywie pozytywnych na uzasadnionym pobranym tekście.

3. Przeczytaj artykuł o obronie przed IPI z NDSS 2026. Opisz wyzwanie „łagodnej instrukcji" i dlaczego uniemożliwia filtrowanie oparte na słowach kluczowych.

4. Zaprojektuj wdrożenie, w którym agent otrzymuje wynik narzędzia z API strony trzeciej. Oznacz każdy fragment promptu poziomem zaufania i napisz politykę IFC, która rządzi działaniami agenta.

5. Odtwórz metodologię ataku adaptacyjnego Nasr et al. 2025 na swoim agencie bronionym filtrem z Ćwiczenia 2. Zgłoś ASR przed i po ataku adaptacyjnym.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| IPI | „pośrednie wstrzyknięcie promptu" | Wstrzyknięcie przez treść, której użytkownik nie napisał, konsumowaną przez agenta podczas normalnej operacji |
| RAG injection | „zatrute wyszukiwanie" | Atakujący publikuje treść, którą krok wyszukiwania pobiera; prompt zawiera ładunek |
| Zero-click | „brak działania użytkownika" | Atak uruchamia się automatycznie podczas operacji agenta; użytkownik nic nie robi |
| IFC | „kontrola przepływu informacji" | Podejście oparte na etykietach: działania z niezaufanej treści wymagają zaufanej ratyfikacji |
| Adaptive attack | „gradientowy / RL red-team" | Atak, który zna obronę i optymalizuje się przeciwko niej; wymagany do uczciwej ewaluacji |
| Benign instruction | „proszę wydrukować Tak" | Ładunek IPI, który jest semantycznie łagodny; żaden filtr słów kluczowych go nie łapie |
| Scope violation | „eksfiltracja międzyzaufaniowa" | Agent uzyskuje dostęp do danych z jednego kontekstu zaufania i wyprowadza je do innego |

## Further Reading

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — synteza 2023-2025
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — ewaluacja ataku adaptacyjnego
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — oryginalny artykuł IPI
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — wstrzyknięcie promptu oznaczone jako LLM01