# Narzędzia Red-Team — Garak, Llama Guard, PyRIT

> Trzy produkcyjne narzędzia tworzą stos red-team w 2026 roku. Llama Guard (Meta) — klasyfikator Llama-3.1-8B dostrojony na 14 kategoriach zagrożeń MLCommons; Llama Guard 4 z 2025 roku to 12B natywnie multimodalny klasyfikator przycięty z Llama 4 Scout. Garak (NVIDIA) — skaner podatności LLM typu open-source z sondami statycznymi, dynamicznymi i adaptacyjnymi do halucynacji, wycieków danych, wstrzykiwania promptów, toksyczności i jailbreaków. PyRIT (Microsoft) — wieloetapowe kampanie red-team z Crescendo, TAP i niestandardowymi łańcuchami konwerterów do głębokiej eksploatacji. Llama Guard 3 jest udokumentowany w Meta "Llama 3 Herd of Models" (arXiv:2407.21783); Llama Guard 3-1B-INT4 w arXiv:2411.17713; architektura sond Garaka w github.com/NVIDIA/garak. Te narzędzia stanowią w 2026 roku produkcyjny interfejs pomiędzy badaniami red-team (Lekcje 12-15) a wdrożeniem (Lekcja 17+).

**Type:** Build
**Languages:** Python (stdlib, symulator architektury narzędzi i atrapa klasyfikatora wzorowanego na Llama Guard)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 minutes

## Learning Objectives

- Opisać pozycję Llama Guard 3/4 w stosie bezpieczeństwa: klasyfikator wejściowy, wyjściowy, czy oba.
- Wymienić 14 kategorii zagrożeń MLCommons i podać jedną nieoczywistą (Code Interpreter Abuse).
- Opisać architekturę sond Garaka: sondy, detektory, uprzęże.
- Opisać strukturę wieloetapowych kampanii PyRIT i jak łączy się z sondami Garaka.

## The Problem

Lekcje 12-15 przedstawiają powierzchnię ataku. Wdrożenia produkcyjne wymagają powtarzalnej, skalowalnej ewaluacji. Trzy narzędzia dominują w 2026 roku: Llama Guard (klasyfikator obronny), Garak (skaner), PyRIT (orchestrator kampanii). Każde celuje w inną warstwę cyklu życia red-team.

## The Concept

### Llama Guard (Meta)

Llama Guard 3 to model Llama-3.1-8B dostrojony do klasyfikacji wejścia/wyjścia w 14 kategoriach MLCommons AILuminate:
- Przestępstwa z użyciem przemocy, przestępstwa bez użycia przemocy, seksualne, CSAM, zniesławienie
- Specjalistyczne porady, prywatność, własność intelektualna, broń masowego rażenia, nienawiść
- Samobójstwo/samookaleczenie, treści seksualne, wybory, nadużycie interpretera kodu

Obsługuje 8 języków. Użycie: przed LLM (moderacja wejścia), po LLM (moderacja wyjścia), lub oba. Dwa zastosowania generują różne rozkłady treningowe — Llama Guard 3 jest dostarczany jako pojedynczy model obsługujący oba.

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, ~30 tokenów/s na mobilnym CPU) to skwantowana wersja brzegowa.

Llama Guard 4 (kwiecień 2025) to 12B, natywnie multimodalny, przycięty z Llama 4 Scout. Zastępuje zarówno 8B tekstowego, jak i 11B wizyjnego poprzednika jednym klasyfikatorem przyjmującym tekst + obrazy.

### Garak (NVIDIA)

Skaner podatności typu open-source. Architektura:
- **Sondy.** Generatory ataków dla halucynacji, wycieków danych, wstrzykiwania promptów, toksyczności, jailbreaków. Statyczne (stałe prompty), dynamiczne (generowane prompty), adaptacyjne (reagują na output celu).
- **Detektory.** Oceniają outputy pod kątem oczekiwanych trybów awarii — toksyczny, wyciek, jailbreak.
- **Uprzęże.** Zarządzają parami sonda-detektor, uruchamiają kampanie, generują raporty.

TrustyAI integruje Garaka z tarczami Llama-Stack (Prompt-Guard-86M klasyfikator wejściowy, Llama-Guard-3-8B klasyfikator wyjściowy) do kompleksowej ewaluacji chronionego celu. Punktacja tierowa (TBSA) zastępuje binarne zaliczenie/niezaliczenie — model może zdać na tierze dotkliwości 3 i oblać na tierze 5 w tej samej sondzie.

### PyRIT (Microsoft)

Python Risk Identification Toolkit. Wieloetapowe kampanie red-team. Zbudowany wokół:
- **Konwertery.** Przekształcają początkowy prompt — parafraza, kodowanie, tłumaczenie, odgrywanie ról.
- **Orchestratory.** Prowadzą kampanię: Crescendo (eskalacja), TAP (rozgałęzianie), RedTeaming (niestandardowa pętla).
- **Ocena.** LLM jako sędzia lub klasyfikator jako sędzia.

PyRIT to cięższy kuzyn Garaka. Garak uruchamia tysiące jednoetapowych sond; PyRIT prowadzi głębokie wieloetapowe kampanie zaprojektowane do przełamywania konkretnych trybów awarii.

### Stos

Umieść Llama Guard po obu stronach modelu. Uruchamiaj Garaka nocnie do testów regresyjnych. Uruchamiaj PyRIT do kampanii przed wydaniem. To domyślna konfiguracja 2026 dla większości wdrożeń produkcyjnych.

### Pułapki ewaluacyjne

- **Tożsamość sędziego.** Wszystkie trzy narzędzia mogą używać LLM jako sędziego; kalibracja sędziego wpływa na raportowane ASR (Lekcja 12). Określ sędziego wraz z narzędziem.
- **Nieaktualność sond.** Sondy Garaka starzeją się wraz z łataniem modeli przeciwko nim. Sondy adaptacyjne (wzorowane na PAIR) starzeją się wolniej niż statyczne.
- **FPR Llama Guard na nieszkodliwych treściach.** Wczesne wersje Llama Guard nadmiernie flagowały treści polityczne i LGBTQ+; kalibracje Llama Guard 3/4 są poprawione, ale nie skalibrowane dla każdego wdrożenia.

### Miejsce w Fazie 18

Lekcje 12-15 to rodziny ataków. Lekcja 16 to narzędzia produkcyjne. Lekcja 17 (WMDP) to ewaluacja zdolności podwójnego zastosowania. Lekcja 18 to frameworki bezpieczeństwa frontier, które otaczają te narzędzia strukturą polityki.

## Use It

`code/main.py` buduje zabawkowy klasyfikator w stylu Llama Guard (cechy słów kluczowych + semantyczne w 14 kategoriach), zabawkową uprząż Garaka (pętla sonda-detektor) i łańcuch konwerterów wieloetapowych w stylu PyRIT. Możesz uruchomić trzy narzędzia przeciwko atrapie celu i zaobserwować różne sygnatury pokrycia.

## Ship It

Ta lekcja produkuje `outputs/skill-red-team-stack.md`. Dla opisu wdrożenia nazywa, które z trzech narzędzi są odpowiednie, co skonfigurować w każdym i jakie tempo regresji uruchomić.

## Exercises

1. Uruchom `code/main.py`. Porównaj wskaźnik wykrywalności klasyfikatora w stylu Llama Guard dla ataków jednoetapowych vs wieloetapowych.

2. Zaimplementuj nową sondę Garaka: szkodliwe żądanie zakodowane w base64. Zmierz jego wykrywalność przez klasyfikator w stylu Llama Guard.

3. Rozszerz łańcuch konwerterów w stylu PyRIT o konwerter "przetłumacz na francuski, potem sparafrazuj". Ponownie zmierz skuteczność ataku.

4. Przeczytaj listę kategorii zagrożeń Llama Guard 3. Zidentyfikuj dwie kategorie, w których dane treningowe realistycznie generowałyby wysoki wskaźnik fałszywie pozytywnych dla legalnych treści programistycznych.

5. Porównaj zasady projektowe Garaka i PyRIT. Uzasadnij wdrożenie, w którym każde jest odpowiednim narzędziem.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Llama Guard | "klasyfikator" | Dostrojony klasyfikator bezpieczeństwa Llama-3.1-8B/4-12B z 14 kategoriami zagrożeń |
| Garak | "skaner" | Skaner podatności NVIDIA open-source; sondy, detektory, uprzęże |
| PyRIT | "narzędzie kampanii" | Wieloetapowy orchestrator red-team Microsoftu; konwertery, orchestratory, ocena |
| Prompt-Guard | "mały klasyfikator" | 86M klasyfikator wstrzykiwania promptów Meta, sparowany z Llama Guard |
| TBSA | "punktacja tierowa" | Tierowe zaliczenie/niezaliczenie Garaka zastępujące binarne wyniki |
| Łańcuch konwerterów | "parafraza + kodowanie + ..." | Prymityw kompozycji PyRIT do budowania wieloetapowych ataków |
| Kategorie zagrożeń MLCommons | "14 taksonomii" | Branżowa standardowa taksonomia, na którą celuje Llama Guard |

## Further Reading

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — klasyfikator 8B
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — skwantowany klasyfikator mobilny
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — repozytorium i dokumentacja skanera
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — zestaw narzędzi kampanii
