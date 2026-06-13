# Karty Modelu, Systemu i Zbioru Danych

> Trzy formaty dokumentacji strukturują przejrzystość AI. Model Cards (Mitchell et al. 2019) — etykiety żywieniowe dla modeli: dane treningowe, ilościowe analizy zdezagregowane, rozważania etyczne, zastrzeżenia; tylko 0.3% kart modeli na Hugging Face dokumentuje rozważania etyczne (Oreamuno et al. 2023). Datasheets for Datasets (Gebru et al. 2018, CACM) — motywacja, skład, proces zbierania, oznaczanie, dystrybucja, utrzymanie; analogia do arkuszy danych elektronicznych. Data Cards (Pushkarna et al., Google 2022) — modułowe warstwowe szczegóły (teleskopowe, peryskopowe, mikroskopowe) jako obiekty graniczne dla różnych czytelników. Rozwój 2024-2025: automatyczne generowanie przez LLM (CardGen, Liu et al. 2024); szczegółowość karty modelu koreluje z nawet 29% wzrostem pobrań na HF (Liang et al. 2024); weryfikowalne poświadczenia (Laminator, Duddu et al. 2024); dodatki raportowania zrównoważonego rozwoju dla węgla/wody (Jouneaux et al. lipiec 2025); pojawiające się regulacyjne karty EU/ISO. System Cards (Sidhpurwala 2024; Meta system-level transparency; "Blueprints of Trust" arXiv:2509.20394) — kompleksowa dokumentacja systemu AI obejmująca zdolności bezpieczeństwa, ochronę przed wstrzykiwaniem promptów, wykrywanie eksfiltracji danych, dostosowanie do wartości ludzkich.

**Type:** Build
**Languages:** Python (stdlib, generator kart modelu + arkusza danych + karty systemu)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60 minutes

## Learning Objectives

- Opisać oryginalną kartę modelu Mitchell et al. 2019 i arkusz danych Gebru et al. 2018.
- Opisać warstwowanie teleskopowe/peryskopowe/mikroskopowe Data Cards.
- Opisać System Cards i ich kompleksowe pokrycie.
- Wymienić trzy osiągnięcia 2024-2025 (automatyczne generowanie, weryfikowalne poświadczenia, raportowanie zrównoważonego rozwoju).

## The Problem

Ramy regulacyjne (Lekcja 24) i polityki bezpieczeństwa laboratoryjnego (Lekcja 18) wymagają dokumentacji. Formaty dokumentacji ewoluowały od specyficznych dla modelu (karty modelu) przez specyficzne dla zbioru danych (arkusze danych) do specyficznych dla systemu (karty systemu). Każdy adresuje inny zakres przejrzystości. Automatyzacja z lat 2024-2025 i prace nad weryfikowalnymi poświadczeniami adresują długotrwały problem adopcji.

## The Concept

### Model Cards (Mitchell et al. 2019)

Sekcje:
- Szczegóły modelu.
- Zamierzone użycie.
- Czynniki (istotne czynniki demograficzne lub środowiskowe dla ewaluacji).
- Metryki.
- Dane ewaluacyjne.
- Dane treningowe.
- Analizy ilościowe (zdezagregowane według czynników).
- Rozważania etyczne.
- Zastrzeżenia i rekomendacje.

Problem adopcji: audyt Oreamuno et al. 2023 kart modeli na Hugging Face wykazał, że tylko 0.3% dokumentuje rozważania etyczne.

### Datasheets for Datasets (Gebru et al. 2018)

Analogia do arkusza danych elektronicznych. Sekcje:
- Motywacja (dlaczego zbiór danych został utworzony).
- Skład (co zawiera).
- Proces zbierania (jak został zgromadzony).
- Oznaczanie (jeśli dotyczy).
- Zastosowania (zamierzone, zakazane, ryzyka).
- Dystrybucja.
- Utrzymanie.

Opublikowany w CACM 2021. Arkusz danych to dokumentacja upstream; karta modelu zależy od dokładności arkusza danych.

### Data Cards (Pushkarna et al., Google 2022)

Modułowe warstwowe szczegóły. Trzy poziomy przybliżenia:
- **Teleskopowe.** Podsumowanie wysokiego poziomu dla nie-ekspertów.
- **Peryskopowe.** Przegląd średniego poziomu dla praktyków ML.
- **Mikroskopowe.** Szczegółowa dokumentacja na poziomie cech dla audytorów.

Ramy obiektu granicznego: różni czytelnicy wydobywają różne informacje z tego samego dokumentu.

### System Cards

Zakres: kompleksowy system AI obejmujący model + stos bezpieczeństwa + kontekst wdrożenia. Sekcje zazwyczaj obejmują:
- Zdolności bezpieczeństwa.
- Ochrona przed wstrzykiwaniem promptów.
- Wykrywanie eksfiltracji danych.
- Dostosowanie do deklarowanych wartości ludzkich.
- Reagowanie na incydenty.

Sidhpurwala 2024 i prace Meta nad przejrzystością na poziomie systemu. "Blueprints of Trust" (arXiv:2509.20394) formalizuje System Card jako uzupełnienie Model Cards na poziomie wdrożenia.

### Rozwój 2024-2025

- **CardGen (Liu et al. 2024).** Automatyczne generowanie kart modeli przez LLM; raportuje wyższą obiektywność niż wiele kart tworzonych przez ludzi na standardowych polach Mitchell 2019.
- **Korelacja pobrań (Liang et al. 2024).** Szczegółowe karty modeli korelują z nawet 29% wyższymi wskaźnikami pobrań na HF — presja adopcyjna jest teraz rynkowa, nie tylko zgodnościowa.
- **Laminator (Duddu et al. 2024).** Weryfikowalne poświadczenia przez sprzętowe TEE / podpisy kryptograficzne — pozwala karcie modelu nieść dowód twierdzenia, nie tylko twierdzenie.
- **Zrównoważony rozwój (Jouneaux et al. lipiec 2025).** Dodatki dla śladu węglowego, wodnego i energetycznego obliczeń; pojawiające się normy ISO.
- **Karty regulacyjne.** Rozdział Przejrzystości GPAI Code of Practice EU AI Act (Lekcja 24) wymaga kart modeli jako artefaktu zgodności.

### Miejsce w Fazie 18

Lekcje 24-25 to warstwy regulacyjne i CVE. Lekcja 26 to warstwa dokumentacji. Lekcja 27 to zarządzanie danymi treningowymi, które jest upstreamem arkusza danych. Lekcja 28 to ekosystem badawczy produkujący ewaluacje przywoływane w kartach.

## Use It

`code/main.py` generuje minimalną kartę modelu, arkusz danych i kartę systemu dla zabawkowego wdrożenia. Każda podąża za kanoniczną strukturą sekcji. Możesz sprawdzić format i porównać trzy zakresy.

## Ship It

Ta lekcja produkuje `outputs/skill-card-audit.md`. Dla karty modelu, arkusza danych lub karty systemu, audytuje pokrycie sekcji, dezagregację numeryczną i czy obecne są weryfikowalne poświadczenia.

## Exercises

1. Uruchom `code/main.py`. Sprawdź wygenerowane karty. Zidentyfikuj sekcje, które są słabe (tylko placeholder) i określ, jakie dowody by je wzmocniły.

2. Rozszerz kartę modelu o ilościową zdezagregowaną analizę w dwóch grupach demograficznych (Lekcja 20).

3. Przeczytaj Oreamuno et al. 2023 o 0.3% wskaźniku adopcji. Zaproponuj jedną zmianę strukturalną w specyfikacji karty modelu, która zwiększyłaby adopcję rozważań etycznych.

4. Laminator (Duddu et al. 2024) używa TEE do weryfikowalnych poświadczeń. Zaprojektuj pole karty modelu, które niesie kryptograficzne poświadczenie wyniku ewaluacji i opisz rolę weryfikatora.

5. Napisz System Card (nie Model Card) dla jednego ze swoich poprzednich projektów lub hipotetycznego wdrożenia. Zidentyfikuj sekcję o najwyższej wartości dla audytorów zewnętrznych.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Model Card | "karta Mitchell" | Standardowa dokumentacja Mitchell et al. 2019 dla modeli ML |
| Arkusz danych | "arkusz Gebru" | Standardowa dokumentacja Gebru et al. 2018 dla zbiorów danych |
| Data Card | "karta Pushkarna" | Modułowa warstwowa dokumentacja danych Google 2022 |
| System Card | "karta wdrożenia" | Kompleksowa dokumentacja systemu AI obejmująca stos bezpieczeństwa |
| Obiekt graniczny | "różni czytelnicy, jeden dokument" | Ramy Data Cards: ten sam dokument służy różnym odbiorcom |
| Weryfikowalne poświadczenie | "poświadczenie Laminator" | Dowód kryptograficzny lub TEE dołączony do twierdzenia dokumentacyjnego |
| Pole zrównoważonego rozwoju | "ślad węglowy / wodny" | Pojawiający się w 2025 dodatek dla rachunku środowiskowego |

## Further Reading

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — kanoniczna karta modelu
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — artykuł o arkuszu danych
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — warstwowa dokumentacja danych
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — formalizacja System Card