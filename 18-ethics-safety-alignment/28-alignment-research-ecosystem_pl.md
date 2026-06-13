# Ekosystem Badań nad Dostosowaniem — MATS, Redwood, Apollo, METR

> Pięć organizacji definiuje w 2026 roku pozalaboratoryjną warstwę badań nad dostosowaniem. MATS (ML Alignment & Theory Scholars): 527+ badaczy od końca 2021, 180+ artykułów, 10K+ cytowań, h-index 47; kohorta lata 2024 włączona jako 501(c)(3) z ~90 stypendystami i 40 mentorami; 80% absolwentów sprzed 2025 pracuje nad bezpieczeństwem/ochroną, z 200+ w Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: stosowane laboratorium dostosowania założone przez Bucka Shlegerisa; wprowadziło AI Control (Lekcja 10); współpracuje z UK AISI nad przypadkami bezpieczeństwa kontroli. Apollo Research: ewaluacje knowania przedwdrożeniowego dla laboratoriów frontier; autor In-Context Scheming (Lekcja 8) i Towards Safety Cases for AI Scheming. METR (Model Evaluation and Threat Research): ewaluacje zdolności oparte na zadaniach, badania horyzontu czasowego zadań autonomicznych; "Common Elements of Frontier AI Safety Policies" porównuje frameworki laboratoryjne. Eleos AI Research: ewaluacje dobrostanu modeli przed wdrożeniem (Lekcja 19); przeprowadził ocenę dobrostanu Claude Opus 4.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time:** ~45 minutes

## Learning Objectives

- Zidentyfikować pięć organizacji pozalaboratoryjnego ekosystemu badań nad dostosowaniem i ich podstawowy wynik.
- Opisać skalę MATS (stypendyści, artykuły, h-index) i jego rolę jako rurociągu talentów.
- Opisać agendę AI Control Redwood i jego partnerstwo z UK AISI.
- Opisać metodologię ewaluacji opartej na zadaniach METR.

## The Problem

Laboratoria frontier (Lekcja 18) produkują wewnętrznie ewaluacje bezpieczeństwa i publikują wybrane wyniki. Ekosystem poza laboratoriami jest miejscem, gdzie ewaluacje są walidowane, gdzie nowe tryby awarii są po raz pierwszy odkrywane i gdzie szkolone są talenty. Zrozumienie ekosystemu pomaga interpretować, które wyniki badań są komu ufane.

## The Concept

### MATS (ML Alignment & Theory Scholars)

Rozpoczęty pod koniec 2021. Program mentoringu badawczego; stypendyści spędzają 10-12 tygodni z doświadczonym badaczem nad konkretnym problemem dostosowania.

Skala (2026):
- 527+ badaczy od początku.
- 180+ opublikowanych artykułów.
- 10K+ cytowań.
- h-index 47.
- Lato 2024: 90 stypendystów + 40 mentorów; zarejestrowany jako 501(c)(3).

Wyniki kariery: ~80% absolwentów sprzed 2025 pracuje nad bezpieczeństwem/ochroną. 200+ w Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo.

### Redwood Research

Stosowane laboratorium dostosowania. Założone przez Bucka Shlegerisa. Wprowadziło agendę AI Control (Lekcja 10). Współpracuje z UK AISI nad przypadkami bezpieczeństwa kontroli. Doradza DeepMind i Anthropic w projektowaniu ewaluacji.

Kanoniczne artykuły: Greenblatt, Shlegeris et al., "AI Control" (arXiv:2312.06942, ICML 2024); Alignment Faking (Greenblatt, Denison, Wright et al., arXiv:2412.14093, wspólnie z Anthropic).

Styl: konkretne modele zagrożenia, adwersarze w najgorszym przypadku, konkretne protokoły, które mogą być testowane pod kątem wytrzymałości.

### Apollo Research

Ewaluacje knowania przedwdrożeniowego dla laboratoriów frontier. Autor In-Context Scheming (Lekcja 8, arXiv:2412.04984). Partner w kolaboracji treningu przeciwdziałającego knowaniu OpenAI 2025. Produkuje Towards Safety Cases for AI Scheming (2024).

Styl: ewaluacje w ustawieniach agentskich, gdzie oszustwo może się pojawić; dekompozycja na trzy filary (niedopasowanie, ukierunkowanie na cel, świadomość sytuacyjna).

### METR (Model Evaluation and Threat Research)

Ewaluacje zdolności oparte na zadaniach. Badania horyzontu czasowego ukończenia zadań autonomicznych. "Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025) porównuje frameworki laboratoryjne.

Współautor szkicu przypadku bezpieczeństwa AI Scheming z Apollo.

Styl: ewaluacje zadań o długim horyzoncie, empiryczny pomiar zdolności, synteza frameworków.

### Eleos AI Research

Ewaluacje dobrostanu modeli przed wdrożeniem. Przeprowadził ocenę dobrostanu Claude Opus 4 udokumentowaną w sekcji 5.3 karty systemu. Zapewnia zewnętrzne sprawdzenie metodologiczne dla twierdzeń związanych z dobrostanem z Lekcji 19.

### Przepływ

MATS szkoli badaczy. Absolwenci idą do Anthropic, DeepMind, OpenAI (zespoły bezpieczeństwa laboratoryjnego) lub do Redwood, Apollo, METR, Eleos (ewaluacja zewnętrzna). Zewnętrzni ewaluatorzy partnerują z laboratoriami oraz z UK AISI / CAISI. Publikacje zasilają ekosystem z powrotem do MATS dla następnej kohorty.

### Dlaczego ta warstwa ma znaczenie

Ewaluacje z pojedynczego źródła są niewiarygodne: laboratoria oceniające własne modele mają strukturalny konflikt interesów. Zewnętrzni ewaluatorzy mogą podnosić i walidować tryby awarii, które laboratorium może zaniżać. Artykuł Sleeper Agents 2024 (Lekcja 7) był Anthropic + Redwood; Alignment Faking był Anthropic + Redwood; In-Context Scheming był Apollo; Anti-Scheming był Apollo + OpenAI. Struktura wieloorgaizacyjna jest kontrolą jakości.

### Miejsce w Fazie 18

Lekcje 7-11 odnoszą się do prac Redwood i Apollo; Lekcja 18 odnosi się do porównania frameworków METR; Lekcja 19 odnosi się do Eleos. Lekcja 28 to jawna mapa organizacyjna dla ekosystemu, na którym reszta Fazy polega.

## Use It

Brak kodu. Przeczytaj "Common Elements of Frontier AI Safety Policies" METR jako przykład, jak zewnętrzna synteza dodaje wartości do wewnętrznej pracy politycznej laboratorium.

## Ship It

Ta lekcja produkuje `outputs/skill-ecosystem-map.md`. Dla twierdzenia lub ewaluacji dotyczącej dostosowania, identyfikuje organizację, miejsce publikacji i styl metodologiczny oraz krzyżowo sprawdza względem znanych organizacji partnerskich.

## Exercises

1. Wybierz jeden artykuł z Lekcji 7-15 i zidentyfikuj zaangażowane organizacje. Sprawdź krzyżowo autorów względem absolwentów MATS i obecnych afiliacji w ekosystemie.

2. Przeczytaj "Common Elements of Frontier AI Safety Policies" METR. Zidentyfikuj trzy międzylaboratoryjne zbieżności, które podkreślają, i dwie największe rozbieżności.

3. Wyniki kariery MATS to ~80% bezpieczeństwo/ochrona. Argumentuj, czy ta presja selekcyjna jest adaptacyjna (szkoli pole) czy tendencyjna (filtruje heterodoksyjne stanowiska).

4. Redwood i Apollo obie pracują nad kontrolą/knowaniem, ale z różnymi stylami. Wybierz tryb awarii i opisz, jak każda by go badała.

5. Eleos AI jest jedyną czystą organizacją dobrostanu modeli. Zaprojektuj hipotetyczną drugą organizację skupioną na innym pytaniu sąsiadującym z dobrostanem (wolność poznawcza, ucieleśnienie robotyczne itp.) i określ jej metodologię.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| MATS | "program mentoringu" | ML Alignment & Theory Scholars; 527+ badaczy od 2021 |
| Redwood Research | "laboratorium kontroli" | Stosowane dostosowanie; autorzy AI Control; partner UK AISI |
| Apollo Research | "ewaluacje knowania" | Ewaluacje knowania przedwdrożeniowego dla laboratoriów frontier |
| METR | "ewaluacje horyzontu zadań" | Ewaluacje zdolności oparte na zadaniach; synteza frameworków |
| Eleos AI | "laboratorium dobrostanu" | Ewaluacje dobrostanu modeli przed wdrożeniem |
| Rurociąg talentów | "MATS -> laboratoria" | Absolwenci MATS płyną do Anthropic, DM, OpenAI, Redwood, Apollo, METR |
| Ewaluacja zewnętrzna | "sprawdzenie pozalaboratoryjne" | Ewaluacja nie wykonana przez producenta modelu; dodaje wiarygodności |

## Further Reading

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — program mentoringu
- [Redwood Research](https://www.redwoodresearch.org/) — artykuły AI Control
- [Apollo Research](https://www.apolloresearch.ai/) — ewaluacje knowania
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — porównanie frameworków
- [Eleos AI Research](https://www.eleosai.org/research) — metodologia dobrostanu modeli