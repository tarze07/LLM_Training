# Znakowanie Wodne — SynthID, Stable Signature, C2PA

> Trzy technologie strukturują w 2026 roku pochodzenie treści generowanych przez AI. SynthID (Google DeepMind) — znakowanie wodne obrazów uruchomione w sierpniu 2023, tekst+video w maju 2024 (Gemini + Veo), tekst open-source'owy w październiku 2024 przez Responsible GenAI Toolkit, ujednolicony wielomedialny detektor w listopadzie 2025 wraz z Gemini 3 Pro. Znakowanie wodne tekstu dostosowuje niedostrzegalnie prawdopodobieństwa próbkowania następnego tokena; znaki wodne obrazów/wideo przetrwają kompresję, przycinanie, filtry, zmiany liczby klatek. Stable Signature (Fernandez et al., ICCV 2023, arXiv:2303.15435) — dostraja dekoder latentnej dyfuzji, aby każdy output zawierał stałą wiadomość; przycięte (do 10% treści) wygenerowane obrazy wykrywane >90% przy FPR<1e-6. Kontynuacja "Stable Signature is Unstable" (arXiv:2405.07145, maj 2024) — dostrajanie usuwa znak wodny przy zachowaniu jakości. C2PA — kryptograficznie podpisany, odporny na manipulacje standard metadanych (C2PA 2.2 Explainer 2025). Znakowanie wodne i C2PA są komplementarne: metadane mogą być usunięte, ale niosą bogatsze pochodzenie; znaki wodne przetrwają transkodowanie, ale niosą mniej informacji.

**Type:** Build
**Languages:** Python (stdlib, osadzanie + wykrywanie znaku wodnego tokena)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 minutes

## Learning Objectives

- Opisać znakowanie wodne na poziomie tokena (styl SynthID-text) i mechanizm, dzięki któremu jest wykrywalne.
- Opisać Stable Signature i atak usunięcia z 2024 roku, który go złamał.
- Wymienić rolę C2PA i dlaczego jest komplementarny do znakowania wodnego.
- Opisać kluczowe ograniczenia: sygnał specyficzny dla modelu, odporność na parafrazę i ataki zachowujące znaczenie (arXiv:2508.20228).

## The Problem

Lata 2023-2024 przyniosły deepfake'i i treści generowane przez AI w kontekstach politycznych i konsumenckich na skalę. Znakowanie wodne jest proponowanym technicznym sygnałem pochodzenia: oznaczaj generacje w momencie tworzenia, wykrywaj je później. Dowody z 2025 roku: żaden znak wodny nie jest bezwarunkowo odporny, ale w połączeniu z metadanymi C2PA kombinacja zapewnia użyteczną historię pochodzenia.

## The Concept

### Znakowanie wodne tekstu (styl SynthID-text)

Mechanizm Kirchenbauer et al. 2023, zprodukcjonalizowany przez Google:

1. Na każdym kroku dekodowania, haszuj poprzednie K tokenów, aby wyprodukować pseudolosowy podział słownictwa na zbiory "zielony" i "czerwony".
2. Przechyl próbkowanie w kierunku zbioru zielonego przez dodanie δ do zielonych logitów.
3. Generacja zawiera więcej zielonych tokenów niż przypadek by wyprodukował.

Wykrywanie: ponownie haszuj każdy prefiks, policz zielone tokeny w generacji, oblicz z-score. Z-score >0 dla tekstu ze znakiem wodnym, ~0 dla tekstu ludzkiego.

Właściwości:
- Niedostrzegalne dla czytelników (δ jest wystarczająco małe, że strata jakości jest niewielka).
- Wykrywalne z dostępem do funkcji podziału słownictwa.
- Nieodporne na parafrazę — przepisanie tekstu niszczy sygnał.

SynthID-text jest open-source'owy od października 2024 przez Responsible GenAI Toolkit Google.

### Stable Signature (obraz)

Fernandez et al. ICCV 2023. Dostrój dekoder latentnej dyfuzji, aby każdy wygenerowany obraz zawierał stałą binarną wiadomość osadzoną w latentnej reprezentacji. Wykrywanie jest dekodowane z latentnej za pomocą neuronowego dekodera. Przycięte (do 10% treści) obrazy wykrywane >90% przy FPR<1e-6.

Maj 2024 "Stable Signature is Unstable" (arXiv:2405.07145): dostrajanie dekodera usuwa znak wodny przy zachowaniu jakości obrazu. Adwersarialne dostrajanie po generacji jest tanie; odporność adwersarialna znaku wodnego jest ograniczona.

### Ujednolicony detektor SynthID (listopad 2025)

Wraz z Gemini 3 Pro: wielomedialny detektor, który czyta sygnały SynthID z tekstu, obrazu, audio i wideo w jednym API. Unifikuje stos pochodzenia Google.

### C2PA

Coalition for Content Provenance and Authenticity. Kryptograficznie podpisany, odporny na manipulacje standard metadanych. C2PA 2.2 Explainer (2025). Manifest C2PA rejestruje twierdzenia o pochodzeniu (kto stworzył, kiedy, jakie transformacje) podpisane kluczem twórcy.

Komplementarny do znakowania wodnego:
- Metadane mogą być usunięte; znaki wodne nie (łatwo).
- Metadane są bogate (pełny łańcuch pochodzenia); znaki wodne niosą bity.
- C2PA zależy od adopcji platform; znaki wodne osadzają się automatycznie.

Google integruje oba w Search, Ads i "About this image."

### Ograniczenia

- **Specyficzność modelu.** SynthID oznacza generacje z modeli obsługujących SynthID. Generacja z modelu bez SynthID nie jest oznaczona, więc "brak sygnału SynthID" nie jest dowodem autentyczności.
- **Parafraza.** Znaki wodne tekstu nie przetrwają parafrazy zachowującej znaczenie.
- **Ataki transformacyjne.** arXiv:2508.20228 (2025) pokazuje ataki zachowujące znaczenie, które niszczą zarówno znaki wodne tekstu, jak i wiele znaków wodnych obrazów.
- **Usunięcie przez dostrajanie.** Według "Stable Signature is Unstable," dostrajanie po generacji usuwa osadzone znaki wodne.

### EU AI Act Artykuł 50

Kodeks Przejrzystości dla oznaczania treści generowanych przez AI (pierwszy projekt grudzień 2025, drugi projekt marzec 2026, oczekiwany finał czerwiec 2026 zgodnie ze [stroną statusu Komisji Europejskiej](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)). Kodeks pozostaje w fazie projektu według stanu na kwiecień 2026, a harmonogram może ulec zmianie. Regulacyjna warstwa wymagająca warstwy technicznej. Deepfake'i muszą być oznakowane.

### Miejsce w Fazie 18

Lekcje 22-23 dotyczą tego, co model emituje (prywatne dane, sygnał pochodzenia). Lekcja 27 obejmuje zarządzanie danymi treningowymi. Lekcja 24 to ramy regulacyjne wymagające tych środków technicznych.

## Use It

`code/main.py` buduje zabawkowy znak wodny tekstu. Tokeny to liczby całkowite 0..N-1; próbkowanie ze znakiem wodnym przechyla w kierunku zbioru zielonego zdefiniowanego przez hasz. Detektor oblicza z-score zielonych tokenów. Możesz zaobserwować wykrywanie przy generacjach 1000 tokenów, zobaczyć, jak parafraza niszczy sygnał i zmierzyć wskaźnik fałszywie pozytywnych na tekście ludzkim.

## Ship It

Ta lekcja produkuje `outputs/skill-provenance-audit.md`. Dla wdrożenia treści z twierdzeniem o pochodzeniu, audytuje: mechanizm znaku wodnego (jeśli istnieje), łańcuch podpisu C2PA (jeśli istnieje), odporność adwersarialną każdego i pokrycie na modalność.

## Exercises

1. Uruchom `code/main.py`. Raportuj z-score dla oznakowanej generacji 1000 tokenów vs tekstu napisanego przez człowieka. Zidentyfikuj wskaźnik fałszywie pozytywnych na progu ufności 95%.

2. Zaimplementuj atak parafrazy, który zastępuje 30% tokenów synonimami. Ponownie zmierz z-score.

3. Przeczytaj Kirchenbauer et al. 2023 Sekcja 6 o odporności. Dlaczego znaki wodne tekstu zawodzą pod parafrazą, ale znaki wodne obrazów przetrwają przycinanie?

4. Zaprojektuj wdrożenie używające SynthID-text + metadane C2PA. Opisz łańcuch pochodzenia, który widzi konsument. Zidentyfikuj jeden tryb awarii każdego komponentu.

5. Wynik "Stable Signature is Unstable" z 2024 pokazuje, że dostrajanie usuwa znak wodny obrazu. Zaprojektuj kontrolę wdrożenia, która ogranicza ten atak — na przykład wymagaj podpisanych wydań dostrojonych punktów kontrolnych.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| SynthID | "znak wodny Google" | Międzymodalny sygnał pochodzenia; tekst, obraz, audio, wideo |
| Znak wodny tokena | "styl Kirchenbauera" | Znak wodny tekstu przez próbkowanie z przechyleniem, wykrywalny przez z-score zielonych tokenów |
| Stable Signature | "znak wodny obrazu" | Znak wodny przez dostrojony dekoder; ICCV 2023 |
| C2PA | "standard metadanych" | Kryptograficznie podpisane, odporne na manipulacje metadane pochodzenia |
| Odporność na parafrazę | "czy przeformułowanie łamie go" | Właściwość znaku wodnego tekstu; obecnie ograniczona |
| Usunięcie przez dostrajanie | "adwersarialne usunięcie znaku" | Atak usuwający znak wodny obrazu przez dostrajanie dekodera |
| Detektor międzymodalny | "ujednolicony SynthID" | Ujednolicone API z listopada 2025 między modalnościami |

## Further Reading

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — mechanizm znaku wodnego tokena
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — artykuł o znaku wodnym obrazu
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — atak usunięcia
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — międzymodalny znak wodny
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — standard metadanych
