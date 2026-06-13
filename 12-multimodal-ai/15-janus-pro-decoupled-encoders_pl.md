# Janus-Pro: Rozdzielone Enkodery dla Ujednoliconych Modeli Multimodalnych

> Ujednolicone modele multimodalne mają nieuniknione napięcie. Rozumienie potrzebuje cech semantycznych — wektory wyjściowe SigLIP lub DINOv2 bogate w informacje na poziomie pojęć. Generacja potrzebuje kodów przyjaznych rekonstrukcji — tokenów VQ, które składają się z powrotem w ostre piksele. Te dwa cele nie są kompatybilne w jednym enkoderze. Janus (DeepSeek, październik 2024) i Janus-Pro (DeepSeek, styczeń 2025) argumentują, że rozwiązaniem jest zaprzestanie prób: rozdziel dwa enkodery. Współdziel ciało transformera między zadaniami, ale kieruj rozumienie przez SigLIP, a generację przez tokenizator VQ. Przy 7B, Janus-Pro pokonuje DALL-E 3 na GenEval, jednocześnie dorównując LLaVA na MMMU. Ta lekcja czyta, dlaczego dwa enkodery działają tam, gdzie jeden zawodzi.

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## Learning Objectives

- Wyjaśnij, dlaczego pojedynczy współdzielony enkoder kompromituje albo jakość rozumienia, albo jakość generacji.
- Opisz routing Janus-Pro: cechy SigLIP po stronie wejściowej dla rozumienia, tokeny VQ zarówno na wejściu, jak i wyjściu dla generacji.
- Prześledź skalowanie mieszanki danych, które sprawia, że Janus-Pro odnosi sukces tam, gdzie Janus nie.
- Porównaj architektury rozdzielonych enkoderów (Janus-Pro), sprzężonych ciągłych (Transfusion) i sprzężonych dyskretnych (Show-o).

## The Problem

Ujednolicone modele współdzielą ciało transformera między rozumieniem a generacją. Poprzednie próby (Chameleon, Show-o, Transfusion) wszystkie używają jednego tokenizatora wizyjnego dla obu kierunków. Tokenizator jest kompromisem:

- Zoptymalizowany dla rekonstrukcji (generacja): VQ-VAE wychwytuje drobnoziarniste szczegóły pikseli, ale produkuje tokeny o słabej spójności semantycznej.
- Zoptymalizowany dla semantyki (rozumienie): osadzenia SigLIP grupują obrazy "kota" blisko tokenów "kota", ale nie pozwalają na dobrą rekonstrukcję.

Show-o i Transfusion płacą za to widocznym podatkiem jakościowym na jednym kierunku. Janus-Pro pyta: dlaczego wymagać jednego tokenizatora, skoro zadania mają różne potrzeby?

## The Concept

### Rozdzielone kodowanie wizyjne

Architektura Janus-Pro rozdziela dwa enkodery:

- Ścieżka rozumienia. Obraz wejściowy → SigLIP-SO400m → 2-warstwowy MLP → ciało transformera.
- Ścieżka generacji. Obraz wejściowy (jeśli warunkowanie na istniejącym obrazie) → tokenizator VQ → ID tokenów → ciało transformera.
- Generacja wyjścia. Tokeny obrazkowe przewidziane przez transformer → dekoder VQ → piksele.

Ciało transformera jest współdzielone. Wszystko powyżej i poniżej ciała jest specyficzne dla zadania.

Wejścia są rozróżniane przez format promptu: znacznik `<understand>` kieruje przez SigLIP; `<generate>` kieruje przez VQ. Lub routing jest domyślny z zadania.

### Dlaczego to działa

Koszt rozumienia otrzymuje cechy SigLIP, które pre-trening w stylu CLIP dostroił do podobieństwa semantycznego. Benchmarki percepcji modelu poprawiają się względem Show-o/Transfusion, ponieważ cechy wejściowe są lepsze dla zadania.

Koszt generacji otrzymuje tokeny VQ, które tokenizator dostroił do rekonstrukcji. Jakość obrazu poprawia się względem Show-o, ponieważ kody VQ składają się z powrotem na piksele czysto.

Współdzielone ciało transformera widzi dwie dystrybucje wejściowe (SigLIP i VQ) i uczy się pracować z obiema. Twierdzenie: wystarczająco danych + wystarczająco parametrów, ciało absorbuje przełączanie.

### Skalowanie danych — Janus vs Janus-Pro

Janus (oryginalny, arXiv 2410.13848) wprowadził rozdzielenie, ale w małej skali (1.3B parametrów, ograniczone dane). Janus-Pro (arXiv 2501.17811) przeskalował:

- 7B parametrów (vs 1.3B).
- 90M par obraz-tekst dla etapu 1 (dopasowanie) w górę z 72M.
- 72M dla etapu 2 (ujednolicenie) w górę z 26M.
- Dodano 200k próbek instrukcji generacji obrazów dla etapu 3.

Konsekwencja: Janus-Pro-7B dorównuje LLaVA na MMMU (60.3 vs ~58) i pokonuje DALL-E 3 na GenEval (0.80 vs 0.67). Jeden otwarty model, konkurencyjny po obu stronach ujednoliconego spektrum.

### JanusFlow — wariant z dopasowaniem przepływu

JanusFlow (arXiv 2411.07975) zamienia ścieżkę generacji VQ na ścieżkę generacji z dopasowaniem przepływu (ciągłą). Podział staje się SigLIP-dla-rozumienia + dopasowanie-przepływu-dla-generacji. Sufity jakości podnoszą się dalej. Architektura pozostaje rozdzielone-enkodery-współdzielone-ciało.

### Zadanie współdzielonego ciała

Ciało transformera przetwarza ujednoliconą sekwencję, ale z dwiema dystrybucjami wejściowymi. Jego zadaniem jest:

- Dla rozumienia: konsumuj cechy SigLIP + tokeny tekstowe → emituj tekst autogresywnie.
- Dla generacji: konsumuj tokeny tekstowe + (opcjonalne VQ tokeny obrazu) → emituj VQ tokeny obrazu autogresywnie.

Ciało nie ma wag specyficznych dla modalności na blok. To transformer w stylu tekstowym, jakiego spodziewałbyś się wewnątrz Qwen lub Llamy, plus dwa adaptery wejściowe.

Co ciekawe, oznacza to, że ciało Janus-Pro mogłoby być zainicjalizowane z pretrenowanego LLM-a. Janus-Pro rzeczywiście inicjalizuje z DeepSeek-MoE-7B. Ten wybór ma znaczenie: LLM wnosi zdolność rozumowania, której czysto-od-zera ujednolicone modele mają trudność osiągnąć.

### W porównaniu z InternVL-U

InternVL-U (Lekcja 12.10) to kontynuacja z 2026 roku. Łączy:

- Rodzimy multimodalny pre-trening (szkielet InternVL3).
- Routing rozdzielonych enkoderów (SigLIP na wejściu, głowy VQ + dyfuzji na wyjściu).
- Ujednolicone rozumienie + generacja + edycja.

InternVL-U wchłania wybór architektoniczny Janus-Pro w większą ramę. Pomysł rozdzielonych enkoderów jest teraz domyślny dla ujednoliconych modeli w skali.

### Ograniczenia

Rozdzielone enkodery dodają złożoności architektonicznej. Dwa tokenizatory do wytrenowania, dwie ścieżki wejściowe do utrzymania, dwa zestawy trybów awarii. Dla produktów, które nie potrzebują generacji, Janus-Pro jest przekonstruowany — wybierz model rozumienia z rodziny LLaVA.

Dla produktów, które nie potrzebują rozumienia, Janus-Pro jest nadmiernie wykwalifikowany — wybierz model Stable Diffusion 3 / Flux.

Dla produktów, które potrzebują obu, Janus-Pro jest teraz referencyjną otwartą architekturą.

## Use It

`code/main.py` symuluje routing Janus-Pro:

- Dwa atrapy enkodery: SigLIP-podobny (produkuje 256-wymiarowe wektory semantyczne) i VQ-podobny (produkuje kody całkowite).
- Router promptów, który wybiera enkoder na podstawie znacznika zadania.
- Współdzielone ciało (zastępnik), które przetwarza sekwencje tokenów niezależnie od tego, który enkoder je wyprodukował.
- Przełącznik z etapu 1 (dopasowanie) na etap 3 (dostrajanie instrukcji) z ważonym harmonogramem samplowania.

Wypisz routowane ścieżki dla 3 przykładów: QA obrazu, T2I, edycja obrazu.

## Ship It

Ta lekcja produkuje `outputs/skill-decoupled-encoder-picker.md`. Na podstawie produktu, który chce ujednoliconej generacji + rozumienia na granicznej jakości, wybiera Janus-Pro, JanusFlow lub InternVL-U z konkretnym zaleceniem skali danych.

## Exercises

1. Janus-Pro-7B pokonuje DALL-E 3 na GenEval. Wyjaśnij, dlaczego otwarty model 7B może dorównać granicznemu zastrzeżonemu modelowi na generacji, ale nie na rozumieniu.

2. Zaimplementuj funkcję routera: mając tekst promptu, sklasyfikuj jako `understand` lub `generate`. Jak obsługujesz niejednoznaczne prompty, takie jak "opisz, a następnie naszkicuj"?

3. JanusFlow zastępuje ścieżkę VQ dopasowaniem przepływu. Co teraz wyprowadza ciało transformera i co zmienia się w koszcie?

4. Zaproponuj czwarte zadanie, które architektura Janus-Pro mogłaby obsłużyć z jednym dodatkowym rozdzielonym enkoderem. Przykłady: segmentacja obrazu (styl DINO), głębokość (styl MiDaS).

5. Przeczytaj Sekcję 4.2 artykułu Janus-Pro o skalowaniu danych. Który etap danych przyczynia się najbardziej do poprawy jakości T2I w porównaniu z Janusem?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Rozdzielone enkodowanie | "Dwa enkodery wizyjne" | Osobny tokenizator lub enkoder na kierunek: semantyczny dla rozumienia, rekonstrukcyjny dla generacji |
| Współdzielone ciało | "Jeden transformer" | Pojedynczy transformer przetwarza wyniki obu enkoderów; brak wag specyficznych dla modalności |
| SigLIP dla rozumienia | "Cechy semantyczne" | Wieża wizyjna z rodziny CLIP zapewniająca bogate cechy konceptualne, ale słabą rekonstrukcję |
| VQ dla generacji | "Kody rekonstrukcji" | Tokeny z kwantyzacją wektorową, które dekodują się czysto z powrotem na piksele |
| JanusFlow | "Wariant z dopasowaniem przepływu" | Janus-Pro z ciągłą głową generacji przez dopasowanie przepływu zamiast VQ |
| Znacznik routingu | "Znacznik zadania" | Marker promptu (`<understand>` / `<generate>`), który wybiera enkoder wejściowy |

## Further Reading

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)