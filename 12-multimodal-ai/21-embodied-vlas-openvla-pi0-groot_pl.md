# Ucieleśnione VLA: RT-2, OpenVLA, π0, GR00T

> Po raz pierwszy model odczytał przepis ze strony internetowej i wykonał go w kuchennym robocie — to było RT-2 (Google DeepMind, lipiec 2023). RT-2 dyskretyzował akcje jako tokeny tekstowe, wspólnie dostrajał VLM na danych internetowych oraz danych z akcji robotów, udowadniając, że wiedza wizualno-językowa z sieci przenosi się na sterowanie robotyczne. OpenVLA (czerwiec 2024) dostarczył otwartą 7B referencję. Seria π0 od Physical Intelligence (2024–2025) dodała ekspertów akcji opartych na dopasowywaniu przepływu (flow-matching). GR00T N1 od NVIDIA (marzec 2025) wprowadził dwusystemowe sterowanie (System 1 / System 2) dla humanoidalnych robotów na dużą skalę. Prymityw VLA — wizja-język-akcja, pojedynczy model, który widzi, czyta i działa — jest pomostem między modelami rozumienia z tej fazy a systemami autonomicznymi z Fazy 15.

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 minutes

## Learning Objectives

- Opisać tokenizację akcji: kodowanie dyskretnych pojemników (RT-2), wydajne tokeny akcji FAST, ciągłe akcje z dopasowywaniem przepływu (π0).
- Wyjaśnić, dlaczego wspólne dostrajanie na danych internetowych i robotycznych zachowuje transfer ogólnej wiedzy do nowych zadań.
- Porównać OpenVLA (otwarta 7B Llama+VLM), π0 (dopasowywanie przepływu) i GR00T N1 (dwusystemowy) na tym samym zadaniu robotycznym.
- Wymienić zbiór danych Open X-Embodiment i jego rolę jako korpusu treningowego RT-X.

## The Problem

Robot wykonujący prace domowe na podstawie instrukcji w języku naturalnym jest celem badawczym od lat 70. XX wieku. Odpowiedź lat 20. XXI wieku: model wizja-język-akcja (VLA). Ta sama architektura VLM używana do VQA, ale wyjściem są akcje (momenty obrotowe w stawach, pozycje efektora końcowego, polecenia dyskretne) zamiast tekstu.

Wyzwania specyficzne dla VLA:

1. Przestrzenie akcji są ciągłe (kąty w stawach, siły) i wielowymiarowe (7-DOF ramię + 3-DOF chwytak = 10 wymiarów przy 30 Hz).
2. Dane treningowe specyficzne dla robotów są skąpe. Open X-Embodiment ma ~1 mln trajektorii; dane tekst-obraz z sieci to 5 mld+.
3. Częstotliwość sterowania ma znaczenie. Pętla sterowania 30 Hz oznacza budżet 33 ms na akcję.
4. Bezpieczeństwo. Błędna akcja uszkadza sprzęt, ludzi lub mienie.

## The Concept

### Tokenizacja akcji (RT-2)

Sztuczka RT-2: reprezentuj każdy cel stawu jako skwantowany token tekstowy. Zdyskretyzuj znormalizowany zakres [-1, 1] do 256 pojemników, mapując każdy pojemnik na ID słownika. 10-DOF akcja staje się 10 tokenami w każdym kroku sterowania.

Wspólne dostrajanie VLM PaLM-X na mieszance:

- Pary obraz-tekst z sieci (opisowe, VQA).
- Demonstracje robotyczne, akcja jako tokeny.

Model widzi "podnieś czerwoną kostkę" (język) → obraz (wizja) → 10-tokenową sekwencję akcji (zdyskretyzowane cele stawów). Trening wstępny na danych internetowych zachowuje transfer ogólnej wiedzy: RT-2 może wykonać "przesuń się w stronę szybko poruszającego się obiektu", mimo że "szybko poruszający się" nie występuje w danych treningowych.

Inferencja przy 3–5 Hz w publikacji RT-2, ograniczona przez autoregresyjne dekodowanie VLM.

### OpenVLA — otwarta 7B referencja

OpenVLA (Kim i in., czerwiec 2024) jest odpowiednikiem RT-2 z otwartymi wagami. 7B Llama jako szkielet, podwójny enkoder wizyjny DINOv2 + SigLIP, tokenizacja akcji w 256 pojemnikach.

Trained on Open X-Embodiment (970k trajektorii z 22 robotów). Dostarczany z obsługą dostrajania LoRA do adaptacji na nowych robotach.

Inferencja: 4–5 Hz na A100 z kwantyzacją. Wystarczająco szybka do powolnej manipulacji, niewystarczająca do sterowania wysokoczęstotliwościowego.

### Tokenizer FAST — szybsze dekodowanie akcji

Pertsch i in. (2024) wykazali, że tokenizacja dyskretnych pojemników jest nieefektywna — większość akcji skupia się w małym regionie przestrzeni pojemników. FAST (Frequency-domain Action Sequence Tokenizer) kompresuje sekwencje akcji za pomocą DCT i kwantyzuje współczynniki.

30-krokowa trajektoria akcji staje się ~10 tokenami FAST zamiast 300 tokenów dyskretnych pojemników. Inferencja przyspiesza 3–5x bez utraty jakości.

### π0 i akcje z dopasowywaniem przepływu

π0 od Physical Intelligence (Black i in., październik 2024) zastępuje dyskretne tokeny akcji ekspertem akcji opartym na dopasowywaniu przepływu:

- Mały transformer akcji odczytuje ukryte stany VLM i wyprowadza ciągłą 50-krokową sekwencję akcji przez prostowany przepływ.
- Głowa akcji trenuje ze stratą dopasowywania przepływu; trening wstępny VLM pozostaje niezmieniony.
- Inferencja: pełna sekwencja akcji emitowana w ~5 krokach denoisingu, efektywnie 50 Hz sterowania.

Twierdzenie π0: pokonuje OpenVLA i Octo w szerokim zestawie zadań manipulacyjnych. Ciągła formuła akcji zachowuje gładkość, którą dyskretyzacja niszczy.

π0.5 i π0-FAST to przyrostowe ulepszenia. π0-FAST łączy tokenizację FAST z dopasowywaniem przepływu.

### GR00T N1 — system dwusystemowy dla humanoidów

GR00T N1 od NVIDIA (marzec 2025) jest zbudowany dla robotów humanoidalnych (>30 DOF, całe ciało):

- System 2: duży VLM odczytujący scenę + instrukcję, produkujący wysokopoziomowe podcele przy ~1 Hz.
- System 1: mały transformer głowy akcji produkujący niskopoziomowe polecenia stawów przy 50–100 Hz, warunkowane podcelami.

Podział odwzorowuje szybkie i wolne myślenie Kahnemana: System 2 planuje, System 1 działa. Korzyści: powolne planowanie wielkości VLM nie blokuje szybkiego sterowania; System 1 pozostaje mały dla niskiego opóźnienia.

GR00T N1.7 (koniec 2025) poprawia skalowanie danych. GR00T dostraja się z danymi sym-do-rzeczywistości z Omniverse.

### Open X-Embodiment

Dane treningowe. RT-X (październik 2023) zgromadził 22 zestawy danych obejmujące 1 mln trajektorii z 22 robotów. Open X-Embodiment to korpus, którego wszyscy używają:

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table.
- Każda próbka: (stan robota, widoki z kamer, instrukcja, sekwencja akcji).
- Higiena treningowa: ujednolicenie przestrzeni akcji, normalizacja zakresów stawów, skalowanie kamer.

OpenVLA i π0 trenują na Open X-Embodiment. Luka domenowa do konkretnego robota jest zamykana przez dostrajanie LoRA na 100–1000 demonstracji specyficznych dla zadania.

### Wspólne dostrajanie vs tylko robotyczne

Wspólne dostrajanie miesza dane VQA z sieci z trajektoriami robotów. Proporcja ma znaczenie: zbyt dużo VQA i model zapomina akcje; zbyt dużo danych robotycznych i model traci ogólną wiedzę.

Proporcja RT-2: ~1:1. OpenVLA: ~0.5:1 sieć-do-robota. π0: podobnie. Dokładna proporcja jest hiperparametrem do dostrojenia na podstawie rozmiaru zestawu danych.

Trenowanie tylko na danych robotycznych produkuje modele specyficzne dla zadania, które zawodzą na instrukcjach spoza dystrybucji. Wspólne dostrajanie to różnica między "podnieś czerwoną kostkę (w demonstracji)" a "podnieś trzeci co do wielkości obiekt od lewej (nowe sformułowanie)."

### Bezpieczeństwo i ograniczenia akcji

Każdy produkcyjny VLA jest dostarczany z:

- Twardymi ograniczeniami stawów (nie można przekroczyć momentu obrotowego poza specyfikacją).
- Ograniczeniami prędkości (miękkie przycinanie).
- Granicami przestrzeni roboczej (efektor końcowy nie może opuścić stołu).
- Zatwierdzeniem z udziałem człowieka dla nowych zadań.

Te elementy znajdują się poza VLA jako kontrole warstwy sterowania. Wynik VLA jest sugestią, nie poleceniem.

## Use It

`code/main.py`:

- Implementuje tokenizację i detokenizację akcji w 256 pojemnikach.
- Szkicuje tokenizer FAST oparty na DCT + kwantyzacji.
- Porównuje liczbę tokenów na krok akcji w (dyskretne pojemniki, FAST, ciągły przepływ).
- Wypisuje podsumowanie linii rozwojowej RT-2 → OpenVLA → π0 → GR00T.

## Ship It

Ta lekcja produkuje `outputs/skill-vla-action-format-picker.md`. Dla zadania robotycznego (manipulacja, nawigacja, całe ciało humanoida) wybiera między dyskretnymi pojemnikami + RT-2, FAST + OpenVLA, dopasowywaniem przepływu + π0 lub systemem dwusystemowym + GR00T.

## Exercises

1. 10-DOF ramię przy częstotliwości sterowania 30 Hz. Tokenizacja dyskretnych pojemników w 256 pojemnikach emituje ile tokenów na sekundę? Czy 7B VLM nadąża?

2. Tokenizacja FAST kompresuje 30-krokowe trajektorie do ~10 tokenów. Co użytkownik traci, jeśli trajektoria zawiera ruch wysokoczęstotliwościowy (np. bębnienie)?

3. Głowa dopasowywania przepływu π0 denoizuje w ~5 krokach. Porównaj przepustowość z autoregresyjnym dekodowaniem OpenVLA przy 4–5 Hz.

4. Podział System 1 / System 2 GR00T odwzorowuje Kahnemana. Zaproponuj inny podział (System 3?), który mógłby pomóc w chodzeniu dwunożnym.

5. Przeczytaj Sekcję 4 Open X-Embodiment dotyczącą kuratacji zbioru danych. Wymień trzy zasady kuratacji, które zapobiegają wyciekowi domenowemu.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| VLA | "Vision-language-action" | Model, który przyjmuje obraz + instrukcję i wyprowadza polecenia akcji |
| Action tokenization | "Discrete bins" | Kwantyzacja ciągłych celów stawów do 256 pojemników na wymiar, każdy jako ID słownika |
| FAST tokenizer | "Frequency action tokens" | DCT + kwantyzacja do kompresji 30-krokowych trajektorii do ~10 tokenów |
| Co-fine-tune | "Mix web + robot" | Trenowanie na danych VQA z sieci wraz z demonstracjami robotów, aby zachować ogólną wiedzę |
| Flow-matching action head | "π0 continuous output" | Mały transformer wyprowadzający 50-krokową sekwencję akcji przez prostowany przepływ |
| System 1 / System 2 | "Dual-system control" | Duży VLM planuje wolno, mała głowa akcji działa szybko; wzorzec GR00T |
| Open X-Embodiment | "RT-X dataset" | Zbiór danych 1 mln trajektorii z wielu robotów; korpus treningowy |

## Further Reading

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)