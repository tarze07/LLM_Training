# Rodzina Qwen-VL i Dynamiczne FPS w Wideo

> Rodzina Qwen-VL — Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025), Qwen3-VL (2025) — to najbardziej wpływowa linia otwartych modeli wizyjno-językowych w 2026. Każda generacja postawiła jeden decydujący zakład architektoniczny, który reszta otwartego ekosystemu skopiowała w ciągu dwunastu miesięcy: natywna dynamiczna rozdzielczość przez M-RoPE, próbkowanie dynamicznego FPS z absolutnym dopasowaniem czasu, uwaga okienna w ViT i ustrukturyzowane formaty wyjścia agenta. Do Qwen3-VL przepis ustabilizował się: enkoder 2D-RoPE-ViT z wejściami o natywnych proporcjach, projektor MLP do dużej bazy językowej Qwen3 i etapy treningu kładące nacisk na OCR, ugruntowanie i zachowanie agenta jako cele pierwszej klasy. Ta lekcja czyta rodzinę chronologicznie, abyś zrozumiał, dlaczego każde pokrętło jest tam, gdzie jest.

**Type:** Learn
**Languages:** Python (stdlib, enkoder M-RoPE + sampler dynamicznego FPS)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 minutes

## Learning Objectives

- Obliczyć trójosiowe rotacje M-RoPE (czasowe, wysokość, szerokość) i wyjaśnić, dlaczego wszystkie trzy są potrzebne.
- Wybrać strategię próbkowania dynamicznego FPS dla wideo i rozumować o tokenach-na-sekundę vs dokładności wykrywania zdarzeń.
- Wymienić cztery generacyjne ulepszenia Qwen-VL w kolejności i co każde umożliwiło.
- Podłączyć format wyjścia agenta JSON w stylu Qwen2.5-VL i parsować ustrukturyzowane wywołania narzędzi z odpowiedzi VLM.

## The Problem

Qwen-VL pojawił się w sierpniu 2023 jako bezpośrednia odpowiedź na LLaVA-1.5 i BLIP-2. Luka, którą zespół Qwen chciał zaadresować, była trójwymiarowa: rozdzielczość, wideo i ustrukturyzowane wyjście.

Rozdzielczość: LLaVA-1.5 działał przy 336x336. Dobre dla zdjęć, bezużyteczne dla chińskiej faktury lub gęstego zrzutu arkusza kalkulacyjnego. Pierwszą innowacją Qwen-VL było 448x448 i ugruntowane wyjście obwiedni, pozwalające modelowi wskazywać rzeczy.

Wideo: Video-LLaMA układał enkodery na klatkę i karmił nimi LLM. Działało dla krótkich klipów, nie dla wielominutowych filmów, gdzie oś czasu jest sygnałem. Zespół Qwen chciał pojedynczego enkodera rozumiejącego czas.

Ustrukturyzowane wyjście: LLaVA emitował dowolny tekst. Agent potrzebuje JSON. Qwen-VL trenował na jawnych formatach wyjścia JSON, w tym współrzędnych obwiedni jako tekstu.

Każda generacja Qwen-VL rozszerza jedną z tych trzech osi.

## The Concept

### Qwen-VL (sierpień 2023)

Pierwsza generacja: enkoder OpenCLIP ViT-bigG/14 (2.5B parametrów), Q-Former kompatybilny z Llamą (1 krok z 256 zapytaniami), baza Qwen-7B. Wkłady:

- Rozdzielczość 448x448 (wówczas SOTA dla otwartego VLM).
- Ugruntowanie: trenowany na parach obraz-tekst z jawnym wyjściem tokenów współrzędnych. "Kot jest w <box>(112, 204), (280, 344)</box>".
- Trening wielojęzyczny chiński + angielski od samego początku.

Benchmarki w tamtym czasie: konkurencyjny z GPT-4V po angielsku, dominujący po chińsku. Nadzór ugruntowania był prawdziwym nagłówkiem.

### Qwen2-VL (wrzesień 2024) — M-RoPE i natywna rozdzielczość

Qwen2-VL zastąpił stos stałej rozdzielczości + Q-Former natywnie dynamicznym enkoderem ViT. Kluczowe zmiany:

- Natywna dynamiczna rozdzielczość. ViT akceptuje dowolne HxW podzielne przez 28 (paczka 14 z 2x scaleniem przestrzennym). Obraz 1120x672 (40x24 scalonych paczek) produkuje 960 tokenów wizualnych. Żadnego skalowania, żadnego kafelkowania, żadnej miniatury.
- M-RoPE (Multimodalne RoPE). Każdy token niesie 3D pozycję (t, h, w) zamiast 1D. Dla obrazów t=0, dla wideo t = indeks_klatki. RoPE obraca wektory zapytań/kluczy przez częstotliwość na oś. Żadnej tablicy osadzeń pozycyjnych.
- Projektor MLP. Odrzuć Q-Former; użyj 2-warstwowego MLP na scalonych tokenach paczek.
- Wideo z dynamicznym FPS. Wideo próbkowane domyślnie przy 1-2 FPS, ale model akceptuje dowolną liczbę klatek.

Wynik: Qwen2-VL-7B dorównał GPT-4o na kilku multimodalnych benchmarkach i pobił go na DocVQA (94.5 vs 88.4). Zmiana architektury była decydującym ruchem.

### Qwen2.5-VL (luty 2025) — dynamiczne FPS + absolutny czas

Dużą zmianą Qwen2.5-VL było wideo. Dynamiczne FPS to nie tylko "próbkuj więcej klatek, gdy potrzebne." Praca sformalizowała:

- Absolutne tokeny czasu. Zamiast indeksów pozycyjnych (klatka 0, 1, 2...), użyj rzeczywistych znaczników czasu. "O 0:04 kot skacze." Model widzi tokeny `<time>0.04</time>` przeplatane z tokenami klatek.
- Dynamiczne FPS. Próbkuj przy 1 FPS dla wolnego materiału, 4+ FPS dla akcji. Użytkownik lub trener wybiera; M-RoPE się dostosowuje.
- Uwaga okienna w ViT. Uwaga przestrzenna jest okienna (lokalna w blokach) dla przepustowości; globalna uwaga co kilka warstw.
- Wyraźny format wyjścia JSON. Trenowany na danych wywołań narzędzi: "{\"tool\": \"click\", \"coords\": [380, 220]}". Gotowy dla agenta po wyjęciu z pudełka.
- Skalowanie MRoPE-v2. Pozycje skalują się z maksymalnym rozmiarem wejścia, więc 10-minutowe wideo nie wychodzi poza zakres częstotliwości.

Benchmarki: Qwen2.5-VL-72B bije GPT-4o na większości benchmarków wideo, dorównuje Gemini 2.0 na dokumentach i ustanawia SOTA otwartych modeli dla ugruntowania GUI (ScreenSpot: 84% dokładności vs 38% dla GPT-4o).

### Qwen3-VL (listopad 2025)

Qwen3-VL to przyrostowa aktualizacja, która konsoliduje, a nie wymyśla na nowo: większy szkielet LLM (Qwen3-72B), rozszerzone dane treningowe, ulepszone OCR, silniejsze rozumowanie przez "tryb myślenia" Qwen3. ViT i M-RoPE pozostają. Praca skupia się na danych i ulepszeniach treningu nad architekturą.

Wniosek z linii: do 2025 architektura Qwen-VL ustabilizowała się. Dodatkowe generacje skalują moc obliczeniową i dane, nie prymitywy.

### M-RoPE matematycznie

Klasyczne RoPE obraca zapytanie `q` o wymiarze `d` przez pozycję `m` używając sparowanych współrzędnych:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE dzieli ukryty wymiar na trzy pasma. Powiedzmy `d = 96`. Przypisz 32 wymiary do czasu, 32 do wysokości, 32 do szerokości. Każde pasmo obraca się według swojej własnej pozycji osi. Paczka na pozycji (t=5, h=10, w=20) otrzymuje rotacje `R_t(5)`, `R_h(10)`, `R_w(20)` zastosowane do swoich trzech pasm.

Tokeny tekstowe używają `t = indeks_tekstu, h = 0, w = 0` (lub znormalizowany wybór), zachowując kompatybilność. Klatki wideo używają `t = czas_klatki, h = wiersz, w = kolumna`. Pojedyncze obrazy używają `t = 0`.

Korzyść: jedno kodowanie pozycji obsługuje tekst, obraz i wideo bez rozgałęzień kodu lub różnych tablic pozycji.

### Logika próbkowania dynamicznego FPS

Mając wideo o czasie trwania `T` sekund i docelowy budżet tokenów `B`:

1. Oblicz maksymalne FPS, na które możesz sobie pozwolić: `fps_max = B / (T * tokens_per_frame)`.
2. Wybierz docelowe FPS z {1, 2, 4, 8} spełniające `fps <= fps_max`.
3. Jeśli ruch jest wysoki (heurystyka przepływu optycznego lub jawne żądanie użytkownika), wybierz wyższe FPS. Jeśli ruch jest niski, wybierz niższe.
4. Próbkuj równomiernie przy wybranym FPS; wstaw tokeny `<time>t</time>` między klatkami.

Qwen2.5-VL trenuje tę logikę implicite; podczas wnioskowania użytkownik kontroluje przez parametr `fps`. 60-sekundowa sekwencja akcji przy 4 FPS z 81 tokenami na klatkę = 19440 tokenów, zarządzalne w kontekście 32k.

### Ustrukturyzowane wyjście agenta

Trening agenta Qwen2.5-VL jawnie celuje w ustrukturyzowane wywołania narzędzi:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Parsowanie jest deterministyczne: JSON.parse na wyjściu modelu. Porównaj z dowolnym "kliknij w (1024, 512)", które wymagało regex i obsługi niejednoznaczności. Zmiana jest powodem, dla którego wyniki ScreenSpot Qwen2.5-VL skoczyły z 55% Qwen2-VL do 84%.

## Use It

`code/main.py` implementuje:

- Obliczanie pozycji M-RoPE dla spakowanej sekwencji mieszającej tekst, patche obrazów i klatki wideo.
- Sampler dynamicznego FPS: mając (czas_trwania, budżet, poziom_ruchu), wybierz FPS i emituj znaczniki czasowe klatek.
- Zabawkowy parser wyjścia JSON Qwen2.5-VL, który obsługuje odpowiedzi wywołań narzędzi z polami współrzędnych.

Uruchom go, a następnie poczuj różnicę, gdy zamienisz stałe FPS na dynamiczne FPS w 5-minutowym wideo.

## Ship It

Ta lekcja produkuje `outputs/skill-qwen-vl-pipeline-designer.md`. Mając zadanie wideo (monitorowanie, agent, rozpoznawanie akcji, dostępność), emituje konfigurację Qwen2.5-VL (budżet klatek, strategia FPS, flaga uwagi okiennej, tryb wyjścia agenta) i oszacowanie opóźnienia. Użyj tego, gdy wdrażasz model rodziny Qwen-VL dla produktu wideo.

## Exercises

1. Oblicz rotacje M-RoPE dla paczki na pozycji (t=3, h=5, w=7) z ukrytym wymiarem 48 (16 na pasmo, base theta 10000). Pokaż kąty obrotu dla pierwszych trzech par w każdym paśmie.

2. 10-minutowe nagranie z kamery bezpieczeństwa przy 1 FPS produkuje ile klatek? Przy rozdzielczości 384 z 3x poolem, ile łącznie tokenów? Czy domyślny kontekst 32k Qwen2.5-VL to obsługuje?

3. Wybierz FPS dla 30-sekundowego meczu tenisa vs 30-sekundowej demonstracji przepisu vs 30-sekundowego nagrania agenta UI. Uzasadnij każdy logiką dynamicznego FPS.

4. Qwen2.5-VL całkowicie odrzuca Q-Former. Dlaczego prosty MLP działa w 2025, ale nie w 2023? (Wskazówka: skala danych i jakość enkodera.)

5. Sparsuj trzy wyjścia wywołań narzędzi JSON Qwen2.5-VL do słowników Pythona. Co zawodzi dla źle sformowanego JSON i jaką strategię odzyskiwania zaleca poradnik Qwen?

## Key Terms

| Term | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| M-RoPE | "Multimodalne RoPE" | 3D obrotowe osadzenie pozycyjne z pasmami czasu, wysokości i szerokości w ukrytym wymiarze |
| Dynamic FPS | "Inteligentne próbkowanie" | Częstotliwość próbkowania klatek wybrana na wideo na podstawie ruchu, czasu trwania i budżetu tokenów |
| Absolute time token | "Token znacznika czasu" | `<time>t</time>` wpleciony w sekwencję, aby model widział rzeczywiste sekundy, a nie indeks klatki |
| Window attention | "Uwaga lokalna" | Samouwaga przestrzenna ograniczona do małych okien dla szybkości; globalna uwaga dodawana okresowo |
| Structured agent output | "Tryb JSON" | Nadzór danych treningowych uczący VLM emitować parsowalny JSON ze współrzędnymi i nazwami narzędzi |
| min_pixels / max_pixels | "Ograniczenia rozdzielczości" | Kontrole Qwen2.5-VL na żądanie ograniczające całkowitą liczbę pikseli, a tym samym liczbę tokenów |
| Grounding | "Wskaż-to" | Wyprowadzanie współrzędnych obwiedni jako tokenów tekstowych; używane od Qwen-VL v1 |

## Further Reading

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)