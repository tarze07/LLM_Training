# Emu3: Przewidywanie Następnego Tokena dla Generacji Obrazów i Wideo

> Emu3 BAAI (Wang et al., wrzesień 2024) to wynik z 2024 roku, który powinien był zakończyć debatę między dyfuzją a autoregresją. Pojedynczy transformer typu decoder-only w stylu Llamy, trenowany wyłącznie na celu przewidywania następnego tokena, na ujednoliconym słownictwie tekstu + tokenów VQ obrazu + 3D VQ tokenów wideo, pokonuje SDXL w generacji obrazów i LLaVA-1.6 w percepcji. Żadnego kosztu CLIP. Żadnego harmonogramu dyfuzji. Classifier-free guidance jest używane na inferencji dla jakości, ale główny cel treningowy to przewidywanie następnego tokena z teacher forcingiem. Opublikowane w Nature. Ta lekcja czyta tezę Emu3 — dlaczego lepszy tokenizator plus skala to wszystko, czego potrzebujesz — i kontrastuje z podejściami dyfuzyjnymi.

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## Learning Objectives

- Wyjaśnij, dlaczego jedno-stratny cel przewidywania następnego tokena w Emu3 działa pomimo długo utrzymywanego założenia, że dyfuzja jest wymagana dla jakości obrazu.
- Opisz 3D tokenizator wideo: jak wygląda时空 VQ kodbuk, dlaczego łatki obejmują czas.
- Porównaj Emu3 ze Stable Diffusion XL pod względem (mocy obliczeniowej treningu, kosztu inferencji, sufitu jakości).
- Wymień trzy role, które ten sam model Emu3 odgrywa: Emu3-Gen (generacja obrazów), Emu3-Chat (percepcja), Emu3-Stage2 (generacja wideo).

## The Problem

Konwencjonalna mądrość do 2024 roku: generacja obrazów potrzebuje dyfuzji. Argument: dyskretne tokeny obrazkowe tracą zbyt wiele informacji, aby odtworzyć szczegóły, a autogresywne samplowanie kumuluje błędy na tysiącach tokenów. Stable Diffusion, DALL-E 3, Imagen, Midjourney wszystkie używają jakiejś formy dyfuzji. Chameleon (Lekcja 12.11) częściowo obalił to w małej skali, ale nie dorównał SDXL pod względem jakości.

Emu3 zaatakował argument bezpośrednio. Twierdzenie: lepszy tokenizator wizyjny + wystarczająca skala + koszt następnego tokena = generacja obrazów bijąca dyfuzję w tym samym modelu, który również robi percepcję.

Zakład był kontrowersyjny w momencie publikacji. Dwa lata później, rodzina unified-generation open-source (Emu3, Show-o, Janus-Pro, Transfusion) jest domyślną ścieżką dla badań; produkcyjne modele graniczne wydają się używać jakiegoś wariantu.

## The Concept

### Tokenizator Emu3

Kluczowym składnikiem jest tokenizator wizyjny. Emu3 trenuje niestandardowy tokenizator klasy IBQ (Inverse Bottleneck Quantizer, rodzina SBER-MoVQGAN) z redukcją rozdzielczości 8x8 na token. Obraz 512x512 staje się 64x64 = 4096 tokenów przy rozmiarze kodbu 32768.

To więcej niż 1024 tokenów Chameleona na obraz 512x512 przy K=8192, ale tańszy na token (mniejsze wyszukiwanie w kodbuk, prostszy kodek). Kluczowa metryka: PSNR rekonstrukcji na poziomie 30.5 dB, konkurencyjny z ciągłą przestrzenią ukrytą Stable Diffusion przy 32 dB.

Dla wideo: 3D VQ tokenizator koduje czasoprzestrzenną łatkę (4x4x4 piksele) do jednej liczby całkowitej. 4-sekundowy klip przy 8 FPS ma 32 klatki; przy 256x256 z 4-krotną redukcją przestrzenną i 4-krotną czasową, liczba tokenów wynosi (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 tokenów.

Jakość tokenizatora to sufit. Wkładem Emu3 jest częściowo "wytrenowaliśmy bardzo dobry tokenizator."

### Trening z jednym kosztem

Emu3 używa jednego celu: przewidywanie następnego tokena na wspólnym słownictwie obejmującym tokeny tekstowe, 2D tokeny obrazkowe i 3D tokeny wideo. Wagi są mnożone przez czynniki specyficzne dla modalności podczas treningu, aby zrównoważyć wkład, ale funkcja kosztu jest identyczna.

Trenowany na mieszance:
- Generacja obrazów: `<tekstowy podpis> <image> tokeny_obrazkowe </image>`
- Percepcja obrazów: `<image> tokeny_obrazkowe </image> <pytanie> tokeny_tekstowe`
- Generacja wideo: `<tekstowy podpis> <video> tokeny_wideo </video>`
- Percepcja wideo: analogicznie.
- Tylko tekst: standardowy NTP.

Model uczy się, kiedy emitować tokeny obrazkowe vs tekstowe na podstawie rozkładu danych. Generacja wyłania się z modelu przewidującego tokeny obrazkowe po znaczniku `<image>`.

### Classifier-free guidance i temperatura

Autogresywna generacja obrazów staje się znacznie lepsza z classifier-free guidance (CFG) na inferencji. Emu3 go używa: generuj dwa razy, raz z pełnym podpisem, raz z pustym podpisem, zmieszaj logity z wagą guidance (typowe 3.0-7.0). To ten sam trik CFG, którego używa dyfuzja, zapożyczony do ustawienia autogresywnego.

Temperatura ma znaczenie: zbyt wysoka — artefakty; zbyt niska — załamanie modalne. Zalecana temperatura Emu3 to 1.0 dla percepcji, 0.8 dla generacji obrazów.

### Trzy role, jeden model

Emu3 jest dostarczany jako trzy funkcjonalnie różne API, ale jeden zestaw wag:

- Emu3-Gen. Generacja obrazów. Wejście tekst, wyjście tokeny obrazkowe.
- Emu3-Chat. VQA i podpisywanie. Wejście obraz (tokeny), wyjście tekst.
- Emu3-Stage2. Generacja wideo i VQA wideo. Wejście tekst lub wideo, wyjście tekst lub wideo.

Żadnych głów specyficznych dla zadania. Tylko różne szablony promptów. Ten sam checkpoint.

### Benchmarki

Z artykułu Emu3 (wrzesień 2024):

- Generacja obrazów: pokonuje SDXL na MJHQ-30K FID (5.4 vs 5.6), GenEval ogólnie (0.54 vs 0.55 — remis statystyczny), i na równi z Deep-Eval composite.
- Percepcja obrazów: pokonuje LLaVA-1.6 na VQAv2 (75.1 vs 72.4) i mniej więcej dorównuje na MMMU.
- Generacja wideo: jakość 4-sekundowego klipu na konkurencyjnym FVD z publicznie benchmarkowanymi modelami ery Sory.

Liczby nie zawsze wygrywają — Emu3 wymienia punkt tu za punkt tam — ale twierdzenie "przewidywanie następnego tokena to wszystko, czego potrzebujesz" jest do obrony we wszystkich modalnościach.

### Koszt obliczeniowy

Emu3 był trenowany na około 300 miliardach multimodalnych tokenów z modelem 7B parametrów. Godziny GPU w przybliżeniu porównywalne z pre treningiem Llama-2-7B (2k-4k GPU-lat na krzemie klasy A100). Modele dyfuzyjne, takie jak Stable Diffusion 3, trenują w podobnych budżetach, ale potrzebują osobnych enkoderów tekstu i bardziej złożonych potoków.

Na inferencji, Emu3 jest wolniejszy niż SDXL na obraz: 4096 tokenów obrazkowych przy 30 tok/s to około 2 minuty na obraz 512x512, vs 2-5 sekund dla SDXL. Spekulacyjne dekodowanie i optymalizacja pamięci podręcznej KV zmniejszają lukę, ale jej nie zamykają. Autogresywna generacja obrazów jest obliczeniowo ciężka; to jest stały kompromis.

### Dlaczego to ma znaczenie

Głęboki wkład Emu3 jest koncepcyjny. Jeśli przewidywanie następnego tokena skaluje się, aby dorównać dyfuzji w generacji obrazów, ścieżka ujednoliconego modelu (jeden koszt, jeden szkielet, dowolna modalność) jest żywotna. Przyszłe modele nie będą potrzebować osobnych enkoderów tekstu, osobnych harmonogramów dyfuzji, osobnych VAE. Jeden transformer, jeden tokenizator na modalność, skala.

Show-o, Janus-Pro i InternVL-U wszystkie budują na tej tezie lub ją kwestionują. Chińskie laboratoria (BAAI, DeepSeek) publikują bardziej agresywnie w tym kierunku niż amerykańskie laboratoria do 2025 roku.

## Use It

`code/main.py` buduje dwa zabawkowe elementy:

- Kalkulator liczby tokenów 2D vs 3D VQ: biorąc (rozdzielczość, łatka, długość_klipu, FPS), oblicza liczbę tokenów dla obrazu vs wideo.
- Autogresywny sampler tokenów obrazkowych z classifier-free guidance i temperaturą.

Implementacja CFG odpowiada przepisowi Emu3 — mieszaj warunkowe i bezwarunkowe logity z wagą guidance.

## Ship It

Ta lekcja produkuje `outputs/skill-token-gen-cost-analyzer.md`. Na podstawie specyfikacji produktu generacyjnego (obraz lub wideo, docelowa rozdzielczość, poziom jakości, budżet opóźnienia) oblicza liczbę tokenów, koszt inferencji i wybiera rodzinę Emu3 vs dyfuzję.

## Exercises

1. Emu3 produkuje 4096 tokenów na obraz 512x512 przy redukcji 8x8. Oblicz odpowiednik dla 1024x1024 i 2048x2048. Co dzieje się z opóźnieniem inferencji?

2. Przeczytaj Sekcję 3.3 artykułu Emu3 o tokenizatorze wideo. Opisz kształt 3D VQ łatki i dlaczego wynosi 4x4x4, a nie 8x8x1.

3. Waga classifier-free guidance 5.0 vs 3.0: jaki efekt wizualny? Prześledź matematykę w `code/main.py`.

4. Oblicz FLOPy treningowe dla Emu3-7B przy 300B tokenów i porównaj ze Stable Diffusion 3. Który był droższy w treningu?

5. Emu3 pokonuje SDXL na FID, ale nie na VQAv2 vs wyspecjalizowane VLM. Wyjaśnij, dlaczego podejście z jednym kosztem wykazuje różne mocne strony vs specjaliści na różnych benchmarkach.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Przewidywanie następnego tokena | "NTP" | Standardowy autogresywny koszt: przewidź token[i+1] mając token[0..i]; działa dla każdej modalności po tokenizacji |
| Tokenizator IBQ | "Odwrotny kwantyzator wąskiego gardła" | Klasa VQ-VAE z większymi kodbukami (32768+) i lepszą rekonstrukcją niż Chameleon |
| 3D VQ | "Kwantyzator czasoprzestrzenny" | Kodbuk indeksowany przez (czas, wiersz, kolumna); jeden token obejmuje sześcian pikseli 4x4x4 |
| Classifier-free guidance | "CFG" | Mieszaj warunkowe i bezwarunkowe logity z wagą gamma; poprawia jakość obrazu na inferencji |
| Ujednolicone słownictwo | "Wspólne tokeny" | Tekst + obraz + wideo wszystkie korzystają z tej samej przestrzeni całkowitej; model przewiduje, która modalność będzie następna |
| MJHQ-30K | "Benchmark generacji obrazów" | Benchmark jakości Midjourney z 30k promptami; Emu3 raportuje tutaj FID |

## Further Reading

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)