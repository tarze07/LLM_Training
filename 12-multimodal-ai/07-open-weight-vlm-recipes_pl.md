# Przepisy na Otwarte VLM: Co Tak Naprawdę Ma Znaczenie

> Literatura otwartych VLM z lat 2024-2026 to las tabel abalacyjnych. MM1 od Apple przetestował 13 kombinacji enkodera obrazu, łącznika i mieszanki danych. Molmo od Allen AI udowodniło, że szczegółowe ludzkie podpisy biją dystylację GPT-4V. Cambrian-1 przeprowadził 20+ porównań enkoderów. Idefics2 sformalizował pięcioosiową przestrzeń projektową. Prismatic VLMs porównały 27 przepisów treningowych na kontrolowanym benchmarku. Z całego tego szumu, mały zestaw wyników utrzymuje się we wszystkich pracach: enkoder obrazu ma większe znaczenie niż architektura łącznika, mieszanka danych ma większe znaczenie niż jedno i drugie, a szczegółowe ludzkie podpisy biją dystylowane dane syntetyczne. Ta lekcja czyta te tabele, abyś nie musiał.

**Type:** Learn + lab
**Languages:** Python (stdlib, parser tabel abalacyjnych + selektor przepisów)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 minutes

## Learning Objectives

- Wymienić pięcioosiową przestrzeń projektową VLM: enkoder obrazu, łącznik, LLM, mieszanka danych, harmonogram rozdzielczości.
- Odczytać tabelę abalacyjną MM1 / Idefics2 / Cambrian-1 i przewidzieć, które pokrętło wpływa na dany benchmark.
- Wybrać przepis (enkoder, łącznik, dane, rozdzielczość) dla nowego VLM mając budżet obliczeniowy i mieszankę zadań.
- Wyjaśnić, dlaczego szczegółowe ludzkie podpisy biją dystylację GPT-4V przy tej samej liczbie tokenów.

## The Problem

Istnieją setki otwartych VLM. Większość luki między "dobrym" a "najnowocześniejszym" nie wynika z architektury. Chodzi o dane, harmonogram rozdzielczości i wybór enkodera. Wiedza, które pokrętło przekręcić najpierw, gdy twój model osiąga gorsze wyniki, oszczędza ci pomyłki wartej 5 milionów godzin GPU.

Fala 2023 (LLaVA-1.5, InstructBLIP, MiniGPT-4) działała na wstępnym treningu par podpisów + LLaVA-Instruct-150k. Dobra linia bazowa. Osiągnęła maksimum w okolicach MMMU 35%.

Fala 2024 (MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs) przeprowadziła wyczerpujące ablacje. Wyniki były zaskakujące i praktyczne.

## The Concept

### Pięcioosiowa przestrzeń projektowa

Idefics2 (Laurençon et al., 2024) nazwał osie:

1. Enkoder obrazu. CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. Enkodery różnią się rozmiarem paczki, rozdzielczością i celem wstępnego treningu.
2. Łącznik. MLP (2-4 warstwy), Q-Former (32 zapytania + uwaga krzyżowa), Perceiver Resampler (64 zapytania), C-Abstractor (splotowy + dwuliniowy pooling).
3. Model językowy. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. Rozmiar LLM jest dominującym kosztem parametrów.
4. Dane treningowe. Pary podpisów (CC3M, LAION), przeplatane (OBELICS, MMC4), instrukcyjne (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. Harmonogram rozdzielczości. Stała 224/336/448, AnyRes, natywna dynamiczna. Zwiększana podczas treningu lub stała.

Każdy produkcyjny VLM dokonuje wyboru na każdej osi. Większość wariancji w wynikach MMMU jest wyjaśniona przez osie 1, 4 i 5 — nie przez wybrany łącznik.

### Oś 1: enkoder > łącznik

MM1 Sekcja 3.2 pokazała: zamiana z CLIP ViT-L/14 na SigLIP SO400m/14 dodała 3+ punkty MMMU. Zamiana łącznika z MLP na Perceiver Resampler dodała mniej niż 1 punkt. Idefics2 potwierdził: SigLIP > CLIP, Q-Former ≈ MLP ≈ Perceiver przy tej samej liczbie tokenów.

"Starcie Enkoderów Wizyjnych" Cambrian-1 (Tong et al., 2024) przebadało 20+ enkoderów na wizyjnym benchmarku (CV-Bench). Szczyt rankingu to mieszanka DINOv2 i SigLIP; CLIP jest w środku stawki; ImageBind i ViT-MAE są niżej. Luka od CLIP ViT-L do DINOv2 ViT-g/14 wynosi ~5-7 punktów na CV-Bench.

Domyślny enkoder 2026 dla otwartych VLM to SigLIP 2 SO400m/14 dla cech semantycznych + gęstych, czasami łączony z cechami DINOv2 ViT-g/14 ("Przestrzenny Agregator Wizyjny" Cambrian to robi).

### Oś 2: projekt łącznika nie ma znaczenia

MM1, Idefics2, Prismatic i MM-Interleaved doszły do tego samego wniosku: przy ustalonej liczbie tokenów wizualnych, architektura łącznika prawie nie ma znaczenia. Dw warstwowy MLP na średnio-poolowanych paczkach działa w granicach 1 punktu od 32-zapytaniowego Q-Former przy tym samym budżecie tokenów.

To, co ma znaczenie, to liczba tokenów. Więcej tokenów wizualnych = więcej mocy obliczeniowej LLM = lepsza wydajność do pewnego punktu, potem malejące zyski. 64 tokeny na obraz to za mało dla OCR. 576-1024 tokenów to optymalny zakres dla większości otwartych VLM. 2048+ pomaga tylko w przypadku dokumentów i wykresów.

Q-Former vs MLP to kwestia kosztu, nie jakości: Q-Former ogranicza tokeny do 32-64 niezależnie od rozdzielczości obrazu; MLP emituje wszystkie tokeny paczek. Dla wejść o wysokiej rozdzielczości Q-Former oszczędza kontekst LLM; dla niskiej rozdzielczości różnica to szum.

### Oś 3: rozmiar LLM ustala sufit

Podwojenie LLM z 7B do 13B niezawodnie dodaje 2-4 punkty na MMMU w każdej pracy VLM. Przy 70B nasycasz większość benchmarków. Sufit rozumowania multimodalnego VLM to sufit rozumowania tekstowego LLM — enkoder wizyjny może go tylko karmić, nie może za niego rozumować.

To dlatego Qwen2.5-VL-72B i Claude Opus 4.7 miażdżą MMMU-Pro i ScreenSpot-Pro: mózg językowy jest ogromny. VLM 7B nie może zastąpić VLM 70B poprzez sprytny projekt łącznika.

### Oś 4: dane — szczegółowe ludzkie podpisy biją dystylację

Molmo + PixMo (Deitke et al., 2024) to wynik z 2024, który każdy powinien przeczytać. Allen AI zatrudnił ludzkich adnotatorów do opisywania obrazów w 1-3 minutowych gęstych przejściach mowa-tekst, uzyskując 712K gęsto podpisanych obrazów. Żadnej dystylacji GPT-4V w danych treningowych.

Molmo-72B pobił Llama-3.2-90B-Vision na 11 z 11 benchmarków. Różnica nie leży w architekturze — to jakość podpisów. Szczegółowe ludzkie podpisy zawierają 5-10x więcej informacji na obraz niż krótkie podpisy internetowe i pozostają faktycznie ugruntowane tam, gdzie dystylacja GPT-4V halucynuje.

ShareGPT4V (Chen et al., 2023) i Cauldron (Idefics2) podążyły tą samą ścieżką z mieszanką ludzkich + GPT-4V podpisów. Trend jest jasny: dla granicy 2026, gęstość podpisu > ilość podpisu > wygoda dystylacji.

### Oś 5: rozdzielczość i jej harmonogram

Ablacje Idefics2: 384 -> 448 dodaje 1-2 punkty. 448 -> 980 z dzieleniem obrazu (AnyRes) dodaje kolejne 3-5 na benchmarkach OCR. Płaski trening rozdzielczości osiąga plateau przy średniej dokładności; zwiększanie rozdzielczości (start 224, koniec 448 lub natywna) trenuje szybciej i kończy wyżej.

Cambrian-1 przeprowadził kompromis rozdzielczość vs tokeny: przy stałej mocy obliczeniowej możesz mieć więcej tokenów przy niższej rozdzielczości lub mniej tokenów przy wyższej rozdzielczości. Wyższa rozdzielczość wygrywa dla OCR; niższa rozdzielczość-więcej tokenów wygrywa dla ogólnego rozumienia scen.

Przepis produkcyjny 2026: trenuj Etap 1 przy stałej 384, Etap 2 z dynamiczną rozdzielczością do 1280 dla zadań intensywnych OCR.

### Kontrolowane porównanie Prismatic

Prismatic VLMs (Karamcheti et al., 2024) to praca, która kontrolowała wszystkie osie. Ten sam LLM 13B, te same dane instrukcyjne, ta sama ewaluacja — tylko jedna oś zmienia się na raz. Wyniki:

- Liczba tokenów wizualnych na obraz wyjaśnia ~60% wariancji.
- Wybór enkodera wyjaśnia ~20%.
- Architektura łącznika wyjaśnia ~5%.
- Wszystko inne (mieszanka danych, scheduler, LR) pozostałe ~15%.

To przybliżona dekompozycja, ale to najczystsza odpowiedź na pytanie "co powinienem ablować najpierw" w literaturze.

### Selektor na 2026

Biorąc pod uwagę dowody, domyślny przepis na otwarty VLM dla nowego projektu w 2026:

- Enkoder: SigLIP 2 SO400m/14 w natywnej rozdzielczości z NaFlex, połączony z DINOv2 ViT-g/14 dla gęstych cech, jeśli potrzebujesz segmentacji/ugruntowania.
- Łącznik: Dw warstwowy MLP na tokenach paczek. Pomiń Q-Former, chyba że jesteś ograniczony tokenami.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2, 7B dla kosztu, 70B dla jakości, wybrany według docelowego opóźnienia.
- Dane: PixMo + ShareGPT4V + Cauldron, uzupełnione danymi instrukcyjnymi specyficznymi dla zadania.
- Rozdzielczość: dynamiczna (min 256, max 1280 pikseli na dłuższym boku).
- Harmonogram: Etap 1 dopasowanie (projektor tylko), Etap 2 pełne dostrojenie, Etap 3 dostrojenie specyficzne dla zadania.

Każda z tych domyślnych wartości wywodzi się z zmierzonej ablacji w pracach cytowanych na końcu tej lekcji.

## Use It

`code/main.py` to parser tabel abalacyjnych i selektor przepisów. Koduje (skondensowane) tabele abalacyjne MM1 i Idefics2 i pozwala zapytać:

- "Mając budżet X i zadanie Y, który przepis wygrywa?"
- "Jeśli zamienię SigLIP na CLIP w Llama 7B, jaka jest oczekiwana różnica MMMU?"
- "Którą oś powinienem ablować najpierw, aby uzyskać 80% pewności odpowiedzi?"

Wynikiem jest rankingowa lista przepisów z oczekiwanymi różnicami benchmarków i rekomendacją "abluj najpierw".

## Ship It

Ta lekcja produkuje `outputs/skill-vlm-recipe-picker.md`. Mając docelową mieszankę zadań, budżet obliczeniowy i cel opóźnienia, emituje pełny przepis (enkoder, łącznik, LLM, mieszanka danych, harmonogram rozdzielczości) z cytatami do ablacji uzasadniającej każdy wybór. Powstrzymuje inżynierów przed wymyślaniem tabeli abalacyjnej Idefics2 za każdym razem, gdy zaczyna się nowy projekt VLM.

## Exercises

1. Przeczytaj MM1 Sekcja 3.2. Dla stałego LLM 2B przy budżecie 50M obrazów, który enkoder wygrywa? Czy odpowiedź zmieniłaby się przy LLM 13B? Dlaczego?

2. Cambrian-1 odkrywa, że łączenie DINOv2 + SigLIP przewyższa każdy z osobna na wizyjnych benchmarkach, ale nie dodaje sygnału na MMMU. Przewidź, które benchmarki zyskują, a które pozostają płaskie.

3. Twoim celem jest mobilny agent UI na LLM 2B. Wybierz enkoder, łącznik, rozdzielczość i mieszankę danych. Uzasadnij każdy wybór konkretną tabelą abalacyjną.

4. Molmo dostarcza modele 4B i 72B. 4B konkuruje z zamkniętymi VLM 7B; 72B bije Llama-3.2-90B-Vision na 11/11 benchmarków. Co to mówi o hipotezie plateau rozmiaru LLM?

5. Zaprojektuj tabelę abalacyjną, aby wyizolować jakość mieszanki danych od jakości enkodera w VLM 7B. Minimum ile przebiegów treningowych? Zaproponuj cztery ustawienia osi.

## Key Terms

| Term | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Ablation | "Przekręcanie jednego pokrętła" | Przeprowadzanie wielu przebiegów treningowych różniących się dokładnie jedną osią przestrzeni projektowej, utrzymując wszystko inne stałe |
| Connector | "Most" / "projektor" | Moduł z możliwością trenowania, który mapuje wyjście enkodera wizyjnego do przestrzeni tokenów LLM (MLP, Q-Former, Perceiver) |
| Detailed human caption | "Gęsty podpis" | Wielozdaniowy ludzki opis (zwykle 80-300 tokenów) bogatszy niż internetowy tekst alternatywny |
| Distillation | "Podpisy GPT-4V" | Dane treningowe generowane przez silniejszy zastrzeżony VLM; wygodne, ale podatne na dziedziczone halucynacje |
| AnyRes / dynamic res | "Ścieżka wysokiej rozdzielczości" | Strategia karmienia obrazów większych niż natywna rozdzielczość enkodera poprzez kafelkowanie lub M-RoPE |
| Resolution ramp | "Program nauczania" | Harmonogram treningu zaczynający się od niskiej rozdzielczości i zwiększający ją, przyspieszający naukę dopasowania |
| Vision-centric bench | "CV-Bench / BLINK" | Ewaluacja kładąca nacisk na drobnoziarnistą percepcję wizualną, a nie rozumowanie oparte na języku |
| PixMo | "Dane Molmo" | Zbiór 712K gęsto podpisanych obrazów Allen AI; transkrybowana ludzka mowa w gęste podpisy |

## Further Reading

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)