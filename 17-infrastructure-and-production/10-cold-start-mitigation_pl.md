# Mitigacja zimnego startu dla bezserwerowych LLM

> Obraz modelu o wadze 20 GB zajmuje 5-10 minut (7B) do 20+ minut (70B), aby przejść od zimnego do serwowania. W prawdziwym bezserwerowym świecie to nie jest rozgrzewka — to awaria. Mitigacje działają na pięciu warstwach: wstępnie zasiane obrazy węzłów (Bottlerocket na AWS, architektura dwuwolumnowa), strumieniowanie modelu (NVIDIA Run:ai Model Streamer, natywny w vLLM), migawki pamięci GPU (checkpointy Modal, do 10x szybszy restart), ciepłe pule (`min_workers=1`), tierowane ładowanie (potok NVMe→DRAM→HBM ServerlessLLM, 10-200x redukcja latencji) i migracja na żywo, która przenosi tokeny wejściowe (KB) zamiast pamięci podręcznej KV (GB). Modal publikuje 2-4s zimne starty jako podłogę; Baseten 5-10s domyślnie, poniżej sekundy z wstępnym ogrzewaniem. Ta lekcja uczy mierzyć, budżetować i układać w stos pięć warstw.

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## Learning Objectives

- Wymień pięć warstw mitigacji zimnego startu i nazwij jedno narzędzie lub wzorzec na każdej warstwie.
- Oblicz całkowity czas zimnego startu jako sumę (provisionowanie węzła) + (pobieranie wag) + (ładowanie wag do HBM) + (inicjalizacja silnika) dla modelu 70B.
- Wyjaśnij, dlaczego migracja na żywo przenosi tokeny wejściowe (KB), a nie pamięć podręczną KV (GB) i jaka jest kara (ponowne obliczenie).
- Nazwij kompromis ciepłej puli (płać za bezczynne GPU lub zaakceptuj ogon zimnego startu) i próg SLA, przy którym `min_workers > 0` staje się obowiązkowe.

## The Problem

Twój bezserwerowy endpoint LLM skaluje się do zera na noc. O 8 rano ruch pikuje. Pierwsze żądanie czeka, podczas gdy:

1. Karpenter provisionuje węzeł GPU: 45-60s.
2. Kontener pobiera 30 GB obraz z wagami: 120-300s.
3. Silnik ładuje wagi do HBM: 45-120s, w zależności od rozmiaru modelu i szybkości przechowywania.
4. vLLM lub TRT-LLM inicjalizuje wykresy CUDA, pulę pamięci podręcznej KV, tokenizer: 10-30s.

Razem: 220-510s (zgrubnie 3-8 minut), zanim wróci pierwszy token. Twoje SLA to 2s. Wdrażasz ciepłą pulę (`min_workers=1`) i problem wydaje się znikać — ale teraz płacisz za jeden bezczynny GPU 24x7. Jeśli Twoja usługa ma 5 produktów, każdy z jedną ciepłą repliką, to 5 × 24 × 30 = 3600 GPU-godzin/miesiąc, niezależnie od tego, czy zadzwonił choć jeden użytkownik.

Mitigacja zimnego startu to sposób na utrzymanie bezserwerowej ekonomiki przy jednoczesnym zbliżeniu się do latencji zawsze włączonego.

## The Concept

### Warstwa 1 — wstępnie zasiane obrazy węzłów (Bottlerocket)

Na AWS, dwuwolumnowa architektura Bottlerocket oddziela system operacyjny od danych. Zrób migawkę woluminu danych z obrazem kontenera wstępnie pobranym; odwołaj się do ID migawki w swoim `EC2NodeClass`. Nowe węzły uruchamiają się z wagami już na lokalnym NVMe — kroki 2 i część 3 znikają. Działa natywnie z Karpenter. Typowe oszczędności: 2-4 minuty na zimny start dla dużych modeli.

Odpowiednik na GCP: niestandardowe obrazy VM z wstępnie upieczonymi warstwami kontenerów. Na Azure: zarządzane migawki dysków z tym samym wzorcem.

### Warstwa 2 — strumieniowanie modelu (Run:ai Model Streamer)

Zamiast ładować cały plik przed odpowiedzią na pierwsze żądanie, strumieniuj wagi do pamięci GPU warstwa po warstwie i rozpocznij przetwarzanie, gdy tylko pierwszy blok transformera jest rezydentny. NVIDIA Run:ai Model Streamer jest dostarczany natywnie w vLLM 2026. Działa z S3, GCS i lokalnym NVMe. Przecina czas ładowania wag zgrubnie o połowę dla dużych modeli, nakładając I/O na konfigurację obliczeniową.

### Warstwa 3 — migawki pamięci GPU (Modal)

Modal robi checkpoint stanu GPU (wagi, wykresy CUDA, region pamięci podręcznej KV) po pierwszym załadowaniu. Kolejne restarty deserializują bezpośrednio do HBM — 10x szybciej niż ponowna inicjalizacja. To najbliższa rzecz do „uruchom ciepłe GPU w 2 sekundy." Kompromis: migawki są na topologię GPU, więc jeśli Karpenter migruje cię do innego SKU, robisz ponowny checkpoint.

### Warstwa 4 — ciepłe pule (min_workers=1)

Najprostsza mitigacja: utrzymuj jedną replikę zawsze gotową. Koszt to stawka godzinowa jednego GPU 24x7. Arytmetyka jest brutalna na małych modelach (płacisz $0,85-$1,50/h, aby uniknąć 30s zimnego startu) i łagodna dla dużych (płać $4/h, aby uniknąć 5-minutowego zimnego startu). Próg SLA, przy którym ciepłe pule stają się obowiązkowe: typowo P99 TTFT < 60s na modelu 70B+.

### Warstwa 5 — tierowane ładowanie (ServerlessLLM)

ServerlessLLM traktuje przechowywanie jako hierarchię: NVMe (szybkie, ale duże), DRAM (średnie, ale tierowane), HBM (małe, ale natychmiastowe). Wagi są wstępnie ładowane do DRAM; ładowane na żądanie do HBM. Artykuł raportuje 10-200x redukcję latencji na zimnych ładunkach vs naiwne dysk-do-HBM. Adopcja produkcyjna jest wczesna, ale istnieją integracje z vLLM.

### Warstwa 6 — migracja na żywo (dodatkowy wzorzec)

Gdy węzeł staje się niedostępny (wywłaszczenie spot, drenaż węzła), tradycyjny wzorzec to zimny start innej repliki i opróżnienie kolejki żądań. Migracja na żywo przenosi tokeny wejściowe (kilobajty) do miejsca docelowego, które ma załadowany model i ponownie oblicza pamięć podręczną KV w miejscu docelowym. Ponowne obliczenie jest tańsze niż przesyłanie GB pamięci podręcznej KV przez sieć. Dotyczy wdrożeń rozdzielonych.

### Matematyka ciepłej puli

Dla usługi z SLA P99 TTFT 2s, pytanie nie brzmi „ciepła pula tak/nie", ale „ile ciepłych replik i które ścieżki je dostają."

- Ścieżki interaktywne o wysokiej wartości (czat na żywo, agent głosowy): `min_workers=1-2`.
- Ścieżki wsadowe w tle (nocna klasyfikacja): skalowanie-do-zera zaakceptowane, 5-10 minut zimnego startu tolerowane.
- Poziom premium: `min_workers` na dzierżawcę z dedykowaną pojemnością.

### Mierz przed optymalizacją

Anatomia zimnego startu dla modelu 70B na świeżym węźle (ilustracyjnie):

| Faza | Czas | Mitigacja |
|-------|------|-----------|
| Provisionowanie węzła | 50s | Bottlerocket + wstępnie zasiany obraz, ciepła pula |
| Pobieranie obrazu | 180s | Wstępnie zasiany wolumin danych (wyeliminowanie) |
| Wagi do HBM | 75s | Model streamer (zmniejszenie o połowę); migawka GPU (wyeliminowanie) |
| Inicjalizacja silnika | 20s | Trwała pamięć podręczna wykresów CUDA |
| Pierwsze przejście | 3s | Minimalna nieodłączna latencja |
| **Razem zimny** | **328s** | |
| **Razem z mitigacjami** | **~15s** | **22x redukcja** |

### Liczby, które powinieneś zapamiętać

- Modal zimny start: 2-4s (z migawkami GPU).
- Baseten domyślny zimny start: 5-10s; poniżej sekundy z wstępnym ogrzewaniem.
- Surowy 70B zimny start: 3-8 minut.
- Run:ai Model Streamer: ~2x przyspieszenie ładowania wag.
- ServerlessLLM tierowane ładowanie: 10-200x redukcja latencji (liczby z artykułu).

## Use It

`code/main.py` modeluje ścieżkę zimnego startu z każdą mitigacją i bez. Raportuje całkowity czas zimnego startu, koszt ciepłej puli i próg rentowności częstości żądań, powyżej którego ciepła pula się opłaca.

## Ship It

Ta lekcja produkuje `outputs/skill-cold-start-planner.md`. Biorąc SLA, rozmiar modelu i kształt ruchu, wybiera, które mitigacje układać w stos.

## Exercises

1. Uruchom `code/main.py`. Oblicz próg rentowności częstości żądań, powyżej którego ciepła replika jest tańsza niż płacenie podatku zimnego startu przez dodatkowe upuszczanie żądań przy SLO.
2. Wdrażasz model 13B z SLA P99 TTFT 3s. Wybierz minimalny stos mitigacji (najmniej warstw), który to osiąga.
3. Wstępne zasianie Bottlerocket eliminuje pobieranie obrazu, ale wagi nadal ładują się z migawki do HBM. Oblicz czas ścienny dla modelu 70B, jeśli NVMe wspierany migawką czyta z prędkością 7 GB/s.
4. Twój dostawca bezserwerowy oferuje migawki GPU (Modal), a Twój zespół odmawia, ponieważ „migawki wyciekają PII." Przedstaw obie strony — jakie jest realistyczne ryzyko i jaka jest mitigacja (efemeryczne migawki, szyfrowanie, izolacja przestrzeni nazw)?
5. Zaprojektuj politykę tierowanej ciepłej puli: ile ciepłych replik dla płatnych użytkowników, użytkowników próbnych i obciążeń wsadowych? Pokaż matematykę.

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Cold start | „wielka pauza" | Czas od żądania do pierwszego tokena na świeżej replice |
| Warm pool | „zawsze-włączone minimum" | `min_workers >= 1`, aby utrzymać co najmniej jedną replikę gotową |
| Pre-seeded image | „upieczony AMI" | Obraz węzła z wagami kontenera wstępnie rezydentnymi |
| Bottlerocket | „system operacyjny węzła AWS" | System operacyjny AWS zoptymalizowany pod kontenery z obsługą migawek dwuwolumnowych |
| Model streamer | „ładowanie strumieniowe" | Nakładanie I/O wag na konfigurację obliczeniową |
| GPU snapshot | „checkpoint do HBM" | Serializacja stanu GPU po załadowaniu; deserializacja przy restarcie |
| Tiered loading | „NVMe + DRAM + HBM" | Hierarchia warstw przechowywania; ładowanie na żądanie |
| Live migration | „przenieś tokeny" | Transfer danych wejściowych (KB), ponowne obliczenie KV w miejscu docelowym |
| `min_workers` | „ciepłe repliki" | Bezserwerowa minimalna liczba podtrzymywanych |
| Scale-to-zero | „pełna bezserwerowość" | Brak kosztu przy bezczynności; zaakceptuj pełny podatek zimnego startu |

## Further Reading

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — opublikowane benchmarki Modala i architektura checkpointów.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — wzorzec migawki wstępnie zasianego woluminu danych.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — nakładanie ładowania wag na konfigurację obliczeniową.
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — podręcznik wstępnego ogrzewania.
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — projekt tierowanego ładowania.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — migracja na żywo dla wdrożeń rozdzielonych.

(End of file - total 126 lines)