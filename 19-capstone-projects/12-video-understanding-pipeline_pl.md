# Capstone 12 — Potok Rozumienia Wideo (Scena, QA, Wyszukiwanie)

> Twelve Labs produktowizował Marengo + Pegasus. VideoDB dostarczyło API CRUD-dla-wideo. AI2's Molmo 2 opublikował otwarte checkpointy VLM. Długi kontekst Gemini obsługuje godziny wideo natywnie. TimeLens-100K zdefiniował ugruntowanie temporalne na skalę. Potok 2026 jest ustalony: segmentacja scen, podpis + osadzenie na scenę, dopasowanie transkryptu, wielowektorowy indeks i zapytanie odpowiadające z (start, stop) znacznikami czasu plus podglądami klatek. Capstone polega na zaindeksowaniu 100 godzin, osiągnięciu publicznych benchmarków i zmierzeniu halucynacji na pytaniach o liczenie i akcje.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (UI)
**Prerequisites:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 hours

## Problem

QA wideo długiego formatu jest najbardziej przepustowościowym multimodalnym problemem w 2026 roku. Gemini 2.5 Pro może czytać 2-godzinne wideo natywnie, ale indeksowanie 100 godzin wideo do przeszukiwalnego korpusu nadal wymaga indeksu na poziomie scen. Produkcyjny kształt łączy segmentację scen (TransNetV2 lub PySceneDetect), tworzenie podpisów na scenę z VLM (Gemini 2.5, Qwen3-VL-Max lub Molmo 2), dopasowanie transkryptu (Whisper-v3-turbo ze znacznikami czasów słów) i wielowektorowy indeks przechowujący podpis, osadzenie klatki i transkrypt obok siebie. Potok zapytań odpowiada z (start, stop) znacznikami czasu plus podglądami klatek.

Benchmarki są publiczne (ActivityNet-QA, NeXT-GQA) plus własny 100-zapytaniowy zestaw niestandardowy. Halucynacja na pytaniach o liczenie i typ akcji to znana trudna klasa awarii; capstone wyraźnie ją mierzy.

## Koncepcja

Trzy potoki działają równolegle podczas indeksacji. **Segmentacja scen** tnie wideo na sceny. **VLM captioning** generuje podpis na scenę i osadzenie klatki z kluczowej klatki. **Dopasowanie ASR** produkuje znaczniki czasów na poziomie słów. Trzy strumienie są łączone przez (scene_id, zakres czasu). Każda scena otrzymuje trzy typy wektorów w wielowektorowym indeksie (Qdrant): osadzenie podpisu, osadzenie kluczowej klatki, osadzenie transkryptu.

W czasie zapytania, pytanie w języku naturalnym odpala się przeciwko wszystkim trzem wektorom; wyniki scalają się z RRF; adapter ugruntowania temporalnego (w stylu TimeLens) udoskonala okno (start, stop) w obrębie najlepszej sceny. Syntezator VLM (Gemini 2.5 Pro lub Qwen3-VL-Max) przyjmuje zapytanie + najlepsze sceny + przycięte klatki i odpowiada z cytowanymi znacznikami czasu i podglądem klatki.

Pomiar halucynacji ma znaczenie. Pytania o liczenie ("ile osób wchodzi do pokoju?") i typ akcji ("czy kucharz nalewa przed mieszaniem?") są notorycznie niewiarygodne. Raportuj dokładność oddzielnie od pytań opisowych.

## Architektura

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## Stack

- Segmentacja scen: TransNetV2 (state-of-the-art 2024-26) lub PySceneDetect
- ASR: Whisper-v3-turbo przez faster-whisper ze znacznikami czasów słów
- VLM captioner + answerer: Gemini 2.5 Pro lub Qwen3-VL-Max lub Molmo 2
- Ugruntowanie temporalne: TimeLens-100K-trained adapter lub VideoITG
- Indeks: Qdrant z obsługą wielowektorową (podpis / klatka / transkrypt)
- UI: Next.js 15 z odtwarzaczem wideo HTML5 i miniaturami scen
- Ewaluacja: ActivityNet-QA, NeXT-GQA, niestandardowy 100-pytaniowy ręcznie oznaczony zestaw
- Benchmark halucynacji: podzbiory liczenia i typu akcji z ręcznymi etykietami

## Build It

1. **Przechodzenie indeksacji.** Akceptuj URL-e YouTube lub lokalne MP4. Zmniejsz skalę do 720p w razie potrzeby. Zapisz `{video_id, file_path}`.

2. **Segmentacja scen.** Uruchom TransNetV2 lub PySceneDetect, aby wyprodukować `[{scene_id, start_ms, end_ms, keyframe_path}]`. Cel: 100 godzin: ~6k-8k scen.

3. **Przejście ASR.** Uruchom Whisper-v3-turbo na audio; wyeksportuj znaczniki czasów na poziomie słów; podziel na wycinki transkryptu na scenę.

4. **VLM captioning.** Na scenę, wywołaj Gemini 2.5 Pro (lub Qwen3-VL-Max) z kluczową klatką i krótkim szablonem podpisu. Wyprodukuj podpis + osadzenie klatki.

5. **Wielowektorowy indeks.** Kolekcja Qdrant z trzema nazwanymi wektorami. Ładunek: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **Zapytanie.** Pytanie w języku naturalnym odpala trzy gęste zapytania; połącz z reciprocal rank fusion; top-k=5 scen.

7. **Ugruntowanie temporalne.** Uruchom adapter w stylu TimeLens na najlepszej scenie, aby udoskonalić okno (start, stop) w obrębie sceny.

8. **Synteza VLM.** Wywołaj Gemini 2.5 Pro z zapytaniem + klipami 3 najlepszych scen (jako obrazy lub krótkie klipy) + transkryptami. Wymagaj cytowań `(video_id, start_ms, end_ms)`.

9. **Ewaluacja.** Uruchom ActivityNet-QA i NeXT-GQA. Zbuduj 100-pytaniowy zestaw niestandardowy. Raportuj ogólną dokładność + podział na klasy (liczenie, akcja, opisowe).

## Use It

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Ship It

`outputs/skill-video-qa.md` jest rezultatem. Mając URL YouTube lub przesłane wideo, potok indeksuje sceny i odpowiada na pytania z cytowanymi znacznikami czasu.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | IoU ugruntowania temporalnego | Intersection-over-union na wstrzymanym zestawie ugruntowania |
| 20 | Dokładność QA | NeXT-GQA i niestandardowy 100-zapytaniowy |
| 20 | Przepustowość indeksacji | Godziny wideo na wydanego dolara |
| 20 | UX i UX cytowań | Linki znaczników czasu, pasek miniatur, skok-do-klatki |
| 15 | Wskaźnik halucynacji | Dokładność liczenia i typu akcji oddzielnie |
| **100** | | |

## Ćwiczenia

1. Zamień Gemini 2.5 Pro na Qwen3-VL-Max w przejściu captioningu. Raportuj różnicę jakości podpisów na 50-scenowej próbce ocenionej przez człowieka.

2. Zmniejsz osadzenie klatki na scenę do jednego połączonego wektora zamiast wielowektorowego. Zmierz regresję wyszukiwania.

3. Zbuduj tryb "ścisłego liczenia": syntezator wyodrębnia każdą liczoną instancję ze znacznikiem czasu, a użytkownik klika, aby zweryfikować. Zmierz, czy weryfikacja użytkownika zmniejsza halucynacje.

4. Porównaj benchmark kosztu indeksacji: godziny-wideo-na-dolara w trzech wyborach VLM. Wybierz optymalny punkt.

5. Dodaj transkrypt z diaryzacją mówców: uruchom pyannote speaker diarization na audio i osadź transkrypty na mówcę. Zademonstruj zapytania "co Alice powiedziała o X?".

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Scene segmentation | "Wykrywanie ujęć" | Cięcie wideo na sceny na granicach ujęć |
| Multi-vector index | "Podpis + klatka + transkrypt" | Kolekcja Qdrant z nazwanymi wektorami na reprezentację |
| Temporal grounding | "Kiedy dokładnie to się stało" | Udoskonalanie okna (start, stop) dla odpowiedzi zapytania |
| Frame embedding | "Reprezentacja wizualna" | Wektorowe osadzenie kluczowej klatki; używane do podobieństwa scena-wizualna |
| RRF fusion | "Reciprocal rank fusion" | Strategia scalania wielu rankingowych list; klasyczna sztuczka hybrydowego wyszukiwania |
| Counting hallucination | "Błędne zliczenie" | Znany tryb awarii VLM na pytaniach "ile X" |
| ActivityNet-QA | "Benchmark QA wideo" | Benchmark dokładności QA wideo długiego formatu |

## Dalsza lektura

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — open VLM checkpoints
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — temporal grounding at scale
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — the hosted reference
- [VideoDB](https://videodb.io) — CRUD-for-video API reference
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — commercial reference
- [TransNetV2](https://github.com/soCzech/TransNetV2) — scene segmentation model
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — classic open alternative
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — reference eval benchmark