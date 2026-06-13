# Obserwowalność Agentów: Langfuse, Phoenix, Opik

> Trzy platformy obserwowalności agentów open-source dominują w 2026. Langfuse (MIT) — 6M+ instalacji/miesiąc, śledzenie + zarządzanie promptami + ewaluacje + odtwarzanie sesji. Arize Phoenix (Elastic 2.0) — głębokie ewaluacje specyficzne dla agentów, trafność RAG, auto-instrumentacja OpenInference. Comet Opik (Apache 2.0) — zautomatyzowana optymalizacja promptów, zabezpieczenia, wykrywanie halucynacji LLM-as-judge.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 minutes

## Learning Objectives

- Wymień trzy najważniejsze platformy obserwowalności agentów open-source i ich licencje.
- Rozróżnij, w czym każda jest najlepsza: Langfuse (zarządzanie promptami + sesje), Phoenix (RAG + auto-instrumentacja), Opik (optymalizacja + zabezpieczenia).
- Wyjaśnij, dlaczego 89% organizacji raportuje posiadanie obserwowalności agentów w 2026.
- Zaimplementuj potok ślad-do-pulpitu w stdlib z ewaluacją LLM-judge.

## Problem

OTel GenAI (Lekcja 23) daje ci schemat. Wciąż potrzebujesz platformy, która konsumuje spany, uruchamia ewaluacje, przechowuje wersje promptów i wykrywa regresje. Trzej kandydaci kładą nacisk na różne części cyklu życia.

## Koncepcja

### Langfuse (MIT)

- 6M+ instalacji SDK/miesiąc, 19k+ gwiazdek na GitHub.
- Funkcje: śledzenie, zarządzanie promptami z wersjonowaniem + placem zabaw, ewaluacje (LLM-as-judge, opinie użytkowników, niestandardowe), odtwarzanie sesji.
- Czerwiec 2025: dawniej komercyjne moduły (LLM-as-a-judge, kolejki adnotacji, eksperymenty promptowe, Playground) otwarto na licencji MIT.
- Najlepsze dla: obserwowalności end-to-end z ciasną pętlą zarządzania promptami.

### Arize Phoenix (Elastic License 2.0)

- Głębsze ewaluacje specyficzne dla agentów: klastrowanie śladów, wykrywanie anomalii, trafność wyszukiwania dla RAG.
- Natywna auto-instrumentacja OpenInference.
- Współpracuje z zarządzanym Arize AX dla produkcji.
- Brak wersjonowania promptów — pozycjonowany jako narzędzie dryfu/regresji behawioralnej obok szerszych platform.
- Najlepsze dla: trafności RAG, dryfu behawioralnego, wykrywania anomalii.

### Comet Opik (Apache 2.0)

- Zautomatyzowana optymalizacja promptów przez eksperymenty A/B.
- Zabezpieczenia (redakcja PII, ograniczenia tematyczne).
- Wykrywanie halucynacji LLM-judge.
- Benchmark z własnego pomiaru Comet: logowanie + ewaluacje Opik w 23,44s vs Langfuse 327,15s (~14× różnica) — traktuj benchmarki dostawców jako orientacyjne.
- Najlepsze dla: pętli optymalizacji, zautomatyzowanego eksperymentowania, egzekwowania zabezpieczeń.

### Dane branżowe

Według Maxim (analiza terenowa 2026): 89% organizacji ma obserwowalność agentów; problemy jakościowe są główną barierą produkcyjną (32% respondentów je wymienia).

### Wybór jednej

| Potrzeba | Wybierz |
|------|------|
| Wszystko w jednym z zarządzaniem promptami | Langfuse |
| Głęboka ewaluacja RAG + dryf | Phoenix |
| Zautomatyzowana optymalizacja + zabezpieczenia | Opik |
| Otwarte licencjonowanie, brak ELv2 | Langfuse (MIT) lub Opik (Apache 2.0) |
| Integracja z Datadog / New Relic | Dowolna — wszystkie eksportują OTel |

### Gdzie ten wzorzec zawodzi

- **Brak strategii ewaluacji.** Śledzenie bez ewaluacji to tylko drogie logowanie.
- **Własnoręczne LLM-judge bez ugruntowania.** Wzorzec CRITIC (Lekcja 05) ma zastosowanie — sędziowie potrzebują zewnętrznych narzędzi do weryfikacji faktów.
- **Wersje promptów niepowiązane ze śladami.** Gdy produkcja regresuje, nie możesz zbisektować do promptu, który to spowodował.

## Build It

`code/main.py` implementuje kolektor śladów w stdlib + ewaluator LLM-judge:

- Konsumuj spany w kształcie GenAI.
- Grupuj według sesji, oznaczaj nieudane uruchomienia (uruchomienia zabezpieczeń, ewaluacje o niskiej pewności).
- Skryptowany LLM-judge, który ocenia odpowiedzi agenta według rubryki.
- Podsumowanie w stylu pulpitu: wskaźnik awarii, główne przyczyny awarii, rozkład wyników ewaluacji.

Uruchom:

```
python3 code/main.py
```

Wynik: wyniki ewaluacji na sesję i kategoryzacja awarii odpowiadające temu, co pokazałyby Langfuse/Phoenix/Opik.

## Use It

- **Langfuse** samodzielnie hostowane lub w chmurze; podłącz przez OTel lub ich SDK.
- **Arize Phoenix** samodzielnie hostowane; auto-instrumentacja OpenInference.
- **Comet Opik** samodzielnie hostowane lub w chmurze; zautomatyzowana pętla optymalizacji.
- **Datadog LLM Observability** dla mieszanych zespołów operacyjno-ML, które już używają Datadog.

## Ship It

`outputs/skill-obs-platform-wiring.md` wybiera platformę i podłącza ślady + ewaluacje + wersje promptów do istniejącego agenta.

## Ćwiczenia

1. Eksportuj tydzień śladów OTel do chmury Langfuse (darmowy poziom). Które sesje się nie powiodły? Dlaczego?
2. Napisz rubrykę LLM-judge dla swojej domeny (poprawność faktów, ton, zgodność z zakresem). Przetestuj na 50 śladach.
3. Porównaj wersjonowanie promptów Langfuse z klastrowaniem śladów Phoenix. Co mówi ci, co zepsuło się szybciej?
4. Przeczytaj dokumentację zabezpieczeń Opik. Podłącz zabezpieczenie redakcji PII do jednego z twoich uruchomień agenta.
5. Wykonaj benchmark trzech platform na swoim korpusie. Zignoruj liczby publikowane przez dostawców; zmierz własne.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| Śledzenie | „Kolektor spanów" | Konsumuj spany OTel / SDK; indeksuj według sesji |
| Zarządzanie promptami | „CMS promptów" | Wersjonowane prompty powiązane ze śladami |
| LLM-as-judge | „Automatyczna ewaluacja" | Osobny LLM ocenia wynik agenta według rubryki |
| Odtwarzanie sesji | „Odtwarzanie śladu" | Przechodzenie przez przeszłe uruchomienia do debugowania |
| Trafność RAG | „Jakość wyszukiwania" | Czy pobrany kontekst pasuje do zapytania |
| Klastrowanie śladów | „Grupowanie behawioralne" | Grupuj podobne uruchomienia do wykrywania dryfu |
| Egzekwowanie zabezpieczeń | „Polityka w czasie logowania" | Sprawdzanie PII/toksyczności/zakresu na logowanej treści |

## Dalsza lektura

- [Langfuse docs](https://langfuse.com/) — śledzenie, ewaluacje, zarządzanie promptami
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentacja, dryf
- [Comet Opik](https://www.comet.com/site/products/opik/) — optymalizacja + zabezpieczenia
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — schemat, który wszystkie trzy konsumują