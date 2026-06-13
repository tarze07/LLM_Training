# Systemy Moderacji — OpenAI, Perspective, Llama Guard

> Produkcyjne systemy moderacji operacjonalizują polityki bezpieczeństwa zdefiniowane w Lekcjach 12-16. OpenAI Moderation API: `omni-moderation-latest` (2024) zbudowany na GPT-4o klasyfikuje tekst + obrazy w jednym wywołaniu; 42% lepiej na wielojęzycznym zbiorze testowym niż poprzednia wersja; schemat odpowiedzi zwraca 13 booleanów kategorii — harassment, harassment/threatening, hate, hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors, violence, violence/graphic; darmowy dla większości programistów. Wzorce warstwowe: Moderacja wejścia (przed generacją), Moderacja wyjścia (po generacji), Moderacja niestandardowa (reguły domenowe). Asynchroniczne równoległe wywołania ukrywają opóźnienie; odpowiedzi zastępcze po fladze. Llama Guard 3/4 (Lekcja 16): 14 zagrożeń MLCommons, Code Interpreter Abuse, 8 języków (v3), wieloobrazy (v4). Perspective API (Google Jigsaw): ocena toksyczności poprzedzająca falę LLM-jako-moderator; głównie jednowymiarowa toksyczność z wariantami severe-toxicity/insult/profanity; linia bazowa dla badań nad moderacją treści. Wycofania: Azure Content Moderator wycofany luty 2024, wycofany z użycia luty 2027, zastąpiony przez Azure AI Content Safety.

**Type:** Build
**Languages:** Python (stdlib, trzywarstwowa uprząż moderacji)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## Learning Objectives

- Opisać taksonomię kategorii OpenAI Moderation API i czym różni się od zestawu MLCommons Llama Guard 3.
- Opisać wzorzec trzech warstw moderacji (wejście, wyjście, niestandardowa) i wymienić jeden tryb awarii każdej.
- Opisać pozycję Perspective API jako linii bazowej z ery przed-LLM i dlaczego pozostaje używane w badaniach.
- Wymienić harmonogram wycofania Azure.

## The Problem

Lekcje 12-16 opisują ataki i narzędzia obronne. Lekcja 29 obejmuje wdrożone systemy moderacji, które operacjonalizują obrony na powierzchni, gdzie użytkownicy stykają się z produktem. Trzywarstwowy wzorzec to domyślna konfiguracja 2026 roku.

## The Concept

### OpenAI Moderation API

`omni-moderation-latest` (2024). Zbudowany na GPT-4o. Klasyfikuje tekst + obrazy w jednym wywołaniu. Darmowy dla większości programistów.

Kategorie (13 booleanów w schemacie odpowiedzi):
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Obsługa multimodalna dotyczy `violence`, `self-harm` i `sexual`, ale nie `sexual/minors`; pozostałe są tylko tekstowe.

Dla uprzęży kodu w `code/main.py` zwijamy podkategorie `/threatening`, `/intent`, `/instructions` i `/graphic` do ich nadrzędnych kategorii dla uproszczenia pedagogicznego. Kod produkcyjny powinien używać pełnego 13-kategoryjnego schematu.

42% lepiej na wielojęzycznym zbiorze testowym niż poprzedni endpoint moderacji. Wyniki na kategorię; aplikacje ustawiają progi.

### Llama Guard 3/4

Omówione w Lekcji 16. 14 kategorii zagrożeń MLCommons (inaczej zorganizowane niż 13 booleanów schematu odpowiedzi OpenAI). Obsługuje 8 języków (v3). Llama Guard 4 (kwiecień 2025) jest natywnie multimodalny, 12B.

Taksonomie OpenAI i Llama Guard nakładają się, ale różnią. OpenAI ma "illicit" jako szeroką kategorię; Llama Guard ma "przestępstwa z użyciem przemocy" i "przestępstwa bez użycia przemocy" osobno. Wdrożenia wybierają na podstawie dopasowania do swojej taksonomii polityki.

### Perspective API (Google Jigsaw)

System oceny toksyczności poprzedzający falę LLM-jako-moderator (przed 2020). Kategorie: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, THREAT, IDENTITY_ATTACK. Jednowymiarowy główny wynik (TOXICITY) z wariantami podwymiarów.

Szeroko używany jako linia bazowa w badaniach nad moderacją treści, ponieważ API jest stabilne, udokumentowane i ma lata danych kalibracyjnych. Dla nowoczesnych przypadków użycia sąsiadujących z LLM, Llama Guard lub OpenAI Moderation jest zazwyczaj lepszym dopasowaniem.

### Wzorzec trzech warstw

1. **Moderacja wejścia.** Klasyfikuj prompt użytkownika przed generacją. Odrzuć, jeśli oznaczony. Opóźnienie: jedno wywołanie klasyfikatora.
2. **Moderacja wyjścia.** Klasyfikuj output modelu przed dostarczeniem. Zastąp odmową, jeśli oznaczony. Opóźnienie: jedno wywołanie klasyfikatora po generacji.
3. **Moderacja niestandardowa.** Reguły specyficzne dla domeny (regex, listy dozwolonych, polityka biznesowa). Działa na wejściu lub wyjściu.

Trzy warstwy są sekwencyjne z założenia: moderacja wejścia musi zakończyć się przed generacją, a moderacja wyjścia działa po generacji. Równoległość ma zastosowanie w ramach warstwy — uruchamianie wielu klasyfikatorów (np. OpenAI Moderation + Llama Guard + Perspective) jednocześnie na tym samym tekście ukrywa opóźnienie na klasyfikator. Jako opcjonalna optymalizacja, odpowiedź zastępcza ("moment, sprawdzam...") może być pokazana, gdy moderacja wejścia jest w toku, a strumieniowanie token-1 jest odroczone. Zachowanie flagi jest konfigurowalne: odrzuć, oczyść, eskaluj do przeglądu ludzkiego.

### Tryby awarii

- **Tylko wejście.** Nie łapie halucynacji outputu (ataki kodowania z Lekcji 12-14 omijają klasyfikatory wejścia).
- **Tylko wyjście.** Pozwala każdemu wejściu dotrzeć do modelu; zwiększa koszt; ujawnia wewnętrzne rozumowanie napastnikowi.
- **Tylko niestandardowe.** Nie jest odporne między kategoriami; regex jest kruche.

Warstwowe jest domyślne. Pasy i szelki.

### Wycofanie Azure

Azure Content Moderator: wycofany luty 2024, wycofany z użycia luty 2027. Zastąpiony przez Azure AI Content Safety, który jest oparty na LLM i integruje się z Azure OpenAI. Migracja to projekt na poziomie terenu 2024-2027 dla wdrożeń Azure.

### Miejsce w Fazie 18

Lekcja 16 obejmuje narzędzia moderacji w kontekście red-team. Lekcja 29 obejmuje wdrożoną moderację. Lekcja 30 zamyka bieżącymi dowodami zdolności podwójnego zastosowania.

## Use It

`code/main.py` buduje trzywarstwową uprząż moderacji: moderator wejścia (słowa kluczowe + wynik kategorii), moderator wyjścia (ten sam klasyfikator na wyjściu), moderator niestandardowy (reguły domenowe). Możesz uruchomić dane wejściowe i zaobserwować, która warstwa łapie co.

## Ship It

Ta lekcja produkuje `outputs/skill-moderation-stack.md`. Dla wdrożenia, rekomenduje konfigurację stosu moderacji: który klasyfikator na wejściu, który na wyjściu, które reguły niestandardowe i jaki sędzia dla przypadków brzegowych.

## Exercises

1. Uruchom `code/main.py`. Uruchom nieszkodliwe, graniczne i szkodliwe wejście przez wszystkie trzy warstwy. Raportuj, która warstwa zadziałała dla każdego.

2. Rozszerz uprząż o ocenę toksyczności w stylu Perspective-API na konkretnej kategorii. Porównaj jego zachowanie progowe z wynikiem kategorii.

3. Przeczytaj dokumentację OpenAI Moderation API i listę kategorii Llama Guard 3. Zmapuj każdą kategorię OpenAI na najbliższe kategorie Llama Guard. Zidentyfikuj trzy kategorie, które nie mapują się czysto.

4. Zaprojektuj stos moderacji dla wdrożenia asystenta kodu (np. GitHub Copilot). Zidentyfikuj kategorie najbardziej i najmniej istotne i zaproponuj reguły niestandardowe.

5. Azure Content Moderator wycofuje się w lutym 2027. Zaplanuj migrację do Azure AI Content Safety. Zidentyfikuj najwyższy element ryzyka migracji.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | 13-kategoryjny klasyfikator (tekst) oparty na GPT-4o z częściową obsługą multimodalną |
| Perspective API | "toksyczność Google Jigsaw" | Linia bazowa oceny toksyczności z ery przed-LLM |
| Llama Guard | "14 kategorii MLCommons" | Klasyfikator zagrożeń Meta (v3: 8B tekst, 8 języków; v4: 12B multimodalny) |
| Moderacja wejścia | "filtr przed generacją" | Klasyfikator na prompcie użytkownika przed wywołaniem modelu |
| Moderacja wyjścia | "filtr po generacji" | Klasyfikator na output modelu przed dostarczeniem |
| Moderacja niestandardowa | "reguły domenowe" | Reguły specyficzne dla wdrożenia (regex, lista dozwolonych, polityka) |
| Moderacja warstwowa | "wszystkie trzy warstwy" | Standardowy wzorzec wdrożenia produkcyjnego |

## Further Reading

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — endpoint omni-moderation
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — repozytorium Llama Guard
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — ocena toksyczności
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — zamiennik Azure