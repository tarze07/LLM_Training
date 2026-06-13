# Pochodzenie Danych i Zarządzanie Danymi Treningowymi

> EU AI Act wymaga maszynowo czytelnych standardów rezygnacji dla GPAI do sierpnia 2025 (poprzez wyjątek TDM dyrektywy EU Copyright Directive). California AB 2013 (podpisana 2024) — przejrzystość danych treningowych AI Generatywnego wymaga od programistów opublikowania podsumowania zbiorów danych z 12 wymaganymi polami. Dostosowanie DPA 2025 w sprawie uzasadnionego interesu: Irish DPC (21 maja 2025) akceptuje trening LLM Meta na publicznych treściach pierwszoosobowych dorosłych użytkowników z UE/EOG z zabezpieczeniami po opinii EDPB; Cologne Higher Regional Court (23 maja 2025) oddala nakaz; Hamburg DPA wycofuje pilność; UK ICO (23 września 2025) wydaje pozytywną odpowiedź regulacyjną na zabezpieczenia treningu AI LinkedIn (przejrzystość, uproszczona rezygnacja, rozszerzone okna sprzeciwu) i kontynuuje monitorowanie — nie formalne zatwierdzenie. Brazylijskie ANPD (2 lipca 2024) zawiesiło przetwarzanie Meta z powodu niewystarczającej przejrzystości informacji; środek zapobiegawczy został zniesiony 30 sierpnia 2024 po przedłożeniu przez Meta planu zgodności. Kluczowy problem nieodwracalności: frameworki zgody na ciasteczka są zaprojektowane dla śledzenia w czasie rzeczywistym, odwracalnego; gdy dane trafią do wag modelu, chirurgiczne usunięcie jest niemożliwe — brak praktycznego prawa do bycia zapomnianym dla wytrenowanych sieci neuronowych. Okno zgodności jest w momencie zbierania. Data Provenance Initiative (dataprovenance.org, Longpre, Mahari, Lee et al., "Consent in Crisis", lipiec 2024): audyt na dużą skalę pokazuje gwałtowny spadek wspólnych zasobów danych AI, gdy wydawcy dodają ograniczenia robots.txt.

**Type:** Learn
**Languages:** Python (stdlib, generator rusztowania 12 pól California AB 2013)
**Prerequisites:** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time:** ~60 minutes

## Learning Objectives

- Opisać 12 wymaganych pól California AB 2013 dla przejrzystości danych treningowych AI Generatywnego.
- Wymienić stanowisko DPA 2025 w sprawie uzasadnionego interesu w treningu LLM (Irish DPC, UK ICO, Hamburg, Cologne).
- Opisać problem nieodwracalności: dlaczego prawo do bycia zapomnianym nie ma praktycznego odpowiednika dla wytrenowanych sieci neuronowych.
- Wymienić odkrycie "Consent in Crisis" Data Provenance Initiative.

## The Problem

Zarządzanie danymi treningowymi jest upstreamem każdej karty modelu (Lekcja 26) i obowiązku regulacyjnego (Lekcja 24). W latach 2024-2025 krajobraz regulacyjny skonsolidował się wokół trzech zasad: infrastruktura rezygnacji, ujawnianie na zbiór danych i dostosowanie uzasadnionego interesu dla publicznie dostępnych danych. Dostawcy, którzy nie są zgodni w momencie zbierania, nie mogą zaradzić downstream.

## The Concept

### California AB 2013

Podpisana 2024. Dokumentacja musi być opublikowana do 1 stycznia 2026 dla systemów wydanych w dniu lub po 1 stycznia 2022. Sekcja 3111(a) wymaga od programistów opublikowania wysokopoziomowego podsumowania zbiorów danych użytych w treningu z 12 ustawowymi pozycjami:
1. Źródła lub właściciele zbiorów danych.
2. Opis, w jaki sposób zbiory danych służą zamierzonemu celowi systemu AI.
3. Liczba punktów danych w zbiorach danych (ogólne zakresy akceptowalne; szacunki dla dynamicznych zbiorów danych).
4. Opis typów punktów danych (typy etykiet dla oznaczonych zbiorów; ogólne cechy dla nieoznaczonych).
5. Czy zbiory danych zawierają dane chronione prawem autorskim, znakiem towarowym lub patentem, czy są w całości w domenie publicznej.
6. Czy zbiory danych zostały zakupione lub licencjonowane.
7. Czy zbiory danych zawierają dane osobowe (według Cal. Civ. Code §1798.140(v)).
8. Czy zbiory danych zawierają zagregowane informacje konsumenckie (według Cal. Civ. Code §1798.140(b)).
9. Czyszczenie, przetwarzanie lub inna modyfikacja przez programistę, z zamierzonym celem.
10. Okres, w którym dane były zbierane, z powiadomieniem, jeśli zbieranie jest w toku.
11. Daty, w których zbiory danych zostały po raz pierwszy użyte podczas rozwoju.
12. Czy system używa lub w sposób ciągły używa generowania danych syntetycznych.

Pozycja 12 (dane syntetyczne) jest nowa względem arkuszy danych Gebru et al. 2018. Pozycja 7 (dane osobowe) uruchamia obowiązki Privacy Rights Act (CPRA). Ustawa zwalnia systemy bezpieczeństwa/integralności, obsługi statków powietrznych i federalne wyłączne systemy bezpieczeństwa narodowego (Sekcja 3111(b)).

### EU AI Act (Lekcja 24) i rezygnacja TDM

Wyjątek tekstu i eksploracji danych dyrektywy EU Copyright Directive pozwala na trenowanie na publicznie dostępnych treściach, chyba że podmiot praw rezygnuje. Rozdział Praw Autorskich GPAI Code of Practice EU AI Act wymaga od dostawców GPAI przestrzegania maszynowo czytelnych sygnałów rezygnacji (robots.txt, C2PA "No AI Training" itp.).

### Konwergencja DPA 2025 w sprawie uzasadnionego interesu

Irish DPC (21 maja 2025): plan Meta dotyczący trenowania na publicznych treściach pierwszoosobowych dorosłych użytkowników z UE/EOG zaakceptowany z zabezpieczeniami po opinii EDPB. Cologne Higher Regional Court (23 maja 2025) oddala nakaz przeciwko Meta: rezygnacja jest wystarczająca. Hamburg DPA wycofuje procedurę pilności dla spójności w całej UE. UK ICO (23 września 2025) wydało pozytywną odpowiedź regulacyjną — nie formalne zatwierdzenie — na wznowienie treningu AI LinkedIn z podobnymi zabezpieczeniami i ciągłym monitorowaniem.

Zasada konwergentna: uzasadniony interes może uzasadniać trenowanie na publicznie dostępnych treściach pierwszoosobowych z rezygnacją. Zgoda nie jest wymagana.

### Brazylijskie ANPD (czerwiec 2024)

Zawiesiło przetwarzanie danych brazylijskich użytkowników przez Meta na potrzeby treningu AI z powodu niewystarczającej przejrzystości informacji. Wynik inny niż w EU DPA — ANPD przedłożyło przejrzystość nad dopuszczalność uzasadnionego interesu.

### Problem nieodwracalności

Zgoda na ciasteczka została zaprojektowana dla śledzenia w czasie rzeczywistym, odwracalnego. Dane treningowe są inne: gdy dane trafią do wag modelu, chirurgiczne usunięcie nie jest możliwe. Ponowne trenowanie od zera jest jedyną kompletną remediacją i jest zbyt kosztowne.

Częściowe środki zaradcze:
- **Odkłócanie.** Przybliżone usunięcie; mierzone przez MIA (Lekcja 22).
- **Lokalizacja oparta na funkcji wpływu.** Zidentyfikuj wagi najbardziej pod wpływem danych; selektywnie aktualizuj.
- **Tłumienie przez dostrajanie.** Trenuj model, aby odmawiał outputów pochodzących z danych.

Żaden w pełni nie rozwiązuje problemu. Okno zgodności jest w momencie zbierania.

### Data Provenance Initiative

dataprovenance.org. Longpre, Mahari, Lee et al. "Consent in Crisis" (lipiec 2024): audyt na dużą skalę wspólnych zasobów danych treningowych AI. Odkrycie: wydawcy dodają ograniczenia robots.txt w przyspieszającym tempie. Wspólne zasoby dostępne do trenowania kurczą się szybko. 2023 -> 2024 odnotowało około 25% najważniejszych źródeł treningowych dodających jakieś ograniczenie. Implikacja: przyszła dostępność danych treningowych zależy od nowych paradygmatów pozyskiwania (licencjonowanie, generowanie syntetyczne, uczestnictwo motywowane).

### Miejsce w Fazie 18

Lekcja 26 to dokumentacja na poziomie modelu. Lekcja 27 to zarządzanie na poziomie zbioru danych. Razem definiują warstwę przejrzystości. Lekcja 28 mapuje ekosystem badawczy pracujący nad tymi pytaniami.

## Use It

`code/main.py` generuje rusztowanie podsumowania zbioru danych zgodnego z California AB 2013 (12 pól) dla zabawkowego zbioru danych. Możesz wypełnić pola i zaobserwować, które uruchamiają obowiązki następcze dotyczące prywatności lub praw autorskich.

## Ship It

Ta lekcja produkuje `outputs/skill-provenance-check.md`. Dla zbioru danych użytego w treningu, sprawdza pokrycie 12 pól AB 2013, zgodność infrastruktury rezygnacji, dostosowanie DPA i ocenę ryzyka nieodwracalności.

## Exercises

1. Uruchom `code/main.py`. Produkuj 12-polowe podsumowanie dla zabawkowego zbioru danych i zidentyfikuj, które pola są niedookreślone.

2. Rezygnacja TDM dyrektywy EU Copyright Directive jest maszynowo czytelna. Zaproponuj standardowy format dla sygnału rezygnacji i porównaj go z robots.txt i C2PA "No AI Training."

3. Przeczytaj "Consent in Crisis" Data Provenance Initiative (lipiec 2024). Opisz trzy najszybciej ograniczane kategorie treści i uzasadnij jedną konsekwencję ekonomiczną.

4. Dostosowanie DPA 2025 akceptuje uzasadniony interes dla trenowania na publicznych treściach. Zbuduj scenariusz, w którym uzasadniony interes nie byłby wystarczający i zidentyfikuj podstawę prawną, której dostawca potrzebowałby zamiast tego.

5. Naszkicuj manifest pochodzenia danych treningowych, który łączy się z polami AB 2013 i łańcuchem pochodzenia podpisanym C2PA dla każdego zbioru danych. Zidentyfikuj jedną barierę techniczną i jedną prawną.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| AB 2013 | "ustawa Kalifornii" | Przejrzystość danych treningowych AI Generatywnego; 12 wymaganych pól |
| Wyjątek TDM | "tekst i eksploracja danych" | Wyjątek danych treningowych dyrektywy EU Copyright Directive z rezygnacją |
| Uzasadniony interes | "podstawa UE" | Podstawa Art. 6 RODO, która może uzasadniać trenowanie na publicznych treściach |
| Sygnał rezygnacji | "maszynowo czytelne nie-trenuj" | robots.txt, C2PA "No AI Training," TDM.Reservation |
| Nieodwracalność | "nie można od-trenować" | Dane w wagach modelu nie są chirurgicznie usuwalne |
| Odkłócanie | "przybliżone usunięcie" | Interwencje po treningu w celu zmniejszenia zależności modelu od konkretnych danych |
| Consent in Crisis | "audyt DPI" | Odkrycie z lipca 2024 o przyspieszających ograniczeniach robots.txt |

## Further Reading

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — ustawa o przejrzystości danych treningowych AI Generatywnego
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — rozdział Praw autorskich
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — audyt DPI
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — kontekst regulacyjny