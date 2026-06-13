# Zgodność — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> Pokrycie wielu frameworków to stawka podstawowa dla umów enterprise w 2026. **EU AI Act**: w mocy od 1 sierpnia 2024. Większość wymogów wysokiego ryzyka wchodzi w życie 2 sierpnia 2026. Grzywny do €15M lub 3% globalnego rocznego obrotu za naruszenia obowiązków systemów wysokiego ryzyka (Art. 99(4)); do €35M lub 7% za zakazane praktyki AI (Art. 99(3)). Ma zastosowanie globalnie, jeśli obsługujesz użytkowników w UE. **Colorado AI Act**: skuteczny od 30 czerwca 2026 (opóźniony z lutego 2026 przez SB25B-004) — oceny wpływu dla systemów wysokiego ryzyka, prawo do odwołania od decyzji AI. Wirginia podobna dla kredytów/ zatrudnienia/mieszkań/edukacji. **SOC 2 Type II**: de facto wymóg B2B AI (Type II, nie Type I, dla fintech). **GDPR**: największa udokumentowana grzywna specyficzna dla AI to €30.5M przeciwko Clearview AI (Holenderski DPA, wrzesień 2024); Włochy Garante nałożyły €15M na OpenAI w grudniu 2024 (później uchylone w apelacji w marcu 2026). Maskowanie PII w czasie rzeczywistym przy inferencji to standard obronny; czyszczenie po przetworzeniu nie wystarcza. **HIPAA**: ograniczenia opieki zdrowotnej — nie można wysyłać PHI do zewnętrznych usług AI bez BAA. **PCI-DSS**: pokrycie warstwy interakcji AI wymaga konfiguracji + uzgodnień umownych, nie automatycznie. **ISO 42001**: pojawiający się standard zarządzania AI, rosnący wymóg procuringu obok ISO 27001. Profil referencyjny: OpenAI utrzymuje SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS dla komponentów płatniczych ChatGPT. Mapowanie międzyframeworkowe redukuje zmęczenie audytem: kontrole dostępu mapują się przez ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a).

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Wymienić siedem frameworków 2026 istotnych dla produktów LLM i dopasować każdy do segmentu klientów.
- Przytoczyć harmonogram egzekwowania EU AI Act (w mocy sierpień 2024; egzekwowanie wysokiego ryzyka sierpień 2026) i dwupoziomowy sufit grzywien (€15M / 3% za obowiązki wysokiego ryzyka, €35M / 7% za zakazane praktyki).
- Wyjaśnić, dlaczego czyszczenie PII po przetworzeniu nie wystarcza dla GDPR i wymienić maskowanie w czasie rzeczywistym na warstwie inferencji jako standard obronny.
- Opisać mapowanie kontroli międzyframeworkowe (np. kontrola dostępu mapuje się do ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)).

## The Problem

Dział zakupów klienta enterprise pyta o SOC 2 Type II, GDPR, HIPAA BAA, ISO 27001 i "oświadczenie o zgodności z EU AI Act." Twój zespół ma SOC 2 Type I. Jesteś sześć miesięcy od Type II i nie zacząłeś rejestrów GDPR Art. 30.

Pokrycie wielu frameworków to nie problem LLM — to problem enterprise-SaaS z nakładkami specyficznymi dla LLM. Zespoły zakupowe w 2026 chcą macierzy z wierszem na framework i kolumną na kontrolę, a nie PDF.

## The Concept

### Siedem frameworków

| Framework | Zakres | Wymóg specyficzny dla LLM |
|-----------|--------|--------------------------|
| SOC 2 Type II | B2B SaaS poziom podstawowy | Kontrole procesów audytowane przez 6-12 miesięcy |
| HIPAA | US ochrona zdrowia | BAA wymagane; PHI nie może opuścić infrastruktury bez podpisanej umowy |
| GDPR | Użytkownicy UE | Maskowanie PII w czasie rzeczywistym; prawa podmiotów danych; rejestry Art. 30 |
| PCI-DSS | Dane płatnicze | Konfiguracja + umowy dla AI dotykającego płatności |
| EU AI Act | Obsługa użytkowników UE | Klasyfikacja poziomu ryzyka; systemy wysokiego ryzyka: ocena zgodności, dokumentacja, logowanie |
| Colorado AI Act | Obsługa mieszkańców CO | Oceny wpływu; prawo do odwołania |
| ISO 42001 | Zarządzanie AI | Pojawiający się; łączy się z ISO 27001 |

### Harmonogram EU AI Act

- 1 sierpnia 2024: w mocy.
- 2 lutego 2025: egzekwowanie zakazanych praktyk AI.
- 2 sierpnia 2026: egzekwowanie systemów wysokiego ryzyka (ocena zgodności, dokumentacja, logowanie).
- Sierpień 2027: systemy wysokiego ryzyka w produktach objętych zharmonizowanym ustawodawstwem.

Poziomy ryzyka: Niedopuszczalne (zakazane), Wysokie ryzyko (zgodność + logowanie), Ograniczone ryzyko (przejrzystość), Minimalne ryzyko (bez ograniczeń). Większość B2B LLM SaaS to ograniczone ryzyko; wysokie ryzyko wchodzi dla zatrudnienia, kredytów, edukacji, egzekwowania prawa, migracji, usług podstawowych.

Grzywny (Art. 99): do €15M lub 3% globalnego rocznego obrotu za naruszenia obowiązków systemów wysokiego ryzyka (Art. 99(4)); do €35M lub 7% za zakazane praktyki AI (Art. 99(3)); która wartość wyższa.

### GDPR — maskowanie w czasie rzeczywistym to standard

Czyszczenie po przetworzeniu (maskowanie PII po tym, jak LLM je zobaczy) nie jest postawą obronną — model już widział dane. Maskowanie w czasie rzeczywistym na warstwie inferencji to standard 2026:

- Rozpoznawanie encji przed wywołaniem LLM.
- Spójna tokenizacja (podejście Mesh) zachowuje semantykę.
- Przechowuj tylko zamaskowane prompty + surowe dane za wyraźną zgodą.

Ostatnie egzekwowanie: €30.5M przeciwko Clearview AI (Holenderski DPA, wrzesień 2024) to największa udokumentowana grzywna specyficzna dla AI; €15M przeciwko OpenAI (Garante Włochy, grudzień 2024) to największa grzywna specyficzna dla LLM, choć została uchylona w apelacji w marcu 2026. Twierdzenia o czyszczeniu po przetworzeniu nie przetrwały audytu.

### HIPAA — BAA nie jest opcjonalne

Nie możesz wysyłać PHI do zewnętrznych usług AI bez podpisanej Business Associate Agreement. Wszystkie trzy platformy LLM hiperskalowców (Bedrock, Azure OpenAI, Vertex) oferują BAA. OpenAI bezpośrednie API oferuje BAA. Anthropic bezpośrednie API oferuje BAA. Potwierdź przed wysłaniem PHI.

### SOC 2 Type II

Type I: kontrole zaprojektowane i udokumentowane.
Type II: kontrole działają skutecznie przez 6-12 miesięcy.

Zakupy B2B w 2026 domyślnie oczekują Type II. Type I to początek; Type II to bramka.

Częste czynniki napędzające audyt: logi dostępu (kto co widział), zarządzanie zmianami (jak zostało wdrożone), oceny ryzyka (kwartalnie), reagowanie na incydenty (testowane?). Log audytowy z Phase 17 · 25 jest bezpośrednio wielokrotnego użytku.

### Mapowanie międzyframeworkowe

Jedna polityka kontroli dostępu zaspokaja wiele kontroli frameworków:

| Kontrola | Frameworki |
|---------|-----------|
| Logowanie dostępu | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| Zarządzanie zmianami | ISO 27001 A.8.32, PCI DSS Req. 6, zakres powiadamiania o naruszeniu HIPAA |
| Szyfrowanie w tranzycie | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| Zarządzanie sekretami | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

Narzędzia zgodności (Drata, Vanta, Secureframe) automatyzują to mapowanie. Warte kosztu na dużą skalę.

### ISO 42001 — pojawiający się

Opublikowany późno 2023. Rosnący wymóg procuringu obok ISO 27001. Framework dla zarządzania AI obejmujący zarządzanie ryzykiem, jakość danych, przejrzystość, nadzór ludzki.

### Profil referencyjny OpenAI

OpenAI utrzymuje SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS dla komponentów płatniczych ChatGPT. To mniej więcej stawka podstawowa enterprise w 2026.

### Liczby, które warto zapamiętać

- Grzywny EU AI Act: do €15M / 3% (obowiązki wysokiego ryzyka, Art. 99(4)); do €35M / 7% (zakazane praktyki, Art. 99(3)).
- Egzekwowanie wysokiego ryzyka EU AI Act: 2 sierpnia 2026.
- Największa udokumentowana grzywna GDPR specyficzna dla AI: €30.5M, Clearview AI (Holenderski DPA, wrzesień 2024).
- Największa grzywna GDPR specyficzna dla LLM: €15M, OpenAI (Garante Włochy, grudzień 2024; uchylona w apelacji marzec 2026).
- Okno SOC 2 Type II: 6-12 miesięcy działania kontroli.
- Data skuteczności Colorado AI Act: 30 czerwca 2026 (opóźniony z lutego 2026 przez SB25B-004).

## Use It

`code/main.py` to arkusz kalkulacyjny mapowania zgodności w Pythonie — dla danej kontroli wymienia frameworki, które zaspokaja.

## Ship It

Ta lekcja produkuje `outputs/skill-compliance-matrix.md`. Na podstawie segmentu klientów i geografii określa wymagane frameworki i kontrole.

## Exercises

1. Twój pierwszy klient enterprise wymaga SOC 2 Type II, HIPAA BAA, oświadczenia EU AI Act. Jaka jest minimalna opłacalna postawa zgodności, aby wygrać umowę?
2. Sklasyfikuj trzy hipotetyczne produkty LLM według poziomów ryzyka EU AI Act. Co zmienia się przy wysokim ryzyku?
3. Przypadkowo wysłałeś PHI do dostawcy bez BAA. Przeprowadź reagowanie na incydent.
4. Uzasadnij, czy ISO 42001 jest "konieczne w 2026" dla średniego sprzedawcy AI.
5. Dopasuj pola swojego logu audytowego LLM (Phase 17 · 25) do co najmniej trzech kontroli frameworków.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SOC 2 Type II | "audited controls" | Controls operating over 6-12 months, independently attested |
| HIPAA BAA | "healthcare contract" | Business Associate Agreement; required for PHI |
| GDPR | "EU privacy" | Real-time PII redaction is the defensible 2026 standard |
| EU AI Act | "EU AI rules" | High-risk enforcement August 2026; €15M / 3% (high-risk obligations) — €35M / 7% (prohibited practices) |
| Colorado AI Act | "US AI state law" | June 30, 2026 effective (delayed by SB25B-004); impact assessments |
| ISO 42001 | "AI governance" | Emerging framework for AI risk + transparency |
| ISO 27001 | "security ISMS" | Information Security Management System baseline |
| Conformity assessment | "EU AI doc package" | High-risk requirement: docs, testing, logging |
| Cross-framework mapping | "one control, many frames" | Single policy satisfies multiple framework controls |

## Further Reading

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard.