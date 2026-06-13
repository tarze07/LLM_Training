# Bezpieczeństwo — Sekrety, Rotacja Kluczy API, Logi Audytowe, Zabezpieczenia

> Wyeliminuj rozprzestrzenianie sekretów poprzez scentralizowane magazyny (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Nigdy nie przechowuj poświadczeń w plikach konfiguracyjnych, plikach env w VCS, arkuszach kalkulacyjnych. Używaj ról IAM zamiast statycznych kluczy; OIDC dla CI/CD. Wzorzec bramki AI to rozwiązanie 2026: aplikacje → bramka → dostawca modelu, z bramką pobierającą poświadczenia z vaultu w czasie wykonania. Obracaj w vault, a wszystkie aplikacje podchwycą w minutach — bez ponownych wdrożeń, bez Slack "kto ma nowy klucz". Polityka rotacji ≤90 dni; skanuj TruffleHog / GitGuardian / Gitleaks przy każdym commicie. Zero-trust: MFA, SSO, RBAC/ABAC, krótkotrwałe tokeny, stanowisko urządzenia. Maskowanie PII używa rozpoznawania encji do maskowania PHI/PII przed przekazaniem; spójna tokenizacja (podejście Mesh) mapuje wrażliwe wartości na stabilne zastępniki, aby LLM zachowywał semantykę kodu/relacji. Wyjście sieciowe: usługi LLM w dedykowanej podsieci VPC/VNet whitelistującej tylko `api.openai.com`, `api.anthropic.com` itp.; blokuj wszystkie inne wychodzące. Incydent 2026: atak na łańcuch dostaw Vercel przez skompromitowane poświadczenia CI/CD, które wyciekły zmienne env w tysiącach wdrożeń klientów.

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Wymienić cztery antywzorce zarządzania sekretami (pliki konfiguracyjne w VCS, zakodowane env, arkusze kalkulacyjne, statyczne klucze) i wymienić ich zamienniki.
- Wyjaśnić wzorzec bramka-pobiera-z-vault jako standard produkcyjny 2026.
- Zaimplementować maskownicę PII ze spójną tokenizacją (ta sama wartość → ten sam zastępnik), aby semantyka przetrwała.
- Wymienić incydent łańcucha dostaw Vercel 2026 i czego nauczył o higienie poświadczeń CI/CD.

## The Problem

Stażysta commit `.env` z kluczami API. Szybko go usuwa. Klucze są już w historii git — skan GitGuardian to wychwytuje, Twój proces rotacji to "wyślij do zespołu na Slacku, zaktualizuj 40 plików konfiguracyjnych, wdróż ponownie wszystkie usługi." 8 godzin później połowa usług jest aktywna, połowa czeka na okna wdrożeniowe.

Osobno, prompty użytkownika zawierają "Mój PESEL to 123-45-6789." Prompt idzie do OpenAI. Masz BAA, ale Twoja wewnętrzna polityka nakazuje maskować PII przed przekazaniem. Nie zrobiłeś tego.

Osobno, Twój pod LLM w EKS może osiągnąć dowolny host internetowy. Ktoś eksfiltruje dane przez zapytanie DNS do domeny kontrolowanej przez atakującego. Nic tego nie zablokowało.

Bezpieczeństwo dla usług LLM musi adresować wszystkie trzy wektory. Poświadczenia wsparte vaultem. Maskowanie PII. Filtrowanie wyjścia sieciowego. Logi audytowe.

## The Concept

### Scentralizowany vault + pull przez rolę IAM

**Vault**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. Jedno źródło prawdy.

**Rola IAM**: aplikacja/bramka uwierzytelnia się przez swoją tożsamość IAM, nie statyczny klucz. Vault zwraca sekret na czas życia tokena.

**Wzorzec bramki AI**: bramka pobiera `OPENAI_API_KEY` z vaultu w momencie żądania. Obróć w vault; następne żądanie dostanie nowy klucz. Bez ponownych wdrożeń.

### Polityka rotacji ≤ 90 dni

Wszystkie klucze API, tokeny root vault, poświadczenia CI/CD. Automatyczna rotacja tam, gdzie to możliwe. Manualna rotacja logowana i śledzona.

### Skanowanie sekretów

- **TruffleHog** — regex + entropia na commitach.
- **GitGuardian** — komercyjne, wysoka dokładność.
- **Gitleaks** — OSS, działa w CI.

Uruchamiaj przy każdym commicie. Blokuj PR, jeśli wykryto nowy sekret.

### Postawa zero-trust

- MFA wymagane na wszystkich kontach.
- SSO przez SAML/OIDC.
- RBAC (oparty na rolach) lub ABAC (oparty na atrybutach) dla szczegółowego dostępu.
- Krótkotrwałe tokeny (godziny, nie dni).
- Stanowisko urządzenia — tylko urządzenia firmowe z szyfrowaniem dysku.

### Maskowanie PII / PHI

Zanim prompt opuści Twoją infrastrukturę:

1. Rozpoznawanie encji (spaCy NER, Presidio, komercyjne).
2. Maskuj dopasowane encje: `"Mój PESEL to 123-45-6789"` → `"Mój PESEL to [PESEL_TOKEN_A3F]"`.
3. Spójna tokenizacja (podejście Mesh): ta sama wartość mapuje do tego samego zastępnika, więc LLM zachowuje relacje.
4. Opcjonalne odwrotne mapowanie dla odpowiedzi LLM.

Statyczne filtry regex wychwytują podstawowe wzorce; NER wychwytuje więcej. Używaj obu.

### Zabezpieczenia wejścia + wyjścia

Wejście: blokuj znane jailbreaky, zabronione tematy; ograniczaj szybkość na użytkownika.

Wyjście: regex maskujący dla wyciekniętych sekretów (wzorce kluczy API, wzorce email w kontekście odmów), klasyfikator naruszeń polityki.

### Biała lista wyjścia sieciowego

Usługi LLM w dedykowanej podsieci:
- Biała lista: `api.openai.com`, `api.anthropic.com`, endpointy baz wektorowych, endpointy vault.
- Wszystko inne: odrzuć.
- DNS przez resolver tylko z białą listą (unikaj eksfiltracji przez tunelowanie DNS).

### Log audytowy

Niezmienny log każdego wywołania LLM z:
- Znacznikiem czasu.
- Użytkownikiem / tenantem.
- Skrótem promptu (nie surowym promptem dla prywatności).
- Modelem + wersją.
- Liczbami tokenów.
- Kosztem.
- Skrótem odpowiedzi.
- Wszelkimi uruchomieniami zabezpieczeń.

Przechowuj zgodnie z wymogami regulacyjnymi (SOC 2 1 rok, HIPAA 6 lat).

### Incydent Vercel 2026

Atak na łańcuch dostaw: skompromitowane poświadczenia CI/CD wyciekły zmienne env w tysiącach wdrożeń klientów. Lekcja: poświadczenia CI/CD są równoważne produkcyjnym. Przechowuj w vault. Ogranicz zakres. Rotuj agresywnie.

### Liczby, które warto zapamiętać

- Polityka rotacji: ≤ 90 dni.
- Skanuj przy każdym commicie: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: poświadczenia CI/CD skompromitowane → tysiące zmiennych env klientów wyciekło.
- Retencja logów audytowych: SOC 2 = 1 rok, HIPAA = 6 lat.

## Use It

`code/main.py` implementuje zabawkową maskownicę PII ze spójną tokenizacją i append-only log audytowy.

## Ship It

Ta lekcja produkuje `outputs/skill-llm-security-plan.md`. Na podstawie zakresu regulacyjnego i obecnego stanu, planuje migrację vault, maskownicę, wyjście sieciowe, log audytowy.

## Exercises

1. Uruchom `code/main.py`. Wyślij dwa prompty odwołujące się do tego samego PESEL. Potwierdź, że oba dostają ten sam zastępnik.
2. Zaprojektuj politykę wyjścia sieciowego dla wdrożenia vLLM-na-EKS wywołującego OpenAI + Anthropic + Weaviate.
3. Odkrywasz klucz w historii git (2 lata stary). Jaka jest prawidłowa odpowiedź — obróć klucz, wyczyść historię, czy oba? Uzasadnij.
4. Twój log audytowy rośnie 10 GB/dzień. Zaprojektuj poziomy retencji (gorący 30d, ciepły 12mo, zimny 6yr).
5. Uzasadnij, czy odwrotna tokenizacja (podstawianie rzeczywistych wartości z powrotem do odpowiedzi LLM) jest warta złożoności wobec pozostawienia widocznych zastępników.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Vault | "secrets store" | Centralized credential management service |
| IAM role | "identity-based auth" | Role assumed by app; returns short-lived creds |
| OIDC for CI/CD | "cloud-issued tokens" | No static keys in CI — identity via OIDC |
| TruffleHog / GitGuardian / Gitleaks | "secret scanners" | Commit-time secret detection |
| RBAC / ABAC | "access control" | Role-based vs attribute-based |
| PII scrubbing | "data masking" | Remove or tokenize sensitive entities |
| Consistent tokenization | "stable placeholders" | Same value → same token each time |
| Mesh approach | "Mesh tokenization" | Semantic-preserving tokenization pattern |
| Egress whitelist | "outbound allowlist" | Only permitted domains reachable |
| Audit log | "immutable history" | Append-only record for compliance |

## Further Reading

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)