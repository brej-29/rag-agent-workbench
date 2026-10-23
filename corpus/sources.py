"""v1 corpus sources: official machine-readable endpoints only."""
from dataclasses import dataclass

CELLAR = "http://publications.europa.eu/resource/celex/{}"
OSCAL = "https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/{}"
EU_LICENCE = "© European Union, https://eur-lex.europa.eu. Reuse authorised under Commission Decision 2011/833/EU"
NIST_LICENCE = "U.S. Government work; not subject to copyright in the United States (17 U.S.C. 105)"


@dataclass(frozen=True)
class Source:
    key: str          # raw file stem
    prefix: str       # provision id prefix
    short: str        # citation prefix, e.g. "AI Act"
    framework: str
    instrument: str
    url: str
    fmt: str          # eu-xhtml | oscal-catalog | airmf-playbook
    licence: str
    binding: str      # regulation | directive | voluntary_framework
    accept: str = "application/xhtml+xml"
    role: str = "text"  # text | recitals


def _eu(key, prefix, short, framework, instrument, celex, binding="regulation", role="text"):
    return Source(key, prefix, short, framework, instrument, CELLAR.format(celex), "eu-xhtml", EU_LICENCE, binding,
                  role=role)


def _nist(key, short, framework, instrument, url, fmt):
    return Source(key, key, short, framework, instrument, url, fmt, NIST_LICENCE, "voluntary_framework",
                  "application/json")


SOURCES = [
    _eu("eu-ai-act", "eu-ai-act", "AI Act", "EU AI Act", "Regulation (EU) 2024/1689 (consolidated 27 July 2026)",
        "02024R1689-20260727"),
    _eu("eu-ai-act-recitals", "eu-ai-act", "AI Act", "EU AI Act", "Regulation (EU) 2024/1689", "32024R1689",
        role="recitals"),
    _eu("gdpr", "gdpr", "GDPR", "GDPR", "Regulation (EU) 2016/679", "32016R0679"),
    _eu("nis2", "nis2", "NIS2", "NIS2", "Directive (EU) 2022/2555", "32022L2555", binding="directive"),
    _eu("nis2-ir", "nis2-ir", "NIS2 IR", "NIS2", "Commission Implementing Regulation (EU) 2024/2690", "32024R2690"),
    _eu("dora", "dora", "DORA", "DORA", "Regulation (EU) 2022/2554", "32022R2554"),
    _eu("cra", "cra", "CRA", "Cyber Resilience Act", "Regulation (EU) 2024/2847", "32024R2847"),
    _nist("nist-800-53", "SP 800-53", "NIST SP 800-53", "NIST SP 800-53 Rev. 5",
          OSCAL.format("SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"), "oscal-catalog"),
    _nist("nist-csf", "CSF", "NIST CSF 2.0", "NIST Cybersecurity Framework 2.0",
          OSCAL.format("CSF/v2.0/json/NIST_CSF_v2.0_catalog.json"), "oscal-catalog"),
    _nist("nist-ai-rmf", "AI RMF", "NIST AI RMF", "NIST AI 100-1 (AI RMF 1.0), Playbook",
          "https://airc.nist.gov/docs/playbook.json", "airmf-playbook"),
]
