"""Citations in the generated supplementary tables (provenance and audit map).

The tables name their sources as "Author (year, locator)"; `cite` adds the
bibliography key of paper/references.bib after each such mention, so that the
reference list of the supplement contains every source the tables name. Only sources
read for the paper are keyed here; others are named as their citing source names them.
"""
import re

# (pattern for the author names, year) -> key of paper/references.bib
SOURCES = [
    (r"Lazar, Agiasofitou and Boehlke|Lazar et al\.(?:\\ )?", "2022", "Lazar2022"),
    (r"Po, Admal and Lazar|Po et al\.(?:\\ )?", "2019", "Po2019"),
    (r"Lazar and Po", "2024", "Lazar2024"),
    (r"Admal et al\.(?:\\ )?", "2017", "Admal2017"),
    (r"Eremeyev and Lazar", "2022", "Eremeyev2022"),
    (r"Solyaev", "2023", "Solyaev2023"),
    (r"Pirmoradi, Suiker and Poorsolhjouy|Pirmoradi et al\.(?:\\ )?", "2024", "Pirmoradi2024"),
    (r"Auffray et al\.(?:\\ )?", "2013", "Auffray2013"),
    (r"Monchiet and Bonnet", "2013", "Monchiet2013"),
    (r"Monchiet and Bonnet", "2011a", "Monchiet2011a"),
    (r"Ma et al\.(?:\\ )?", "2018", "Ma2018"),
    (r"Blaschke", "2017", "Blaschke2017"),
    (r"Kube and de Jong", "2016", "Kube2016"),
]


def cite(text):
    """`text` with ~\\cite{key} after every "Author (year...)" of SOURCES."""
    for names, year, key in SOURCES:
        pat = re.compile(rf"((?:{names})\s*\({year}(?:[,;][^)]*)?\))")
        text = pat.sub(lambda m, k=key: m.group(1) + f"~\\cite{{{k}}}", text)
    return text


def keys(text):
    return re.findall(r"\\cite\{([^}]+)\}", text)
