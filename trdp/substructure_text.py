import re


ELEMENT_BY_ATOMIC_NUMBER = {
    "1": "hydrogen",
    "5": "boron",
    "6": "carbon",
    "7": "nitrogen",
    "8": "oxygen",
    "9": "fluorine",
    "14": "silicon",
    "15": "phosphorus",
    "16": "sulfur",
    "17": "chlorine",
    "35": "bromine",
    "53": "iodine",
}


SMARTS_FRIENDLY_MAP = {
    "[#6]-[#8]": "a carbon-oxygen single bond (C-O)",
    "[#6]-[#7]": "a carbon-nitrogen single bond (C-N)",
    "[#6]=[#8]": "a carbonyl-like carbon-oxygen double bond (C=O)",
    "[#16]=*": "a sulfur double-bond related pattern",
    "[#16]": "a sulfur-containing fragment",
    "[#8]": "an oxygen-containing fragment",
    "[#7]": "a nitrogen-containing fragment",
    "[#8R]": "oxygen in a ring environment",
    "[O;!H0]": "an oxygen atom with at least one hydrogen (e.g., hydroxyl-like)",
    "[F,Cl,Br,I]": "a halogen atom (F/Cl/Br/I)",
    "[#7]~[#8]": "a nitrogen-oxygen linkage",
}


def _element_summary_from_smarts(smarts: str) -> str:
    nums = re.findall(r"#(\d+)", smarts)
    names = []
    for num in nums:
        if num in ELEMENT_BY_ATOMIC_NUMBER:
            names.append(ELEMENT_BY_ATOMIC_NUMBER[num])

    uniq = []
    for name in names:
        if name not in uniq:
            uniq.append(name)

    if not uniq:
        return "mixed atom types"
    if len(uniq) == 1:
        return f"{uniq[0]}-related motif"
    return f"motif involving {', '.join(uniq)}"


def smarts_to_human_text(smarts: str) -> str:
    if not smarts or smarts == "?":
        return "an unresolved substructure pattern"

    if smarts in SMARTS_FRIENDLY_MAP:
        return SMARTS_FRIENDLY_MAP[smarts]

    if "~" in smarts and "=" in smarts:
        return f"{_element_summary_from_smarts(smarts)} with mixed single/double-bond connectivity"
    if "=" in smarts:
        return f"{_element_summary_from_smarts(smarts)} with double-bond involvement"
    if "R" in smarts:
        return f"{_element_summary_from_smarts(smarts)} in a ring-related context"
    if "~" in smarts:
        return f"{_element_summary_from_smarts(smarts)} with linked-neighbor pattern"
    return _element_summary_from_smarts(smarts)
