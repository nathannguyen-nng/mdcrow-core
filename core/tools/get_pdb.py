
from __future__ import annotations
from pathlib import Path
import requests

from langchain.tools import tool


# Repo root: core/tools/get_pdb.py -> parents: tools, core, repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOWNLOADS_DIR = _REPO_ROOT / "notebooks" / "downloads"

@tool
def get_pdb(query_string: str) -> tuple[str | None, str | None]:
    """
    Search RCSB PDB with ``query_string``, download the top hit, and save it under
    ``notebooks/downloads``.
    Returns
    -------
    path, pdb_id
        Absolute path to the saved file and the PDB identifier, or ``(None, None)``
        if nothing was found or the download failed.
    """
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query_string},
        },
        "return_type": "entry",
    }
    r = requests.post(search_url, json=query, timeout=60)
    if r.status_code == 204:
        return None, None
    if "cif" in query_string or "CIF" in query_string:
        filetype = "cif"
    else:
        filetype = "pdb"
    payload = r.json()
    if "result_set" not in payload or len(payload["result_set"]) == 0:
        return None, None
    results = payload["result_set"]
    pdbid = max(results, key=lambda x: x["score"])["identifier"]
    print(f"PDB file found with this ID: {pdbid}")
    download_url = f"https://files.rcsb.org/download/{pdbid}.{filetype}"
    pdb = requests.get(download_url, timeout=120)
    if pdb.status_code != 200:
        return None, None
    filename = f"{pdbid}.{filetype}"
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOWNLOADS_DIR / filename
    out_path.write_text(pdb.text, encoding="utf-8")
    return str(out_path.resolve()), pdbid
