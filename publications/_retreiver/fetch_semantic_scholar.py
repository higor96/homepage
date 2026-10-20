"""Busca publicações de um autor via a API pública do Semantic Scholar
(api.semanticscholar.org). Mesma divisão de sync_from_google_scholar.py:
papers com o dono do site vão direto pra publications/ (oficial, sem
estágio de rascunho); os demais exigem --person (slug em people/) pra
saber de quem são e vão pra publications/_external/<slug>/.

Esta é a via alternativa ao Google Scholar (ver sync_from_google_scholar.py
para a via que você pediu de fato usar como fonte principal). Mantive esta
também porque não depende de navegador/Playwright e não tem risco de
bloqueio — útil pra conferir/comparar dados quando quiser.

Uso:
    python3 fetch_semantic_scholar.py --author-id 2373214840 --owner-name "M. M. Raimundo"
    python3 fetch_semantic_scholar.py --author-id 2373214840 --person gvaldrighi

O --owner-name é usado pra decidir author-ids/scholar-ids — só autores da
equipe (people/) ou coautores já confirmados pelo Scholar (ver
resolve_authors.py) ganham foto; o resto cai num ícone genérico.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from resolve_authors import _name_key  # noqa: E402
from render_publication import (  # noqa: E402
    ROOT,
    existing_external_titles,
    existing_titles,
    render_publication,
    slugify,
    write_external_publication,
    write_publication,
)

API_BASE = "https://api.semanticscholar.org/graph/v1"
REQUEST_DELAY_SECONDS = 1.5  # a API pública já pede um request/seg no máximo


def _api_get(path: str, params: dict) -> dict:
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "personal-site-publication-sync/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                wait_time = (attempt + 1) * 3
                print(f"[!] Semantic Scholar 429 (Rate Limit). Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)
            else:
                raise
    return {}


def search_author(name: str) -> list[dict]:
    """Retorna candidatos {authorId, name, paperCount} para um nome —
    útil pra desambiguar manualmente antes de rodar com --author-id."""
    data = _api_get("/author/search", {"query": name, "fields": "name,paperCount,affiliations"})
    return data.get("data", [])


def fetch_author_papers(author_id: str) -> list[dict]:
    fields = "title,abstract,year,venue,publicationDate,externalIds,authors"
    data = _api_get(f"/author/{author_id}/papers", {"fields": fields})
    return data.get("data", [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-id", required=True, help="Semantic Scholar authorId (use search_author() pra achar)")
    parser.add_argument("--owner-name", default="M. M. Raimundo", help="Nome do dono do site, pra decidir quando incluir fotos")
    parser.add_argument("--person", help="slug em people/ — obrigatório pra saber onde salvar papers sem o dono do site (publications/_external/<person>/)")
    args = parser.parse_args()

    owner_key = _name_key(args.owner_name)
    known_titles = existing_titles() | (existing_external_titles(args.person) if args.person else set())
    papers = fetch_author_papers(args.author_id)
    print(f"[i] {len(papers)} publicação(ões) encontradas no perfil.")

    written, externalized, skipped_existing = 0, 0, 0
    for paper in papers:
        title_key = paper["title"].strip().lower()
        if title_key in known_titles:
            print(f"[=] Já cadastrado: {paper['title']}")
            skipped_existing += 1
            continue

        authors = [a["name"] for a in paper["authors"]]
        rendered = render_publication(
            title=paper["title"],
            authors=authors,
            date_iso=paper.get("publicationDate") or f"{paper.get('year', '')}-01-01",
            year=str(paper.get("year", "")),
            venue=paper.get("venue") or "Preprint",
            abstract=paper.get("abstract"),
            source_url=f"https://www.semanticscholar.org/paper/{paper['paperId']}",
            owner_name=args.owner_name,
        )

        involves_owner = any(_name_key(a) == owner_key for a in authors)
        if involves_owner:
            out_path = write_publication(slugify(paper["title"]), rendered)
            print(f"[+] Gerado em publications/ (envolve {args.owner_name}): {out_path.relative_to(ROOT)}")
            written += 1
        elif args.person:
            out_path = write_external_publication(args.person, slugify(paper["title"]), rendered)
            print(f"[+] Gerado em publications/_external/ (sem {args.owner_name}): {out_path.relative_to(ROOT)}")
            externalized += 1
        else:
            print(f"[i] Sem {args.owner_name} entre os autores e sem --person — pulando: {paper['title']}")

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n[✔] {written} arquivo(s) novo(s) em publications/ (já oficiais), "
          f"{externalized} novo(s) em publications/_external/, {skipped_existing} já existente(s) ignorado(s).")


if __name__ == "__main__":
    main()
