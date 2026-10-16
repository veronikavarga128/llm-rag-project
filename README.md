# llm-rag-project

RAG-rendszer építése és kiértékelése egy választott korpuszon. 


## Cél

Egy reprodukálható RAG-pipeline, plusz egy kísérleti rész, amely összehasonlítja a változatait:

- chunkméret és átfedés hatása,
- BM25 vs. embedding vs. hibrid keresés,
- reranker igen/nem,
- retrieval minősége (recall@k, MRR) és a generált válaszok hűsége.

## Korpusz

**Téma:** magyarországi várak, kastélyok és kúriák cikkei a magyar Wikipédiából.
**Forrás és licenc:** [hu.wikipedia.org](https://hu.wikipedia.org), CC BY-SA 4.0.
**Méret:** 498 cikk (a 509 gyűjtöttből 9 túl rövid, 300 karakter alatti, ezt kihagytuk). _TODO: összes karakterszám a `data/corpus_manifest.json`-ból._

A nyers adat nincs a repóban, a `scripts/download_corpus.py` tölti le (csak standard könyvtárat használ):

```powershell
# 1. Pontos kategórianevek keresése
python scripts/download_corpus.py find-categories "kastély"

# 2. Próbafuttatás: csak összegyűjti és megszámolja a cikkeket
python scripts/download_corpus.py download --dry-run --categories "Magyarország várai" "Magyarország kastélyai, palotái" --depth 2 --exclude listái

# 3. Éles letöltés (ez adta a jelenlegi korpuszt)
python scripts/download_corpus.py download --categories "Magyarország várai" "Magyarország kastélyai, palotái" --depth 2 --exclude listái
```

Ajánlott a letöltés előtt megadni az elérhetőséged (a Wikimedia az általános User-Agentet szigorúbban korlátozza):
`$env:WIKI_USER_AGENT = "llm-rag-project/0.1 (email@pelda.hu)"`

A kimenet a `data/raw/corpus.jsonl` (egy sor egy cikk, nem commitolt) és a `data/corpus_manifest.json` (paraméterek, letöltés ideje, cikkek revízióazonosítói, commitolható). A Wikipédia változik, ezért a manifest rögzíti, melyik állapotot használtuk.

## Követelmények

- Python 3.11+
- Docker Desktop (Qdrant, opcionálisan Ollama)
- Generáláshoz: helyi Ollama vagy egy OpenAI-kompatibilis API

## Telepítés és futtatás

```powershell
# 1. Virtuális környezet (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Konfiguráció
copy .env.example .env

# 3. Vektoradatbázis
docker compose up -d

# (opcionális) helyi LLM
docker compose --profile ollama up -d
docker exec -it rag-ollama ollama pull llama3.1:8b
```

Bash/WSL esetén az aktiválás `source .venv/bin/activate`, a másolás `cp .env.example .env`.

A Qdrant webes felülete: <http://localhost:6333/dashboard>

## Pipeline

```powershell
# Korpusz chunkolása, embeddingje és betöltése Qdrantba
# (az első futáskor letölti az embedding-modellt, ez eltarthat egy ideig)
python -m rag.ingest --recreate

# Keresés kipróbálása
python -m rag.search "diósgyőri vár története" --k 5

# Tesztek
pytest
```

Kísérletekhez minden konfigurációnak külön gyűjtemény való, hogy az eredmények összevethetők maradjanak:

```powershell
python -m rag.ingest --recreate --chunk-size 800 --overlap 150 --collection corpus_c800_o150
python -m rag.ingest --recreate --no-title-prefix --collection corpus_notitle
```

A chunkméret karakterben értendő. Alapból minden chunk embeddingjének elejére bekerül a cikk címe és a szakasz címe (`--no-title-prefix`-szel kikapcsolható), a Qdrantban viszont az eredeti chunk-szöveg tárolódik.

## Kiértékelő készlet

Az `eval/questions.jsonl` 50 kérdést tartalmaz (soronként egy JSON). A készletet LLM segítségével állítottam össze a korpusz szövegéből, majd ellenőriztem; minden válasz mellett ott a szó szerinti forrásidézet.

| Mező | Jelentés |
|---|---|
| `id`, `question`, `answer` | azonosító, kérdés, elvárt válasz |
| `type` | `tény` (40), `több cikk` (4), `nincs válasz` (6) |
| `popularity` | `ismert` / `kevésbé ismert` hely (25–25) |
| `gold_titles` | a cikk(ek), amelyekben a válasz megtalálható (ebből jön a recall@k és az MRR) |
| `evidence` | cikkcím + szó szerinti idézet a forrásszövegből |
| `related_titles` | csak a `nincs válasz` kérdéseknél: a cikk, ahol a válaszra számítani lehetne, de nincs benne |

A `nincs válasz` kérdések a hallucináció mérésére valók: a helyes viselkedés az, ha a rendszer jelzi, hogy a forrásokból nem derül ki a válasz.

A készlet és a korpusz egyezésének ellenőrzése (a bizonyítékok szó szerint szerepelnek-e a cikkekben):

```powershell
python eval/validate_questions.py
```

## Repóstruktúra

```
scripts/        letöltő szkript (download_corpus.py)
src/rag/        a pipeline forráskódja (betöltés, chunkolás, embedding, keresés, generálás)
data/           korpusz (nyers adat nem commitolt)
eval/           kérdés-válasz értékelő készlet és metrikák
experiments/    kísérleti konfigurációk és összesítő eredmények
docs/           rövid dokumentáció, jegyzetek
tests/          tesztek
```

## Ütemterv

| Mérföldkő | Időpont | Tartalom |
|---|---|---|
| Téma kiválasztása | 4. hét | korpusz, irány |
| Köztes állapot | kb. 9. hét | működő baseline, első mérések |
| Végleges leadás és bemutatás | 13. hét / vizsgaidőszak | összes kísérlet, dokumentáció |