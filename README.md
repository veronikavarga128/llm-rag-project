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

## Retrieval-mérés

```powershell
# a baseline gyűjtemény mérése (futó Qdrant és betöltött gyűjtemény kell hozzá)
python eval/run_retrieval_eval.py

# másik konfiguráció mérése
python eval/run_retrieval_eval.py --collection corpus_c800_o150
```

A szkript a 44 válaszolható kérdést futtatja végig, és két szinten számol Recall@k-t (k = 1, 3, 5, 10) és MRR-t, típusonként és hely-ismertség szerint bontva:

- **Cikkszint:** a helyes cikk szerepel-e a találatok között.
- **Chunkszint:** szerepel-e olyan chunk a találatok között, amely a forrásidézetet is tartalmazza. Ez a szigorúbb mérték, a chunkméret és a keresési módszer hatása ebben látszik jobban.

A 6 `nincs válasz` kérdésnél retrieval-metrika nem értelmezett, ezeknél a legjobb találat hasonlósági pontszámát írja ki a szkript, hogy később eldönthető legyen, használható-e pontszámküszöb a "nem tudom" válaszhoz. A részletes eredmény (kérdésenként a találatok is) az `experiments/results/<gyűjtemény>.json` fájlba kerül, ez commitolható. A chunkszintű relevancia a gyűjtemény tényleges chunkjaiból számolódik, ezért a különböző chunkméretű gyűjtemények eredményei összevethetők.

## Kísérletek

```powershell
python experiments/run_experiments.py --dry-run            # mit futtatna
python experiments/run_experiments.py --only retrievers chunking prefix
python experiments/run_experiments.py --only rerank        # nagy modellt tölt le (~2 GB)
python experiments/run_experiments.py summary              # csak az összesítő
```

Csoportok: `baseline` (chunk 500/100, cikkcím-előtaggal, dense), `retrievers` (BM25, BM25 5 karakteres szótövezéssel, hibrid dense + BM25 Reciprocal Rank Fusion-nel), `chunking` (300/50, 800/150, 1000/200, 500/0, 500/200), `prefix` (cikkcím-előtag nélkül), `rerank` (cross-encoder, `BAAI/bge-reranker-v2-m3`, top-20). A futtató a hiányzó gyűjteményeket felépíti, a kész méréseket kihagyja (`--force` újramér, `--reingest` újraépít), így megszakítás után folytatható. A teljes futás kb. 25–30 perc a reranker letöltése nélkül.

Kimenet: `experiments/results/<név>.json` (kérdésenként is), valamint `summary.md` és `summary.csv` (chunk MRR szerint rendezve). A mérések naplója az `experiments/runs/` mappába kerül (nem commitolandó).

## Válaszgenerálás (Ollama)

A válaszokat helyi, CPU-n futó kis modell generálja az [Ollama](https://ollama.com/download) segítségével (nem kell hozzá GPU vagy API-kulcs).

```powershell
# 1. Ollama telepítése (Windows-telepítő), majd modell letöltése (kb. 3 GB)
ollama pull gemma3:4b

# 2. gyors próba kontextus nélkül
python -m rag.generate "Hány helyiségből áll a keszthelyi Festetics-kastély?"

# 3. kiértékelés: először néhány kérdéssel, utána mind az 50-nel (futó Qdrant is kell)
python eval/run_generation_eval.py --limit 5
python eval/run_generation_eval.py
```

Minden kérdésre két válasz készül: **rag** (a kereső top-k chunkja a promptban, "csak a forrásokból válaszolj" utasítással) és **norag** (ugyanaz a kérdés kontextus nélkül, ez a baseline). Az alapértelmezett keresés a `hybrid_stem5`, k = 5; módosítható (`--retriever`, `--k`, `--collection`, `--model`). A futás folytatható, a kész válaszokat nem kéri le újra.

Kimenet az `experiments/generation/` mappában: a nyers válaszok (`.jsonl`) és egy `.csv` kézi értékeléshez (a `kézi` oszlopot kell kitölteni, pl. 1 / 0,5 / 0). A szkript automatikus jeleket is számol: az elvárt válasz szavainak szótő-egyezését, az elutasítás felismerését, és hogy a lekért kontextusban benne volt-e a bizonyíték. Utóbbi választja szét a keresési hibát a generálási hibától. A "nincs válasz" kérdéseknél a kitalált (nem elutasított) válaszok aránya a hallucináció mérőszáma. Az automatikus jelek durva közelítések, a végső helyességi arány a kézi értékelésből számolandó.

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