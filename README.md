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
**Méret:** 498 cikk (a 509 gyűjtöttből 9 túl rövid, 300 karakter alatti, ezt kihagytuk).

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
- Docker Desktop (Qdrant)
- Generáláshoz: [Ollama](https://ollama.com/download) (helyben futó modell, GPU és API-kulcs nélkül)

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
```

A generáláshoz az Ollamát a gépre kell telepíteni.

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

A válaszokat helyben futó modell generálja az [Ollama](https://ollama.com/download) segítségével (CPU-n, GPU és API-kulcs nélkül). Az Ollamát a gépre kell telepíteni, a modell az Ollama saját mappájába töltődik le, nem a repóba.

```powershell
# 1. Ollama telepítése (Windows-telepítő), majd modell letöltése
ollama pull gemma3:4b      # kb. 3 GB, gyors, de gyenge
ollama pull gemma3:12b     # kb. 8 GB, lassabb, sokkal jobb (kb. 16 GB RAM-tól)

# 2. gyors próba kontextus nélkül
python -m rag.generate "Hány helyiségből áll a keszthelyi Festetics-kastély?"

# 3. kiértékelés: először néhány kérdéssel, utána mind az 50-nel (futó Qdrant is kell)
python eval/run_generation_eval.py --limit 5
python eval/run_generation_eval.py --prompt balanced --model gemma3:12b
```

Minden kérdésre két válasz készül: **rag** (a kereső top-k chunkja a promptban, a forrásokból válaszolj utasítással) és **norag** (ugyanaz a kérdés kontextus nélkül, ez a baseline). Alapértelmezett keresés: `hybrid_stem5`, k = 5. Kapcsolók: `--model`, `--retriever`, `--k`, `--collection`, `--prompt {strict,balanced}`. A `strict` prompt szigorúan előírja az elutasító mondatot, ha a válasz nem derül ki a forrásokból, a `balanced` enyhébb: ha bármelyik forrásban szerepel a válasz, azt kell használni, és csak akkor szabad elutasítani, ha tényleg nincs benne. A futás folytatható, a kész válaszokat nem kéri le újra, a fájlnévben benne van a modell, a gyűjtemény, a kereső, a `k` és a prompt.

Kimenet az `experiments/generation/` mappában: a nyers válaszok (`.jsonl`) és egy `.csv` kézi értékeléshez (a `kézi` oszlop: 1 / 0,5 / 0). Az automatikus jelek (az elvárt válasz szavainak szótő-egyezése, az elutasítás felismerése, és hogy a lekért kontextusban benne volt-e a bizonyíték) durva közelítések, a végső helyességi arány az értékelésből számolandó.

## Eredmények

Minden mérés 50 kérdésen készült, ebből 44 válaszolható és 6 `nincs válasz` típusú. Egy kérdés a válaszolhatóknál kb. 2,3 százalékpont, ezért pár kérdésnyi különbség a zaj határán belül van.

### Retrieval (chunkszint, 44 válaszolható kérdés)

Rendezve chunk MRR szerint. Ha nincs más jelölve, a keresés dense (embedding), a gyűjtemény `corpus` (chunk 500 / átfedés 100, cikkcím-előtaggal). A teljes táblázat: `experiments/results/summary.md`.

| Konfiguráció | R@1 | R@5 | MRR |
|---|---|---|---|
| chunk 300 / 50 | 0,826 | 0,958 | 0,917 |
| hibrid (dense + BM25, szótövezéssel) | 0,826 | 0,955 | 0,904 |
| chunk 1000 / 200 | 0,792 | 0,958 | 0,890 |
| chunk 500 / 200 | 0,792 | 0,981 | 0,885 |
| chunk 800 / 150 | 0,701 | 0,981 | 0,856 |
| BM25 szótövezéssel | 0,758 | 0,890 | 0,855 |
| **baseline** (chunk 500 / 100) | 0,735 | 0,955 | 0,849 |
| hibrid szótövezés nélkül | 0,761 | 0,886 | 0,827 |
| chunk 500 / 0 | 0,655 | 0,989 | 0,805 |
| cikkcím-előtag nélkül | 0,689 | 0,811 | 0,758 |
| BM25 szótövezés nélkül | 0,648 | 0,769 | 0,713 |

- A cikkcím-előtag az embeddingben sokat számít (MRR 0,849 → 0,758 nélküle).
- A BM25-höz magyar toldalékolás miatt szótövezés kell (5 karakteres csonkolás: MRR 0,713 → 0,855). A szótövezett hibrid a legjobb keresési módszer a baseline gyűjteményen, a szótövezés nélküli hibrid viszont rosszabb a dense-nél.
- Az átfedés növelése javítja az eredményt, de ebben részben a mérőszám torzít: átfedésnél ugyanaz az idézet több chunkban is szerepel, így több chunk számít relevánsnak.
- A cikkszintű Recall@5 szinte minden konfigurációnál 1,0 közeli, ezért a chunkszintű R@1 és MRR az informatív mérték.
- A reranker (`bge-reranker-v2-m3`) mérése nem sikerült: a modell letöltése hibára futott a fejlesztői gépen.

### Generálás (44 válaszolható + 6 `nincs válasz` kérdés)

Automatikus jelek, `hybrid_stem5` kereséssel, k = 5 (ahol más nincs jelölve):

| Modell és prompt | Helyes válasz (aut.) | Elutasított (válaszolható) | `nincs válasz` helyesen elutasítva | RAG-válasz ideje |
|---|---|---|---|---|
| gemma3:4b, strict | 17 (39%) | 25 (57%) | 6/6 | 12 mp |
| gemma3:4b, balanced | 21 (48%) | 20 (45%) | 6/6 | 11 mp |
| gemma3:4b, balanced, k = 3 | 19 (43%) | 22 (50%) | 6/6 | 6 mp |
| gemma3:12b, balanced | 38 (86%) | 1 (2%) | 5/6 | 37 mp |

Baseline (kontextus nélkül): a 4B modell 0 helyes választ ad, a kérdések 86%-át elutasítja ("Nem tudom"). A 12B modell 2 helyes választ ad, szinte semmit nem utasít el, és mind a 6 `nincs válasz` kérdésre kitalál egy választ.

Értékelés a gemma3:12b futásra (1 = helyes, 0,5 = részben helyes, 0 = téves vagy elutasított):

| | RAG | Baseline |
|---|---|---|
| Teljesen helyes válaszolható kérdés (44) | 40 (91%) | 1 (2%) |
| Pontszám összesen | 41,0 / 44 (93%) | 3,0 / 44 (7%) |
| `nincs válasz` helyesen kezelve (6) | 5, plusz 1 részleges | 0 |

- A modellméret hatása a legnagyobb: ugyanazokkal a forrásokkal és prompttal a 4B-ről 12B-re váltva a helyes válaszok aránya 48%-ról 86%-ra nőtt (automatikus jel), és a téves elutasítások szinte megszűntek.
- A kis modell a forrásban szó szerint szereplő választ is gyakran elutasítja (a lekért kontextusban 42/44 esetben benne volt a bizonyíték), ezért a 6/6 helyes elutasítás a 4B-nél félrevezető: a modell majdnem mindent elutasít.
- A kontextus nélküli (baseline) nagy modell magabiztosan hallucinál, a RAG ezt jelentősen csökkenti.
- A prompt enyhítése és a `k` csökkentése a 4B modellnél csak a zaj határát megközelítő különbséget hozott. Ugyanazon kérdésre a modell futásonként eltérően válaszolhat.
- A többcikkes kérdéseknél az automatikus jel túlbecsülhet, ezért érdemes az értékelést kézzel is végezni.

## Repóstruktúra

```
scripts/        letöltő szkript (download_corpus.py)
src/rag/        a pipeline forráskódja (betöltés, chunkolás, embedding, keresés, retrieval-változatok, generálás)
data/           korpusz (nyers adat nem commitolt)
eval/           kérdés-válasz értékelő készlet és metrikák
experiments/    kísérletfuttató, retrieval-eredmények (results/) és generált válaszok (generation/)
docs/           rövid dokumentáció, jegyzetek
tests/          tesztek
```