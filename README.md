# llm-rag-project

RAG-rendszer építése és kiértékelése egy választott korpuszon. A projekt a *Nagy nyelvi modellek* tárgy beadandója (b) iránya.

> **Állapot:** kezdeti váz. A korpusz még nincs kiválasztva.

## Cél

Egy reprodukálható RAG-pipeline, plusz egy kísérleti rész, amely összehasonlítja a változatait:

- chunkméret és átfedés hatása,
- BM25 vs. embedding vs. hibrid keresés,
- reranker igen/nem,
- retrieval minősége (recall@k, MRR) és a generált válaszok hűsége.

## Korpusz

_TODO: a választott korpusz megnevezése, forrása, licence, mérete._

A nyers adat nincs a repóban, letöltő szkript tölti le (`scripts/download_corpus.py`, még nincs megírva).

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

## Repóstruktúra

```
src/rag/        a pipeline forráskódja (betöltés, chunkolás, embedding, keresés, generálás)
data/           korpusz (nyers adat nem commitolt)
eval/           kérdés-válasz értékelő készlet és metrikák
experiments/    kísérleti konfigurációk és összesítő eredmények
docs/           rövid dokumentáció, jegyzetek
tests/          tesztek
```
