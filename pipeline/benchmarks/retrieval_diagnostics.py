"""반팔 질의의 조회 구간만 재현한다. 기본은 새 RPC, --legacy는 기존 전송 방식 비교.

모델 생성 시간을 섞지 않도록 한 질의 벡터를 모든 경로에서 재사용한다.
실제 Supabase를 읽는다. --legacy는 벡터 1,000개(약 18MB)를 다운로드한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import urllib.request

from pipeline.benchmarks.legacy_vector_ranking import _rank_stored_gemini_vectors
from pipeline.vector_db_client import SupabaseVectorSearchClient

DESCRIPTION = "A plain white short-sleeve T-shirt."


def legacy_probe(client, embedding):
    """수정 전의 select * + 100개씩 10회 요청을 비교 실험에만 보존한다."""
    rows, pages = [], []
    started = perf_counter()
    for offset in range(0, 1000, 100):
        request = urllib.request.Request(
            f"{client.supabase_url}/rest/v1/musinsa_top_clothes"
            f"?select=*&sleeve=eq.short&order=id&offset={offset}&limit=100",
            headers={"apikey": client.supabase_key,
                     "Authorization": f"Bearer {client.supabase_key}"},
        )
        before = perf_counter()
        with urllib.request.urlopen(request, timeout=client.timeout) as response:
            headers_at = perf_counter()
            body = response.read()
            body_at = perf_counter()
        page = json.loads(body)
        parsed_at = perf_counter()
        rows.extend(page)
        pages.append({
            "offset": offset, "rows": len(page), "response_bytes": len(body),
            "headers_wait_ms": round((headers_at-before)*1000, 2),
            "body_read_ms": round((body_at-headers_at)*1000, 2),
            "json_decode_ms": round((parsed_at-body_at)*1000, 2),
        })
        if len(page) < 100:
            break
    before = perf_counter()
    ranked = _rank_stored_gemini_vectors(embedding, rows, 3) if rows else []
    rank_ms = (perf_counter()-before)*1000
    record = {
        "strategy": "legacy_rest_first_1000_then_python", "pages": pages,
        "elapsed_ms": round((perf_counter()-started)*1000, 2),
        "local_rank_ms": round(rank_ms, 2), "candidate_count": len(rows),
        "request_count": len(pages), "response_bytes": sum(p["response_bytes"] for p in pages),
        "top_results": [{"id": r["id"], "similarity": r["similarity"]} for r in ranked],
    }
    return record, {r["id"] for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-file", type=Path, help="재사용할 768차원 JSON 배열; 없으면 설명문을 한 번 임베딩")
    parser.add_argument("--legacy", action="store_true", help="기존 대용량 조회도 한 번 실행")
    parser.add_argument("--runs", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--output", type=Path, default=Path(".omx/state/retrieval/diagnostics.json"))
    args = parser.parse_args()
    client = SupabaseVectorSearchClient()
    if args.embedding_file:
        embedding = json.loads(args.embedding_file.read_text())
    else:
        from pipeline.retrieval.target_description_retrieval import RetrieveModule
        embedding = RetrieveModule().encode_text([DESCRIPTION])[0].tolist()
    client._rpc_for_dim(len(embedding))
    report = {"query": "긴팔 말고 흰색 반팔 티셔츠", "fixed_description": DESCRIPTION,
              "fixed_filters": {"sleeve": "short"}, "top_k": 3, "rpc_runs": []}
    legacy_ids = None
    if args.legacy:
        report["legacy"], legacy_ids = legacy_probe(client, embedding)
    for _ in range(args.runs):
        started = perf_counter()
        products, diagnostics = client.search_filtered_exact(
            embedding, 3, table_filter="musinsa_top_clothes", metadata_filters={"sleeve": "short"},
        )
        run = {"elapsed_ms": round((perf_counter()-started)*1000, 2), "diagnostics": diagnostics,
               "top_results": [{"id": r["id"], "similarity": r["similarity"]} for r in products]}
        if legacy_ids is not None:
            run["outside_legacy_first_1000"] = [r["id"] for r in products if r["id"] not in legacy_ids]
        report["rpc_runs"].append(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
