-- Benchmark-only exhaustive baseline for retrieval latency experiments.
-- Apply to a non-production benchmark database. The function disables index and
-- bitmap scans transaction-locally so all stored image embeddings are compared.
-- 입력은 질의 벡터와 반환 개수, 출력은 상품·유사도·전체 비교 개수다. 운영 검색용 함수와 구분한다.

create or replace function public.benchmark_full_scan_fashion_items_768(
  p_query_embedding extensions.vector,
  p_match_count integer default 10
)
returns table(
  id text,
  source_table text,
  name text,
  brand text,
  image_url text,
  category1 text,
  category2 text,
  price numeric,
  color text,
  sleeve text,
  length text,
  sex text,
  season text,
  stretch text,
  thickness text,
  fit text,
  fabric text,
  similarity double precision,
  candidate_count bigint
)
language plpgsql
volatile
security invoker
set search_path = public, extensions
set statement_timeout = '60s'
as $function$
begin
  -- 이 벤치마크 호출의 트랜잭션에서만 인덱스 사용을 끄고 전수 비교한다.
  perform set_config('enable_indexscan', 'off', true);
  perform set_config('enable_indexonlyscan', 'off', true);
  perform set_config('enable_bitmapscan', 'off', true);

  return query
  with products as materialized (
    -- 세 테이블의 이미지 벡터를 하나의 중간 결과로 모은다. 이 기준선에는 속성 필터가 없다.
    select
      t.id::text,
      'musinsa_top_clothes'::text as source_table,
      t.name::text,
      t.brand::text,
      t.image_url::text,
      t.category1::text,
      t.category2::text,
      t.price::numeric,
      t.color::text,
      t.sleeve::text,
      null::text as length,
      t.sex::text,
      t.season::text,
      t.stretch::text,
      t.thickness::text,
      t.fit::text,
      t.fabric::text,
      t.gemini_image_embedding_768 as image_embedding
    from public.musinsa_top_clothes t
    where t.gemini_image_embedding_768 is not null

    union all

    select
      t.id::text,
      'musinsa_pants'::text,
      t.name::text,
      t.brand::text,
      t.image_url::text,
      t.category1::text,
      t.category2::text,
      t.price::numeric,
      t.color::text,
      null::text,
      t.length::text,
      t.sex::text,
      t.season::text,
      t.stretch::text,
      t.thickness::text,
      t.fit::text,
      t.fabric::text,
      t.gemini_image_embedding_768
    from public.musinsa_pants t
    where t.gemini_image_embedding_768 is not null

    union all

    select
      t.id::text,
      'musinsa_skirt_dress'::text,
      t.name::text,
      t.brand::text,
      t.image_url::text,
      t.category1::text,
      t.category2::text,
      t.price::numeric,
      t.color::text,
      t.sleeve::text,
      t.length::text,
      t.sex::text,
      t.season::text,
      t.stretch::text,
      t.thickness::text,
      t.fit::text,
      t.fabric::text,
      t.gemini_image_embedding_768
    from public.musinsa_skirt_dress t
    where t.gemini_image_embedding_768 is not null
  ),
  -- 이미지 유사도만 비교하는 기준선이다. 운영 검색의 소재 가중 점수는 포함하지 않는다.
  scored as (
    select
      products.*,
      (1 - (products.image_embedding <=> p_query_embedding))::double precision
        as similarity,
      -- LIMIT 적용 전 전체 비교 상품 수를 각 결과 행에 함께 싣는다.
      count(*) over () as candidate_count
    from products
  )
  select
    scored.id,
    scored.source_table,
    scored.name,
    scored.brand,
    scored.image_url,
    scored.category1,
    scored.category2,
    scored.price,
    scored.color,
    scored.sleeve,
    scored.length,
    scored.sex,
    scored.season,
    scored.stretch,
    scored.thickness,
    scored.fit,
    scored.fabric,
    scored.similarity,
    scored.candidate_count
  from scored
  order by scored.similarity desc
  -- 점수를 매긴 뒤 반환 개수만 줄인다. 따라서 LIMIT이 비교한 전체 상품 수를 줄이지는 않는다.
  limit least(greatest(coalesce(p_match_count, 10), 1), 100);
end;
$function$;

comment on function public.benchmark_full_scan_fashion_items_768(
  extensions.vector, integer
) is 'Benchmark-only exact cosine baseline with HNSW/index scans disabled.';

revoke all on function public.benchmark_full_scan_fashion_items_768(
  extensions.vector, integer
) from public, anon, authenticated;

grant execute on function public.benchmark_full_scan_fashion_items_768(
  extensions.vector, integer
) to service_role;

