-- Canonical definition of the live Supabase vector-search RPC.
-- Apply through a reviewed Supabase migration before production use.
-- 요청 시 만든 설명문 벡터를 입력받아, 이미 저장된 상품 벡터와 비교하고 상품 행을 반환한다.
-- p_fabric_embedding이 NULL이면 이미지 점수만, 값이 있으면 이미지·소재 가중 점수를 쓴다.

create or replace function public.approximate_vector_search_768(
  p_query_embedding extensions.vector,
  p_match_count integer default 10,
  p_table_filter text default null,
  p_category2_filter text default null,
  p_category2_keyword_filter text default null,
  p_fabric_embedding extensions.vector default null
)
returns table(
  -- 모든 상품 테이블이 이 반환 형식을 공유한다. 없는 속성은 아래 SELECT에서 NULL로 맞춘다.
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
  similarity double precision
)
language plpgsql
volatile
security invoker
set search_path = public, extensions
set statement_timeout = '30s'
as $function$
begin
  -- pgvector defaults to ef_search=40. This RPC needs up to 100 image
  -- candidates, so widen the HNSW search only for the current transaction.
  perform set_config('hnsw.ef_search', '100', true);
  -- 이 값은 그래프 내부 탐색 폭이다. 아래 candidate_count(재정렬할 행 수)와 의미가 다르다.

  return query
  -- 최종 반환 수와 재정렬할 후보 수는 별개다. 둘 다 이 함수에서 최대 100개로 제한한다.
  with params as (
    select
      result_count,
      least(greatest(result_count * 5, 50), 100) as candidate_count
    from (
      select least(greatest(coalesce(p_match_count, 10), 1), 100) as result_count
    ) bounds
  ),
  -- 세 카테고리는 같은 방식으로 이미지 거리순 후보를 고른다.
  -- 거리순 ORDER BY + LIMIT이 이미지 HNSW 인덱스를 활용할 수 있는 검색 부분이다.
  top_clothes as (
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
      -- <=>는 코사인 거리다. 1에서 빼면 클수록 유사한 점수가 된다.
      -- 요청 소재가 있되 상품 소재 벡터가 없으면 소재 점수만 0으로 처리한다.
      case
        when p_fabric_embedding is null then
          (1 - (t.gemini_image_embedding_768 <=> p_query_embedding))::double precision
        else
          (
            0.8 * (1 - (t.gemini_image_embedding_768 <=> p_query_embedding))
            + 0.2 * coalesce(
                1 - (t.gemini_fabric_text_embedding_768 <=> p_fabric_embedding),
                0
              )
          )::double precision
      end as similarity
    from public.musinsa_top_clothes t
    where t.gemini_image_embedding_768 is not null
      -- 필터 인자가 NULL이면 해당 제한을 적용하지 않는다. category2는 정확 일치 또는 부분 일치다.
      and (p_table_filter is null or p_table_filter = 'musinsa_top_clothes')
      and (p_category2_filter is null or t.category2 = p_category2_filter)
      and (
        p_category2_keyword_filter is null
        or t.category2 ilike '%' || p_category2_keyword_filter || '%'
      )
    order by t.gemini_image_embedding_768 <=> p_query_embedding
    limit (select candidate_count from params)
  ),
  pants as (
    select
      t.id::text,
      'musinsa_pants'::text as source_table,
      t.name::text,
      t.brand::text,
      t.image_url::text,
      t.category1::text,
      t.category2::text,
      t.price::numeric,
      t.color::text,
      null::text as sleeve,
      t.length::text,
      t.sex::text,
      t.season::text,
      t.stretch::text,
      t.thickness::text,
      t.fit::text,
      t.fabric::text,
      case
        when p_fabric_embedding is null then
          (1 - (t.gemini_image_embedding_768 <=> p_query_embedding))::double precision
        else
          (
            0.8 * (1 - (t.gemini_image_embedding_768 <=> p_query_embedding))
            + 0.2 * coalesce(
                1 - (t.gemini_fabric_text_embedding_768 <=> p_fabric_embedding),
                0
              )
          )::double precision
      end as similarity
    from public.musinsa_pants t
    where t.gemini_image_embedding_768 is not null
      and (p_table_filter is null or p_table_filter = 'musinsa_pants')
      and (p_category2_filter is null or t.category2 = p_category2_filter)
      and (
        p_category2_keyword_filter is null
        or t.category2 ilike '%' || p_category2_keyword_filter || '%'
      )
    order by t.gemini_image_embedding_768 <=> p_query_embedding
    limit (select candidate_count from params)
  ),
  skirt_dress as (
    select
      t.id::text,
      'musinsa_skirt_dress'::text as source_table,
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
      case
        when p_fabric_embedding is null then
          (1 - (t.gemini_image_embedding_768 <=> p_query_embedding))::double precision
        else
          (
            0.8 * (1 - (t.gemini_image_embedding_768 <=> p_query_embedding))
            + 0.2 * coalesce(
                1 - (t.gemini_fabric_text_embedding_768 <=> p_fabric_embedding),
                0
              )
          )::double precision
      end as similarity
    from public.musinsa_skirt_dress t
    where t.gemini_image_embedding_768 is not null
      and (p_table_filter is null or p_table_filter = 'musinsa_skirt_dress')
      and (p_category2_filter is null or t.category2 = p_category2_filter)
      and (
        p_category2_keyword_filter is null
        or t.category2 ilike '%' || p_category2_keyword_filter || '%'
      )
    order by t.gemini_image_embedding_768 <=> p_query_embedding
    limit (select candidate_count from params)
  ),
  -- 각 테이블에서 고른 후보를 합친 뒤, 소재가 반영된 최종 점수로 다시 정렬한다.
  candidates as (
    -- UNION ALL은 중복 제거 없이 세 결과를 붙인다. source_table로 같은 ID라도 출처를 구별한다.
    select * from top_clothes
    union all
    select * from pants
    union all
    select * from skirt_dress
  )
  select *
  from candidates
  order by candidates.similarity desc
  limit (select result_count from params);
end;
$function$;

comment on function public.approximate_vector_search_768(
  extensions.vector, integer, text, text, text, extensions.vector
) is 'Backend-only Gemini retrieval. Sets HNSW ef_search to 100, selects 50-100 image candidates, then applies optional 80/20 image/fabric reranking.';

-- 브라우저 공개 키로 직접 호출하지 않고 백엔드의 service_role만 실행하도록 제한한다.
revoke all on function public.approximate_vector_search_768(
  extensions.vector, integer, text, text, text, extensions.vector
) from public, anon, authenticated;

grant execute on function public.approximate_vector_search_768(
  extensions.vector, integer, text, text, text, extensions.vector
) to service_role;
