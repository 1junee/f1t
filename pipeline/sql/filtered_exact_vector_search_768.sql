-- 개인 서비스용: 기존 팀 RPC/테이블/인덱스는 변경하지 않는다.
-- 조건을 만족하는 모든 상품을 DB 안에서 정확 코사인 점수로 정렬한다.
-- ID순 1,000개 제한과 벡터 다운로드가 없으며, ANN 근사 검색이 아니다.
create or replace function public.filtered_exact_vector_search_768(
  p_query_embedding extensions.vector,
  p_match_count integer default 10,
  p_table_filter text default null,
  p_category2_filter text default null,
  p_category2_keyword_filter text default null,
  p_fabric_embedding extensions.vector default null,
  p_metadata_filters jsonb default '{}'::jsonb
)
returns jsonb
-- 같은 요청의 필터 조회와 순위 계산은 동일한 DB 스냅샷을 읽는다.
language plpgsql stable security invoker
set search_path = public, extensions
as $function$
declare
  started timestamptz := clock_timestamp();
  step_started timestamptz;
  filter_ms double precision := 0;
  rank_ms double precision := 0;
  matched_count integer := 0;
  result_count integer := least(greatest(coalesce(p_match_count, 10), 1), 100);
  table_name text;
  field text;
  predicate text;
  ids text[];
  table_rows jsonb;
  combined jsonb := '[]'::jsonb;
  results jsonb;
  sleeve_column text;
  length_column text;
begin
  if p_query_embedding is null or vector_dims(p_query_embedding) <> 768
      or vector_norm(p_query_embedding) = 0 then
    raise exception 'query embedding must be a nonzero 768-dimensional vector';
  end if;
  if p_fabric_embedding is not null and
      (vector_dims(p_fabric_embedding) <> 768 or vector_norm(p_fabric_embedding) = 0) then
    raise exception 'fabric embedding must be a nonzero 768-dimensional vector';
  end if;
  if p_table_filter is not null and p_table_filter not in
      ('musinsa_top_clothes','musinsa_pants','musinsa_skirt_dress') then
    raise exception 'invalid table filter';
  end if;
  if p_metadata_filters is null or jsonb_typeof(p_metadata_filters) <> 'object' then
    raise exception 'metadata filters must be an object';
  end if;
  -- 필터값은 모두 배열이다. 컬럼은 허용 목록으로 제한하고 값은 USING으로 바인딩한다.
  for field in select jsonb_object_keys(p_metadata_filters) loop
    if field not in ('sleeve','length','sex','season','stretch','thickness','fit') then
      raise exception 'unsupported metadata field';
    end if;
    if jsonb_typeof(p_metadata_filters->field) <> 'array' then
      raise exception 'metadata filter values must be arrays';
    end if;
    if jsonb_array_length(p_metadata_filters->field) = 0 or exists (
      select 1 from jsonb_array_elements(p_metadata_filters->field) v
      where jsonb_typeof(v) <> 'string'
    ) then
      raise exception 'metadata filter values must be nonempty string arrays';
    end if;
  end loop;

  foreach table_name in array array['musinsa_top_clothes','musinsa_pants','musinsa_skirt_dress'] loop
    if p_table_filter is not null and table_name <> p_table_filter then continue; end if;
    -- 없는 속성을 다른 카테고리에서 만족한 것처럼 처리하지 않는다.
    if (table_name = 'musinsa_top_clothes' and p_metadata_filters ? 'length')
        or (table_name = 'musinsa_pants' and p_metadata_filters ? 'sleeve') then
      continue;
    end if;
    predicate := 't.gemini_image_embedding_768 is not null
      and ($2 is null or t.category2 = $2)
      and ($3 is null or t.category2 ilike ''%'' || $3 || ''%'')';
    for field in select jsonb_object_keys(p_metadata_filters) loop
      if field in ('sleeve','length') then
        predicate := predicate || format(
          ' and t.%I = any(array(select jsonb_array_elements_text($1->%L)))', field, field);
      elsif field = 'sex' and jsonb_array_length(p_metadata_filters->field) > 1 then
        -- 공용: 남성/여성이 모두 있어야 한다. 다른 배열 조건은 하나만 겹쳐도 된다.
        predicate := predicate || ' and t.sex @> array(select jsonb_array_elements_text($1->''sex''))';
      else
        predicate := predicate || format(
          ' and t.%I && array(select jsonb_array_elements_text($1->%L))', field, field);
      end if;
    end loop;

    step_started := clock_timestamp();
    execute format('select array_agg(t.id) from public.%I t where %s', table_name, predicate)
      into ids using p_metadata_filters, p_category2_filter, p_category2_keyword_filter;
    filter_ms := filter_ms + extract(epoch from clock_timestamp()-step_started)*1000;
    matched_count := matched_count + coalesce(cardinality(ids), 0);
    if coalesce(cardinality(ids), 0) = 0 then continue; end if;

    -- 필터 단계에는 벡터를 읽지 않고, 순위 계산 단계에 필요한 이미지/소재 벡터만 읽는다.
    -- 컬럼을 직접 지정하므로 벡터가 JSON 응답에 섞이지 않는다.
    sleeve_column := case when table_name='musinsa_pants' then 'null::text' else 't.sleeve' end;
    length_column := case when table_name='musinsa_top_clothes' then 'null::text' else 't.length' end;
    step_started := clock_timestamp();
    execute format($query$
      select coalesce(jsonb_agg(to_jsonb(r) order by r.similarity desc, r.id), '[]'::jsonb)
      from (
        select t.id, %L::text as source_table, t.name, t.brand, t.image_url,
          t.category1, t.category2, t.price, t.color, %s as sleeve, %s as length,
          t.sex, t.season, t.stretch, t.thickness, t.fit, t.fabric,
          (case when $3 is null then 1.0 else 0.8 end) *
            (case when vector_norm(t.gemini_image_embedding_768)=0 then 0
              else 1-(t.gemini_image_embedding_768 <=> $2) end)
          + case when $3 is null or t.gemini_fabric_text_embedding_768 is null then 0
              when vector_norm(t.gemini_fabric_text_embedding_768)=0 then 0
              else 0.2*(1-(t.gemini_fabric_text_embedding_768 <=> $3)) end as similarity
        from public.%I t where t.id = any($1)
        order by similarity desc, t.id limit $4
      ) r
    $query$, table_name, sleeve_column, length_column, table_name)
      into table_rows using ids, p_query_embedding, p_fabric_embedding, result_count;
    rank_ms := rank_ms + extract(epoch from clock_timestamp()-step_started)*1000;
    combined := combined || table_rows;
  end loop;
  select coalesce(jsonb_agg(r.value order by (r.value->>'similarity')::double precision desc,
    r.value->>'source_table', r.value->>'id'), '[]'::jsonb) into results
  from (
    select value from jsonb_array_elements(combined)
    order by (value->>'similarity')::double precision desc, value->>'source_table', value->>'id'
    limit result_count
  ) r;
  return jsonb_build_object('results', results, 'diagnostics', jsonb_build_object(
    'strategy','filtered_exact_cosine', 'matched_count',matched_count,
    'returned_count',jsonb_array_length(results),
    'db_filter_ms',round(filter_ms::numeric,2), 'db_rank_ms',round(rank_ms::numeric,2),
    'db_total_ms',round((extract(epoch from clock_timestamp()-started)*1000)::numeric,2)
  ));
end;
$function$;

-- 브라우저에서 직접 호출하지 않고 백엔드의 service_role만 실행할 수 있다.
revoke all on function public.filtered_exact_vector_search_768(
  extensions.vector,integer,text,text,text,extensions.vector,jsonb
) from public, anon, authenticated;
grant execute on function public.filtered_exact_vector_search_768(
  extensions.vector,integer,text,text,text,extensions.vector,jsonb
) to service_role;
