# F1T Backend

This directory contains the HTTP API layer only. The fashion recommendation
workflow lives in `../pipeline/recommendation_service.py`.

Run the local API from the repository root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Runtime credentials remain in `pipeline/.env` because they are consumed by the
search pipeline.

## 공개 데모의 모델 요청 제한

`/search`와 `/analyze`는 같은 요청 한도를 공유한다. 모델 호출 직전에
한도를 확인하며, 초과하면 모델을 호출하지 않고 HTTP `429`와
`Retry-After`(초)를 반환한다. 요청을 시작한 뒤 발생한 실패도 사용량에
포함하지만 동시 실행 슬롯은 성공·실패 모두 반환한다. 입력 검증에서
거절한 요청과 한도 초과 요청은 사용량에 포함하지 않는다.

| 환경변수 | 기본값 | 의미 |
| --- | ---: | --- |
| `MODEL_MAX_CONCURRENT_REQUESTS` | 2 | 동시에 실행할 수 있는 모델 요청 수 |
| `MODEL_REQUESTS_PER_MINUTE` | 10 | 최근 60초 동안 시작한 요청 수 |
| `MODEL_REQUESTS_PER_DAY` | 100 | 최근 24시간 동안 시작한 요청 수 |

세 값은 양의 정수이며, 잘못된 값은 경고 로그와 함께 기본값을 사용한다.
`backend/.env.example`의 항목을 기존 `pipeline/.env` 또는 서버 환경변수에
설정하고 서버를 재시작한다. `backend/.env`는 자동으로 읽지 않는다.

이 설정은 **API 요청 수 제한**이다. 검색 한 번에 여러 모델 호출과
fallback이 발생할 수 있고 이미지·토큰 수·모델 단가도 달라지므로,
달러나 원화의 지출 상한을 보장하지 않는다.

카운터는 단일 프로세스 메모리에 있으며 사용자/IP별로 나누지 않는다.
작은 데모는 `uvicorn backend.api:app --workers 1 --host 0.0.0.0 --port 8000`으로
실행한다. 서버 재시작 시 카운터가 초기화되고, 여러 worker나 replica를
실행하면 각각 별도 한도가 생긴다. 여러 인스턴스에서 전역 한도를
유지하려면 공유 저장소 기반 제한이 필요하다.

`/status`는 모델명과 키/URL의 설정 여부만 반환하며 외부 모델이나 DB를
호출하지 않는다. 응답의 `embedding: "not_checked"`와
`check_mode: "configuration_only"`는 실제 연결 성공을 뜻하지 않는다.
`/health`는 기존의 서버 응답 확인 용도를 유지한다. `/models` 응답 형식은
유지하며, `/search`의 `provider`는 `gemini`, `openai`, `qwen`만 허용한다.
