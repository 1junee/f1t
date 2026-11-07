import json
import os
import base64

# 실행하면 로컬 JSON·이미지를 읽고 HTML을 만든다. main 가드가 없어 import만 해도 아래 작업이 실행된다.
BASE = os.path.dirname(__file__)

def embed_images(results):
    # 이미지 경로를 data URL로 바꿔 HTML 한 파일에서 볼 수 있게 한다. 전달받은 목록 자체를 수정한다.
    for item in results:
        path = item.get("image_path")
        if path and os.path.exists(path):
            ext = path.rsplit(".", 1)[-1].lower()
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            item["image_path"] = f"data:{mime};base64,{b64}"
    return results

with open(os.path.join(BASE, "simple_test_results.json"), encoding="utf-8") as f:
    simple = embed_images(json.load(f))

with open(os.path.join(BASE, "image_test_results.json"), encoding="utf-8") as f:
    image = embed_images(json.load(f))

with open(os.path.join(BASE, "results_viewer.html"), encoding="utf-8") as f:
    html = f.read()

# 템플릿의 자리표시자에 결과를 넣는다. 외부 입력을 공유용 HTML에 넣으려면 별도 이스케이프가 필요하다.
html = html.replace("__SIMPLE__", json.dumps(simple, ensure_ascii=False))
html = html.replace("__IMAGE__",  json.dumps(image,  ensure_ascii=False))

out = os.path.join(BASE, "results_viewer_built.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Built: {out}")
