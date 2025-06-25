# API Documentation

This document describes the main API endpoints for the Daylight server.  
Example requests are provided in both **Python** (using `requests`) and **TypeScript** (using `fetch`). Use [the example notebook](./example/1.ipynb) for playing with the code.

---

## `/to_values`  
**POST** `/to_values`  
Converts an RGB daylight estimated image to DF (daylight factor) values.

### Request
- **Content-Type:** `multipart/form-data`
- **Body:**  
  - `file`: The image file (PNG/JPG) to convert.

### Response
- **200 OK**  
  ```json
  {
    "success": true,
    "content": "<base64-encoded DF values>"
  }
  ```
- **400/422/500** on error

### Python Example
```python
import requests
import numpy as np
import base64
import io

imgname = "floorplan0_s.png"
with open(".assets/W_RN1018/{}".format(imgname), "rb") as f:
    files = {"file": (imgname, f)}
    resp = requests.post("http://127.0.0.1:8081/to_values", files=files)
    
k = resp.json()["content"]
# k = base64.b64decode(k)
img = np.load(io.BytesIO(base64.b64decode(k)))
```

### TypeScript Example
```typescript
const form = new FormData();
form.append("file", new Blob([yourImageBlob]), "image.png");

fetch("http://localhost:8081/to_values", {
  method: "POST",
  body: form
})
  .then(res => res.json())
  .then(console.log);
```

---

## `/to_rgb`  
**POST** `/to_rgb`  
Converts DF values (as a base64-encoded NumPy array) to an RGB image.

### Request
- **Content-Type:** `multipart/form-data`
- **Body:**  
  - `file`: The base64-encoded `.npy` file containing DF values.

### Response
- **200 OK**  
  ```json
  {
    "success": true,
    "content": "<base64-encoded RGB image>"
  }
  ```
- **400/422/500** on error

### Python Example
```python
import requests
import numpy as np
import io
import base64

arr = np.ones((8, 8), dtype=np.float32)
buf = io.BytesIO()
np.save(buf, arr)
buf.seek(0)
arr_b64 = base64.b64encode(buf.read())
files = {"file": ("values.b64", io.BytesIO(arr_b64))}
resp = requests.post("http://localhost:8081/to_rgb", files=files)
print(resp.json())
```

### TypeScript Example
```typescript
const arrB64 = ... // base64 string of your .npy file
const form = new FormData();
form.append("file", new Blob([Uint8Array.from(atob(arrB64), c => c.charCodeAt(0))]), "values.b64");

fetch("http://localhost:8081/to_rgb", {
  method: "POST",
  body: form
})
  .then(res => res.json())
  .then(console.log);
```

---

## `/get_stats`  
**POST** `/get_stats`  
Calculates statistics from a daylight estimation image.

### Request
- **Content-Type:** `multipart/form-data`
- **Body:**  
  - `file`: The image file (PNG/JPG) to analyze.

### Response
- **200 OK**  
  ```json
  {
    "metrics": {
      "average_value": 0.95,
      "ratio_gt1": 0.42
    },
    "success": true
  }
  ```
- **400/422/500** on error

### Python Example
```python
import requests

with open("image.png", "rb") as f:
    files = {"file": ("image.png", f)}
    resp = requests.post("http://localhost:8081/get_stats", files=files)
    print(resp.json())
```

### TypeScript Example
```typescript
const form = new FormData();
form.append("file", new Blob([yourImageBlob]), "image.png");

fetch("http://localhost:8081/get_stats", {
  method: "POST",
  body: form
})
  .then(res => res.json())
  .then(console.log);
```

---

## `/get_df`  
**POST** `/get_df`  
Runs the full pipeline for a set of images and transformation parameters, returning an estimation matrix.

### Request
- **Content-Type:** `multipart/form-data`
- **Body:**  
  - `file`: Multiple image files (PNG/JPG), one per view (use the same field name for each).
  - `rotation`: JSON array of rotation values, e.g. `[0,0,0]`
  - `translation`: JSON object for translation, e.g. `{"x":0,"y":0}`

### Response
- **200 OK**  
  ```json
  {
    "content": "<base64-encoded result image>",
    "success": true
  }
  ```
- **400/422/500** on error

### Python Example
```python
import requests
from werkzeug.datastructures import MultiDict
import io

files = [
    ("file", ("img0.png", open("img0.png", "rb"))),
    ("file", ("img1.png", open("img1.png", "rb"))),
    ("file", ("img2.png", open("img2.png", "rb")))
]
data = {
    "rotation": "[0,0,0]",
    "translation": '{"x":0,"y":0}'
}
resp = requests.post("http://localhost:8081/get_df", files=files, data=data)
print(resp.json())
```

### TypeScript Example
```typescript
const form = new FormData();
form.append("file", new Blob([img0Blob]), "img0.png");
form.append("file", new Blob([img1Blob]), "img1.png");
form.append("file", new Blob([img2Blob]), "img2.png");
form.append("rotation", "[0,0,0]");
form.append("translation", '{"x":0,"y":0}');

fetch("http://localhost:8081/get_df", {
  method: "POST",
  body: form
})
  .then(res => res.json())
  .then(console.log);
```

---

## Notes

- All endpoints return JSON responses.
- For file uploads, always use `multipart/form-data`.
- For `/get_df`, you must send multiple files with the same field name (`file`).
- Error responses will include `"success": false` and may include an `"error"` message.

---