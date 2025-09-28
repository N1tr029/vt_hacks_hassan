import os
import json
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, Request, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from datetime import datetime
from pathlib import Path

# ---- paths ----
BASE_DIR = Path(__file__).resolve().parent          # final_hack/parking_app
PROJECT_ROOT = BASE_DIR.parent                      # final_hack/
REPO_ROOT = PROJECT_ROOT.parent                     # vt_hacks_2025/

RESULTS_DIR = REPO_ROOT / "results"
CROPS_DIR = REPO_ROOT / "crops"
GRIDS_DIR = REPO_ROOT / "grids"
IMAGES_DIR = REPO_ROOT / "images"
ANALYTICS_DIR = REPO_ROOT / "analytics"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

for folder in [RESULTS_DIR, CROPS_DIR, GRIDS_DIR, IMAGES_DIR, ANALYTICS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ---- FastAPI setup ----
app = FastAPI()
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---- CNN identifier ----
class SpotCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SpotCNN()
model_path = BASE_DIR / "spot_cnn.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

CLASS_NAMES = ["Free", "Occupied"]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_spot(crop):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]

# ---- analytics helpers ----
def get_analytics_path(lot_name):
    return ANALYTICS_DIR / f"{lot_name}_history.json"

def load_analytics(lot_name):
    path = get_analytics_path(lot_name)
    if path.exists():
        return json.loads(path.read_text())
    return {"checks": [], "spot_history": {}}

def save_analytics(lot_name, data):
    path = get_analytics_path(lot_name)
    path.write_text(json.dumps(data, indent=2))

def calculate_spot_analytics(spot_history):
    if not spot_history:
        return {
            "total_checks": 0,
            "occupied_count": 0,
            "free_count": 0,
            "occupancy_rate": 0,
            "turnover_count": 0,
            "avg_occupied_duration": 0,
            "utilization_score": 0,
            "current_status": "Unknown"
        }
    total = len(spot_history)
    occupied = sum(1 for h in spot_history if h["status"] == "Occupied")
    free = total - occupied
    occupancy_rate = round((occupied / total) * 100, 1)
    current_status = spot_history[-1]["status"]

    turnovers = sum(
        1 for i in range(1, len(spot_history))
        if spot_history[i]["status"] != spot_history[i-1]["status"]
    )

    occupied_streaks, current_streak = [], 0
    for h in spot_history:
        if h["status"] == "Occupied":
            current_streak += 1
        elif current_streak > 0:
            occupied_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        occupied_streaks.append(current_streak)

    avg_duration = round(sum(occupied_streaks) / len(occupied_streaks), 1) if occupied_streaks else 0
    turnover_rate = turnovers / total if total > 1 else 0
    utilization_score = round((occupancy_rate * 0.7) + (turnover_rate * 100 * 0.3), 1)

    return {
        "total_checks": total,
        "occupied_count": occupied,
        "free_count": free,
        "occupancy_rate": occupancy_rate,
        "turnover_count": turnovers,
        "avg_occupied_duration": avg_duration,
        "utilization_score": utilization_score,
        "current_status": current_status
    }

def find_trends(analytics_data):
    spot_stats = {sid: calculate_spot_analytics(hist)
                  for sid, hist in analytics_data.get("spot_history", {}).items()}
    if not spot_stats:
        return {
            "underutilized": [], "overutilized": [],
            "high_turnover": [], "low_turnover": [],
            "avg_occupancy": 0, "avg_turnover": 0
        }

    avg_occupancy = sum(s["occupancy_rate"] for s in spot_stats.values()) / len(spot_stats)
    avg_turnover = sum(s["turnover_count"] for s in spot_stats.values()) / len(spot_stats)

    underutilized = [ {"spot": int(sid), **stats}
                      for sid, stats in spot_stats.items()
                      if stats["occupancy_rate"] < avg_occupancy*0.7 and stats["total_checks"] >= 5 ]
    overutilized = [ {"spot": int(sid), **stats}
                     for sid, stats in spot_stats.items()
                     if stats["occupancy_rate"] > avg_occupancy*1.3 and stats["total_checks"] >= 5 ]
    high_turnover = [ {"spot": int(sid), "turnover_count": stats["turnover_count"]}
                      for sid, stats in spot_stats.items()
                      if stats["turnover_count"] > avg_turnover*1.5 and stats["total_checks"] >= 5 ]
    low_turnover = [ {"spot": int(sid), "turnover_count": stats["turnover_count"]}
                     for sid, stats in spot_stats.items()
                     if stats["turnover_count"] < avg_turnover*0.5 and stats["total_checks"] >= 5 ]

    return {
        "underutilized": sorted(underutilized, key=lambda x: x["occupancy_rate"])[:10],
        "overutilized": sorted(overutilized, key=lambda x: x["occupancy_rate"], reverse=True)[:10],
        "high_turnover": sorted(high_turnover, key=lambda x: x["turnover_count"], reverse=True)[:10],
        "low_turnover": sorted(low_turnover, key=lambda x: x["turnover_count"])[:10],
        "avg_occupancy": round(avg_occupancy, 1),
        "avg_turnover": round(avg_turnover, 1)
    }

# ---- routes ----
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# Upload only (no OpenCV GUI here)
@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...), lot_name: str = Form(...)):
    image_path = IMAGES_DIR / file.filename
    image_path.write_bytes(await file.read())
    return {
        "message": f"Uploaded image for lot {lot_name}",
        "lot_name": lot_name,
        "image_url": f"/images/{file.filename}"
    }

# Save rectangles drawn in the browser
@app.post("/admin/save_grid")
async def admin_save_grid(payload: dict = Body(...)):
    # Expect {"lot_name": "...", "rects": [{"x1":..,"y1":..,"x2":..,"y2":..}, ...]}
    lot_name = payload.get("lot_name")
    rects = payload.get("rects", [])
    if not lot_name or not rects:
        return JSONResponse({"error": "lot_name and rects required"}, status_code=400)

    spots = [(int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])) for r in rects]
    grid_path = GRIDS_DIR / f"{lot_name}.json"
    grid_path.write_text(json.dumps(spots, indent=2))
    return {"message": f"Saved {len(spots)} spots for lot {lot_name}"}

@app.get("/student", response_class=HTMLResponse)
async def student_page(request: Request):
    return templates.TemplateResponse("student.html", {"request": request})

@app.get("/lots")
async def list_lots():
    lots = [f.stem for f in GRIDS_DIR.glob("*.json")]
    return {"lots": lots}

@app.post("/student/check")
async def student_check(file: UploadFile = File(...), lot_name: str = Form(...)):
    image_path = IMAGES_DIR / file.filename
    image_path.write_bytes(await file.read())

    grid_path = GRIDS_DIR / f"{lot_name}.json"
    spots = json.loads(grid_path.read_text())

    img = cv2.imread(str(image_path))
    crop_dir = CROPS_DIR / lot_name
    crop_dir.mkdir(parents=True, exist_ok=True)

    free_count, total = 0, len(spots)
    spot_status, spot_images = [], []
    timestamp = datetime.now().isoformat()
    analytics_data = load_analytics(lot_name)

    for i, (x1, y1, x2, y2) in enumerate(spots):
        crop = img[y1:y2, x1:x2]
        crop_filename = f"spot_{i}.jpg"
        crop_path = crop_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)
        label = predict_spot(crop)
        spot_status.append(label)
        spot_images.append(f"/crops/{lot_name}/{crop_filename}")

        # update spot history
        spot_id = str(i)
        if spot_id not in analytics_data["spot_history"]:
            analytics_data["spot_history"][spot_id] = []
        analytics_data["spot_history"][spot_id].append({
            "timestamp": timestamp,
            "status": label
        })

        color = (0, 255, 0) if label == "Free" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        spot_number = str(i + 1)
        text_size = cv2.getTextSize(spot_number, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(img, spot_number, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if label == "Free":
            free_count += 1

    empty_spots = round(100 * (free_count / total), 2)
    taken_spots = round(100 * ((total - free_count) / total))

    analytics_data["checks"].append({
        "timestamp": timestamp,
        "free_count": free_count,
        "occupied_count": total - free_count,
        "total": total
    })

    for spot_id in analytics_data["spot_history"]:
        if len(analytics_data["spot_history"][spot_id]) > 1000:
            analytics_data["spot_history"][spot_id] = analytics_data["spot_history"][spot_id][-1000:]

    save_analytics(lot_name, analytics_data)

    annotated_path = RESULTS_DIR / file.filename
    cv2.imwrite(str(annotated_path), img)

    grid_size = int(total ** 0.5) + 1
    cell_size, padding = 60, 5
    grid_img = 255 * np.ones((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)

    for idx, status in enumerate(spot_status):
        row, col = divmod(idx, grid_size)
        x1, y1 = col * cell_size + padding, row * cell_size + padding
        x2, y2 = (col + 1) * cell_size - padding, (row + 1) * cell_size - padding
        color = (0, 255, 0) if status == "Free" else (0, 0, 255)
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), (50, 50, 50), 1)
        text = str(idx + 1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(grid_img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    grid_img_path = RESULTS_DIR / f"{lot_name}_grid.jpg"
    cv2.imwrite(str(grid_img_path), grid_img)

    return JSONResponse({
        "free_count": free_count,
        "occupied_count": total - free_count,
        "total": total,
        "empty_spots": empty_spots,
        "taken_spots": taken_spots,
        "annotated_image": f"/results/{file.filename}",
        "grid_map": f"/results/{lot_name}_grid.jpg",
        "spot_images": spot_images,
        "spot_status": spot_status
    })

@app.get("/analytics/{lot_name}")
async def get_analytics(lot_name: str):
    analytics_data = load_analytics(lot_name)
    spot_analytics = {sid: calculate_spot_analytics(hist)
                      for sid, hist in analytics_data.get("spot_history", {}).items()}
    if spot_analytics:
        avg_occupancy = sum(s["occupancy_rate"] for s in spot_analytics.values()) / len(spot_analytics)
        avg_turnover = sum(s["turnover_count"] for s in spot_analytics.values()) / len(spot_analytics)
        avg_duration = sum(s["avg_occupied_duration"] for s in spot_analytics.values()) / len(spot_analytics)
        avg_utilization = sum(s["utilization_score"] for s in spot_analytics.values()) / len(spot_analytics)
    else:
        avg_occupancy = avg_turnover = avg_duration = avg_utilization = 0

    trends = find_trends(analytics_data)
    return JSONResponse({
        "spot_analytics": spot_analytics,
        "averages": {
            "avg_occupancy_rate": round(avg_occupancy, 1),
            "avg_turnover": round(avg_turnover, 1),
            "avg_occupied_duration": round(avg_duration, 1),
            "avg_utilization_score": round(avg_utilization, 1)
        },
        "trends": trends,
        "total_checks": len(analytics_data.get("checks", []))
    })