import os
import json
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np


app = FastAPI()
app.mount("/results", StaticFiles(directory="results"), name="results")

# creates grids, images, results, crops folders
for folder in ["grids", "images", "results", "crops"]:
    os.makedirs(folder, exist_ok=True)

# templates folder
templates = Jinja2Templates(directory=os.path.join("parking_app", "templates"))

#CNN identifier
class SpotCNN(nn.Module):
    def __init__(self):
        super(SpotCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)



        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 2)  #2 neurons --> free or occupied softmax

#forward pass (add relu after conv3) + flatten + relu

    def forward(self, x): 


        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))

        x = self.fc2(x)
        return x


model = SpotCNN()
model_path = os.path.join(os.path.dirname(__file__), "spot_cnn.pth")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

CLASS_NAMES = ["Free", "Occupied"]

# image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


#prediction helper - takes in images, sends through pth file
def predict_spot(crop):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]


# ROUTES


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...), lot_name: str = Form(...)):
    image_path = os.path.join("images", file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())


    img = cv2.imread(image_path)
    spots = []

    display_w = 1200
    scale = display_w / img.shape[1]
    display_img = cv2.resize(img, (display_w, int(img.shape[0] * scale)))


    while True:
        rect = cv2.selectROI("Select Spot", display_img, showCrosshair=True, fromCenter=False)
        if rect == (0, 0, 0, 0):
            break
        x, y, w, h = rect
        spots.append((int(x/scale), int(y/scale), int((x+w)/scale), int((y+h)/scale)))

    cv2.destroyAllWindows()

    grid_path = os.path.join("grids", f"{lot_name}.json")
    with open(grid_path, "w") as f:
        json.dump(spots, f, indent=2)

    return {"message": f"Saved {len(spots)} spots for lot {lot_name}"}


@app.get("/student", response_class=HTMLResponse)
async def student_page(request: Request):
    return templates.TemplateResponse("student.html", {"request": request})


@app.get("/lots")
async def list_lots():
    lots = [f.replace(".json", "") for f in os.listdir("grids") if f.endswith(".json")]
    return {"lots": lots}


@app.post("/student/check")
async def student_check(file: UploadFile = File(...), lot_name: str = Form(...)):
    image_path = os.path.join("images", file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())

    grid_path = os.path.join("grids", f"{lot_name}.json")
    with open(grid_path) as f:
        spots = json.load(f)

    img = cv2.imread(image_path)
    crop_dir = os.path.join("crops", lot_name)
    os.makedirs(crop_dir, exist_ok=True)

    free_count = 0
    total = len(spots)
    spot_status = []

    for i, (x1, y1, x2, y2) in enumerate(spots):
        crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(crop_dir, f"spot_{i}.jpg")
        cv2.imwrite(crop_path, crop)
        label = predict_spot(crop)
        spot_status.append(label)

        
        color = (0, 255, 0) if label == "Free" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if label == "Free":
            free_count += 1

    empty_spots = round(100 * (free_count / total), 2)
    taken_spots = round(100 * ((total - free_count) / total))

    cv2.putText(img, f"Free: {free_count}/{total} ({empty_spots:.0f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(img, f"Occupied: {total - free_count}/{total} ({taken_spots:.0f}%)",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    annotated_path = os.path.join("results", file.filename)
    cv2.imwrite(annotated_path, img)

    # GRID MAP
    grid_size = int(total ** 0.5) + 1
    cell_size = 60
    padding = 5
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

    grid_path = os.path.join("results", f"{lot_name}_grid.jpg")
    cv2.imwrite(grid_path, grid_img)

    return JSONResponse({
        "free_count": free_count,
        "occupied_count": total - free_count,
        "total": total,
        "empty_spots": empty_spots,
        "taken_spots": taken_spots,
        "annotated_image": f"/results/{file.filename}",
        "grid_map": f"/results/{lot_name}_grid.jpg"
    })
