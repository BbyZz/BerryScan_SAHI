import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import torch
import numpy as np

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    print("❌ SAHI not installed. Please run: pip install sahi")
    SAHI_AVAILABLE = False
    from ultralytics import YOLO  


SAHI_CONFIG = {
    "model_path": "cbb.pt",
    "slice_height": 256,
    "slice_width": 256,
    "overlap_height_ratio": 0.4,
    "overlap_width_ratio": 0.4,
    "model_confidence_threshold": 0.3,
    "certainty_threshold": 0.59,
    "iou_threshold": 0.5,
    "postprocess_class_agnostic": True
}


REF_W, REF_H = 1280, 720


cap = cv2.VideoCapture(0)  # Try 1 if 0 doesn't work


is_paused = False
last_frame = None
camera_label = None
results_label = None
model = None

SCREEN_W = REF_W
SCREEN_H = REF_H
PREVIEW_W = 810
PREVIEW_H = 640



print("--- Loading AI Model... Please wait ---")
try:
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ CUDA not available, using CPU.")

    if SAHI_AVAILABLE:
        model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=SAHI_CONFIG["model_path"],
            confidence_threshold=SAHI_CONFIG["model_confidence_threshold"],
            device=device,
        )
        print("✅ SAHI + YOLO Model loaded successfully!")
    else:
        model = YOLO(SAHI_CONFIG["model_path"])
        model.to(device)
        print("✅ Standard YOLO Model loaded (SAHI missing).")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"Make sure '{SAHI_CONFIG['model_path']}' is in the folder.")



COLOR_INFESTED     = (0, 0, 255)    # Red
COLOR_NON_INFESTED = (0, 255, 0)    # Green
COLOR_UNKNOWN      = (0, 255, 255)  # Yellow


def classify_detection(label_name, score):
    """Map a raw model label + confidence to (display_label, BGR_color).

    - Low-confidence detections fall through to "Unknown" (yellow).
    - Anything matching non/healthy/clean keywords becomes Non-Infested (green).
    - Anything matching infest/borer/cbb keywords becomes Infested (red).
    - Unrecognized class names default to Unknown (yellow).
    The non-infested check runs BEFORE the infested check, otherwise a label
    like "non-infested" would match the substring "infest" and turn red.
    """
    if score < SAHI_CONFIG["certainty_threshold"]:
        return "Unknown", COLOR_UNKNOWN

    lname = label_name.lower().strip()

    if any(k in lname for k in ("non", "healthy", "clean", "good")):
        return label_name, COLOR_NON_INFESTED

    if any(k in lname for k in ("infest", "borer", "cbb", "damaged", "bad")):
        return label_name, COLOR_INFESTED

    return "Unknown", COLOR_UNKNOWN


def detect_objects(input_image):
    """Runs SAHI or fallback YOLO. Returns (annotated_frame, label_counts)."""
    global model

    if model is None:
        print("Error: Model not loaded.")
        return input_image, {}

    annotated_img = input_image.copy()
    rgb_image = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    label_counts = {}

    try:
        if SAHI_AVAILABLE:
            result = get_sliced_prediction(
                rgb_image,
                model,
                slice_height=SAHI_CONFIG["slice_height"],
                slice_width=SAHI_CONFIG["slice_width"],
                overlap_height_ratio=SAHI_CONFIG["overlap_height_ratio"],
                overlap_width_ratio=SAHI_CONFIG["overlap_width_ratio"],
                postprocess_type="NMS",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=SAHI_CONFIG["iou_threshold"],
                postprocess_class_agnostic=SAHI_CONFIG["postprocess_class_agnostic"]
            )

            object_prediction_list = result.object_prediction_list
            print(f"✅ SAHI Complete: Found {len(object_prediction_list)} objects.")

            for prediction in object_prediction_list:
                score = prediction.score.value
                bbox = prediction.bbox
                x_min = int(bbox.minx)
                y_min = int(bbox.miny)
                x_max = int(bbox.maxx)
                y_max = int(bbox.maxy)

                raw_label = prediction.category.name
                display_label, color = classify_detection(raw_label, score)

                label_counts[display_label] = label_counts.get(display_label, 0) + 1

                cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), color, 2)
                label_text = f"{display_label} {score:.2f}"
                text_loc = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
                cv2.putText(annotated_img, label_text, text_loc,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if not object_prediction_list:
                print("No objects detected.")

        else:
            results = model(annotated_img, iou=0.4, conf=0.4, agnostic_nms=True)
            if results and results[0].boxes:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].int().tolist()
                    class_id = int(box.cls[0].item())
                    raw_label = model.names[class_id]
                    score = float(box.conf[0].item()) if hasattr(box, "conf") else 1.0
                    display_label, color = classify_detection(raw_label, score)
                    label_counts[display_label] = label_counts.get(display_label, 0) + 1
                    cv2.rectangle(annotated_img,
                                  (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),
                                  color, 2)
                    cv2.putText(annotated_img, display_label, (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"📊 Tally: {label_counts}")
        return annotated_img, label_counts

    except Exception as e:
        print(f"Error during inference: {e}")
        return input_image, {}


def display_on_label(frame):
    """Resize and push a frame to the main preview label.
    Uses PREVIEW_W / PREVIEW_H globals so it adapts to the current screen size.
    """
    global camera_label, PREVIEW_W, PREVIEW_H
    if frame is None or camera_label is None:
        return
    w = max(1, PREVIEW_W)
    h = max(1, PREVIEW_H)
    frame_resized = cv2.resize(frame, (w, h))
    cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk  
    camera_label.configure(image=imgtk)


def update_results_label(tally: dict):
    """Refresh the tally box on the left panel."""
    global results_label
    if results_label is None:
        return
    if not tally:
        results_label.config(text="No detections found.")
        return
    output = "RESULTS TALLY\n" + "=" * 15 + "\n"
    for label, count in tally.items():
        output += f"{label.upper()}: {count}\n"
    output += "=" * 15 + f"\nTOTAL: {sum(tally.values())}"
    results_label.config(text=output)



def gallery_button_clicked():
    save_folder = "processed_images"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    images = [
        f for f in os.listdir(save_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    base = min(SCREEN_W / REF_W, SCREEN_H / REF_H)
    f_big   = max(10, int(24 * base))
    f_med   = max(9,  int(14 * base))
    f_small = max(8,  int(11 * base))

    if not images:
        popup = tk.Toplevel()
        popup.title("Gallery")
        popup.attributes("-fullscreen", True)
        popup.configure(bg="#FFD782")
        tk.Label(popup, text="No saved images yet.", bg="#FFD782",
                 font=("Arial", f_big, "bold")).pack(expand=True)
        tk.Button(popup, text="CLOSE", bg="#C91B1A", fg="white",
                  font=("Arial", f_med, "bold"),
                  command=popup.destroy).pack(pady=int(SCREEN_H * 0.04))
        popup.bind("<Escape>", lambda e: popup.destroy())
        return

    images.sort(reverse=True)

    gal_win = tk.Toplevel()
    gal_win.attributes("-fullscreen", True)
    gal_win.configure(bg="#FFD782")
    gal_win.bind("<Escape>", lambda e: gal_win.destroy())

    # --- Top bar ---
    top_bar = tk.Frame(gal_win, bg="#FFD782")
    top_bar.pack(side="top", fill="x",
                 padx=int(SCREEN_W * 0.015), pady=int(SCREEN_H * 0.02))

    tk.Button(top_bar, text="← CLOSE", command=gal_win.destroy,
              font=("Arial", f_med, "bold"), bg="white",
              fg="#C91B1A", borderwidth=0, cursor="hand2").pack(side="left")

    counter_var = tk.StringVar()
    tk.Label(top_bar, textvariable=counter_var, bg="#FFD782",
             font=("Arial", f_med, "bold")).pack(side="right")

    viewer = tk.Label(gal_win, bg="#FFD782")
    viewer.pack(expand=True)

    name_var = tk.StringVar()
    tk.Label(gal_win, textvariable=name_var, bg="#FFD782",
             font=("Arial", f_small)).pack(pady=int(SCREEN_H * 0.006))

    nav_frame = tk.Frame(gal_win, bg="#FFD782")
    nav_frame.pack(side="bottom", pady=int(SCREEN_H * 0.03))

    current_idx = [0]

    thumb_w = int(SCREEN_W * 0.85)
    thumb_h = int(SCREEN_H * 0.75)

    def show_image():
        filename = images[current_idx[0]]
        filepath = os.path.join(save_folder, filename)
        try:
            pil_img = Image.open(filepath)
            pil_img.thumbnail((thumb_w, thumb_h))
            tk_img = ImageTk.PhotoImage(pil_img)
            viewer.config(image=tk_img)
            viewer.image = tk_img
            name_var.set(filename)
            counter_var.set(f"{current_idx[0] + 1} / {len(images)}")
        except Exception as e:
            print(f"Gallery load error: {e}")

    def prev_image():
        current_idx[0] = (current_idx[0] - 1 + len(images)) % len(images)
        show_image()

    def next_image():
        current_idx[0] = (current_idx[0] + 1) % len(images)
        show_image()

    gal_win.bind("<Left>", lambda e: prev_image())
    gal_win.bind("<Right>", lambda e: next_image())

    btn_pad = int(SCREEN_W * 0.025)
    tk.Button(nav_frame, text="<< PREV", command=prev_image,
              font=("Arial", f_med, "bold"), bg="white",
              borderwidth=0).pack(side="left", padx=btn_pad)
    tk.Button(nav_frame, text="NEXT >>", command=next_image,
              font=("Arial", f_med, "bold"), bg="white",
              borderwidth=0).pack(side="left", padx=btn_pad)

    show_image()


def toggle_capture():
    global is_paused
    is_paused = not is_paused
    if not is_paused:
        update_camera_feed()
        if results_label:
            results_label.config(text="Ready")
    print("Feed Paused" if is_paused else "Feed Resumed")


def upload_image_clicked():
    global is_paused, last_frame
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
    if file_path:
        is_paused = True
        loaded_frame = cv2.imread(file_path)
        if loaded_frame is not None:
            last_frame = loaded_frame
            display_on_label(loaded_frame)
            if results_label:
                results_label.config(text="Image loaded.\nPress RUN MODEL.")
            print(f"Input: Loaded image from {file_path}")


def run_model_clicked():
    global last_frame
    if not is_paused or last_frame is None:
        if results_label:
            results_label.config(text="⚠️ Capture or upload\nan image first.")
        print("Please Capture/Upload first.")
        return

    if results_label:
        results_label.config(text="Running model…\nPlease wait.")
    print("Run Model: Processing...")

    annotated_frame, tally = detect_objects(last_frame)

    display_on_label(annotated_frame)
    update_results_label(tally)

    save_folder = "processed_images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, f"berry_scan_{timestamp}.jpg")
    cv2.imwrite(filename, annotated_frame)
    print(f"✅ Saved: {filename}")


def update_camera_feed():
    global is_paused, last_frame
    if is_paused:
        return
    ret, frame = cap.read()
    if ret:
        last_frame = frame
        display_on_label(frame)
    if camera_label:
        camera_label.after(10, update_camera_feed)



def create_berryscan_interface():
    global camera_label, results_label
    global SCREEN_W, SCREEN_H, PREVIEW_W, PREVIEW_H

    root = tk.Tk()
    root.title("BerryScan")
    root.attributes("-fullscreen", True)
    root.configure(bg="#FFD782")

    # --- Get the ACTUAL screen dimensions and build scale helpers ---
    root.update_idletasks()
    SCREEN_W = root.winfo_screenwidth()
    SCREEN_H = root.winfo_screenheight()

    # Independent x/y scale so the layout fills the whole display
    sx_ratio = SCREEN_W / REF_W
    sy_ratio = SCREEN_H / REF_H
    # Uniform scale for fonts so text stays proportional and never overflows
    f_ratio = min(sx_ratio, sy_ratio)

    def SX(v): return int(round(v * sx_ratio))
    def SY(v): return int(round(v * sy_ratio))
    def F(pt): return max(8, int(round(pt * f_ratio)))

    # --- Exit binding ---
    root.bind("<Escape>", lambda e: [cap.release(), root.destroy()])
    root.protocol("WM_DELETE_WINDOW", lambda: [cap.release(), root.destroy()])

    title_font = tkfont.Font(family="Arial", size=F(46), weight="bold")
    btn_font   = tkfont.Font(family="Arial", size=F(12), weight="bold")

    # 1. Gallery / Logo Button (top-left)
    logo_w, logo_h = SX(96), SY(96)
    image_path = r"E:\Personal Files\Feb2FinalThesis\Button.png"
    try:
        # Use PIL so the icon scales to the button size on every screen
        pil_icon = Image.open(image_path).resize((logo_w, logo_h), Image.LANCZOS)
        icon_image = ImageTk.PhotoImage(pil_icon)
        logo_btn = tk.Button(root, image=icon_image, bg="#FFD782",
                             command=gallery_button_clicked,
                             borderwidth=0, cursor="hand2")
        logo_btn.image = icon_image
    except Exception:
        logo_btn = tk.Button(root, text="GALLERY", bg="#262626", fg="white",
                             font=btn_font, command=gallery_button_clicked,
                             borderwidth=0, cursor="hand2")
    logo_btn.place(x=SX(40), y=SY(40), width=logo_w, height=logo_h)

    # 2. Branding
    tk.Label(root, text="Berry", bg="#FFD782", fg="#E82C2A",
             font=title_font).place(x=SX(40), y=SY(140))
    tk.Label(root, text="Scan", bg="#FFD782", fg="black",
             font=title_font).place(x=SX(225), y=SY(140))

    # 3. Control Buttons
    btn_x, btn_w, btn_h = SX(40), SX(340), SY(60)

    tk.Button(root, text="CAPTURE PHOTO", bg="#262626", fg="white",
              font=btn_font, command=toggle_capture,
              borderwidth=0).place(x=btn_x, y=SY(250), width=btn_w, height=btn_h)

    tk.Button(root, text="RUN MODEL (SAHI)", bg="#C91B1A", fg="white",
              font=btn_font, command=run_model_clicked,
              borderwidth=0).place(x=btn_x, y=SY(325), width=btn_w, height=btn_h)

    tk.Button(root, text="UPLOAD IMAGE", bg="#005A9C", fg="white",
              font=btn_font, command=upload_image_clicked,
              borderwidth=0).place(x=btn_x, y=SY(400), width=btn_w, height=btn_h)

    # 4. Preview Screen — sized off the actual screen so the camera fills it
    #    Height is reduced slightly to leave room for the color legend strip.
    PREVIEW_W = SX(810)
    PREVIEW_H = SY(595)
    camera_label = tk.Label(root, bg="black")
    camera_label.place(x=SX(430), y=SY(40), width=PREVIEW_W, height=PREVIEW_H)

    # 4b. Color Legend (map-style legend strip below the preview)
    legend_h = SY(40)
    legend_y = SY(40) + PREVIEW_H + SY(5)
    legend_frame = tk.Frame(root, bg="white",
                            highlightbackground="black", highlightthickness=2)
    legend_frame.place(x=SX(430), y=legend_y,
                       width=PREVIEW_W, height=legend_h)

    legend_font = tkfont.Font(family="Arial", size=F(11), weight="bold")
    swatch_size = SY(18)

    def _add_legend_item(color_hex, text):
        item = tk.Frame(legend_frame, bg="white")
        swatch = tk.Frame(item, bg=color_hex,
                          highlightbackground="black", highlightthickness=1)
        swatch.configure(width=swatch_size, height=swatch_size)
        swatch.pack_propagate(False)
        swatch.pack(side="left", padx=(SX(14), SX(8)), pady=SY(8))
        tk.Label(item, text=text, bg="white", fg="black",
                 font=legend_font).pack(side="left", padx=(0, SX(20)))
        item.pack(side="left", fill="y")

    # Hex values match the BGR colors drawn on the bounding boxes.
    _add_legend_item("#FF0000", "INFESTED")       # Red
    _add_legend_item("#00FF00", "NON-INFESTED")   # Green
    _add_legend_item("#FFFF00", "UNKNOWN")        # Yellow

    # 5. Tally / Status Box — bottom edge leaves a ~40px margin from screen bottom
    tally_frame = tk.Frame(root, bg="white",
                           highlightbackground="black", highlightthickness=2)
    tally_frame.place(x=SX(40), y=SY(475),
                      width=SX(340), height=SY(205))

    results_label = tk.Label(
        tally_frame,
        text="Ready",
        bg="white",
        font=("Courier", F(15), "bold"),
        justify="left",
        anchor="nw",
        padx=SX(15),
        pady=SY(12)
    )
    results_label.pack(fill="both", expand=True)

    # 6. Start live feed
    update_camera_feed()

    root.mainloop()


if __name__ == "__main__":
    create_berryscan_interface()
