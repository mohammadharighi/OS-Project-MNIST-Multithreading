# Multithreaded CNN Training on MNIST (Operating Systems Final Project)

This repository contains a complete, self-contained implementation of a Convolutional Neural Network (CNN) designed for the MNIST dataset. The project demonstrates core Operating Systems concepts, utilizing **native multithreading** and synchronization primitives like **Semaphores** and **Locks** to safely manage training, model states, and asynchronous requests.

Developed by **Mohammad Harighi** as the Final Project for the Operating Systems course.

---

## Key Features & OS Integration

- **Asynchronous Training Thread:** The training loop runs on a dedicated background thread, preventing blocking of the main Flask application thread and ensuring UI responsiveness.
- **Thread Synchronization (Semaphores & Locks):**
  - **Mutex Lock:** Protects shared model weights during updates, prediction calls, and asynchronous model-state fetches.
  - **Thread-safe Communication:** Avoids race conditions between backend status updates and the frontend event-stream.
- **Interactive Web Interface:** A dark-themed, clean web UI built with HTML/CSS/JS that displays live training progress, loss, accuracy, and lets users draw digits for real-time inference.
- **Pure Python & Flask Architecture:** Lightweight Flask server hosting API endpoints for starting/stopping training, checking status, and running predictions.

---

## Project Structure

- **`cnn.py`:** Core neural network logic containing convolutional layers, pooling, backpropagation, and weights optimization.
- **`app.py`:** Flask web application, handling route dispatching, training thread initialization, and Mutex/Semaphore synchronization.
- **`index.html`:** The frontend interface providing an interactive drawing canvas and real-time training dashboard.

---

## Installation & Setup

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install Flask numpy tensorflow
```

> Tip: If you prefer a reproducible setup, create a `requirements.txt` file and install via:
> ```bash
> pip install -r requirements.txt
> ```

### 4. Run the Server
```bash
python app.py
```

After starting the server, open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## How to Use the UI

1. **Configure Training Parameters:** Set the learning rate, number of training epochs, and batch size from the configuration panel.
2. **Start Training:** Click the **Start Training** button. The frontend will dynamically stream loss and accuracy metrics via Server-Sent Events (SSE).
3. **Canvas Drawing:** Draw any digit from `0-9` on the integrated HTML5 canvas. The background thread will safely read the weights using locks and display the predicted digit instantly.

---

## Developer

**Mohammad Harighi**  
Operating Systems Final Project
```