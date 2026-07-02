"""
model/cnn.py

Multi-threaded Convolutional Neural Network (CNN) implementation for MNIST classification.

General logic:
    - Each network layer (Conv2D, MaxPooling2D, Flatten, Dense, ...) runs in a separate
      LayerThread to capture the real OS-level native id for each layer and display it
      in the Timeline.
    - Each training Epoch runs in a separate EpochThread. The number of concurrently
      running EpochThreads is configurable by the user via the UI.
    - To prevent race conditions, the minimum possible number of semaphores is used:
          data_semaphore    -> Protects access to x_train and y_train
                                (batch sampling from training data)
          compute_semaphore -> Protects final calculations: forward pass,
                                backward pass (GradientTape), and applying
                                gradients to model weights
                                (optimizer.apply_gradients)

      Note: In addition to these two semaphores, a counting Semaphore named
      concurrency_limiter is used to limit the number of concurrent EpochThreads.
      This semaphore is for "preventing race conditions on shared data" but rather
      a throttling/scheduling tool to control concurrency. Therefore, it is not
      counted in the "minimum number of semaphores required to protect shared data".
      Also, threading.Lock (not Semaphore) is used for safely writing to log lists
      and the status dictionary, as these are purely for bookkeeping, not for
      protecting the main training data.

Important technical note about LayerThread:
    The tf.GradientTape object in TensorFlow maintains its state in a thread-local
    manner. If the actual forward pass operations that need gradient calculation
    are executed in a thread other than the one where the tape was opened, those
    operations will not be recorded by the tape, making gradient calculation impossible.
    Therefore, LayerThreads are used to *demonstrate real concurrent layer execution
    on the OS* (a demonstrative/pass on a sample batch, layer by layer, each in its
    own thread, logging native_id and real start/end times). The actual computations
    that update the weights (forward+backward+apply_gradients) are performed within
    the EpochThread itself, protected by compute_semaphore, to ensure training
    correctness while proving real thread concurrency.

Author: Mohammad Harighi
"""

import os
import time
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# ----------------------------------------------------------------------
# Global semaphores (as required: minimum possible number)
# ----------------------------------------------------------------------
data_semaphore = threading.Semaphore(1)     # Protects x_train / y_train
compute_semaphore = threading.Semaphore(1)  # Protects forward/backward/apply_gradients

# Auxiliary locks purely for bookkeeping (logs and status) - not for main data
_log_lock = threading.Lock()
_status_lock = threading.Lock()


class CNNTrainer:
    """
    Main class managing the CNN model, data, threads, and training status.
    A singleton instance of this class is created in app.py.
    """

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # Overall status sent to the frontend via /status
        self.status = {
            "state": "idle",          # idle | loading_data | training | evaluating | completed | error
            "message": "",
            "epochs_total": 0,
            "epochs_done": 0,
            "epoch_logs": [],         # [{epoch, loss, accuracy}]
            "thread_log": [],         # [{type, name, native_id, start, end}]
            "test_accuracy": None,
            "test_loss": None,
            "error": None,
        }

        self._train_thread = None  # The thread that runs the whole train() process in the background

    # ------------------------------------------------------------------
    # Build CNN model
    # ------------------------------------------------------------------
    def build_model(self):
        """
        Build the CNN architecture using Keras:
        Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D -> Flatten -> Dense -> Dense
        """
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation="relu",
                   input_shape=(28, 28, 1), name="conv2d_1"),
            MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
            Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
            MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense_1"),
            Dense(10, activation="softmax", name="dense_output"),
        ])
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam()
        return model

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    def load_data(self):
        """
        Load the MNIST dataset from the standard Keras path:
        ~/.keras/datasets/mnist.npz
        """
        self.status["state"] = "loading_data"
        self.status["message"] = "Loading MNIST dataset..."

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize and add channel dimension (28,28) -> (28,28,1)
        x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
        x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

        self.x_train = x_train
        self.y_train = y_train.astype("int64")
        self.x_test = x_test
        self.y_test = y_test.astype("int64")

    # ------------------------------------------------------------------
    # Thread logging utility (for Timeline / Gantt Chart)
    # ------------------------------------------------------------------
    def _log_thread_event(self, thread_type, name, native_id, start_ts, end_ts):
        entry = {
            "type": thread_type,          # "layer" or "epoch"
            "name": name,
            "native_id": native_id,       # threading.get_native_id() -> real OS-level thread ID
            "start": start_ts,
            "end": end_ts,
        }
        with _log_lock:
            self.status["thread_log"].append(entry)

    # ------------------------------------------------------------------
    # LayerThread: Execute each layer in a separate thread (to prove real concurrency)
    # ------------------------------------------------------------------
    def _run_layer_in_thread(self, layer, input_tensor, result_holder, epoch_idx):
        """
        Function executed inside each LayerThread: computes the output of a layer
        from the previous layer's output and logs its native_id and real start/end times.
        """
        start_ts = time.time()
        native_id = threading.get_native_id()

        output_tensor = layer(input_tensor, training=False)

        end_ts = time.time()
        result_holder["output"] = output_tensor

        self._log_thread_event(
            thread_type="layer",
            name=f"epoch{epoch_idx}-{layer.name}",
            native_id=native_id,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def _demo_layer_pass(self, sample_batch, epoch_idx):
        """
        A demonstrative pass of a sample batch through all layers of the model,
        such that each layer runs in its own LayerThread.
        Since the output of each layer depends on the previous one, each LayerThread
        is started after the previous LayerThread finishes (start then join), but each
        still has a separate real native_id at the OS level, visible in the Timeline.
        These can overlap with LayerThreads from other concurrent EpochThreads.
        """
        current_tensor = sample_batch
        for layer in self.model.layers:
            result_holder = {}
            t = threading.Thread(
                target=self._run_layer_in_thread,
                args=(layer, current_tensor, result_holder, epoch_idx),
                name=f"LayerThread-e{epoch_idx}-{layer.name}",
            )
            t.start()
            t.join()  # Due to data dependency between layers
            current_tensor = result_holder["output"]

    # ------------------------------------------------------------------
    # A single training step on a batch (forward + backward + apply_gradients)
    # ------------------------------------------------------------------
    def _train_step(self, x_batch, y_batch):
        """
        This function performs all final calculations (forward, backward, weight updates)
        under the protection of compute_semaphore to prevent race conditions on shared model weights.
        """
        with compute_semaphore:
            with tf.GradientTape() as tape:
                logits = self.model(x_batch, training=True)
                loss_value = self.loss_fn(y_batch, logits)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            preds = tf.argmax(logits, axis=1)
            correct = tf.reduce_sum(tf.cast(tf.equal(preds, y_batch), tf.float32))
            accuracy = correct / tf.cast(tf.shape(y_batch)[0], tf.float32)

        return float(loss_value.numpy()), float(accuracy.numpy())

    # ------------------------------------------------------------------
    # EpochThread: Full logic for a single epoch
    # ------------------------------------------------------------------
    def _epoch_worker(self, epoch_idx, batch_size, concurrency_limiter):
        """
        Function executed by each EpochThread:
            1) Enters the concurrent execution section protected by concurrency_limiter.
            2) A demonstrative layer-by-layer pass (LayerThreads) on a sample batch.
            3) Iterates through the entire training data batch by batch:
                 - Access/sample x_train, y_train under data_semaphore
                 - Compute and update weights under compute_semaphore (in _train_step)
            4) Logs the average loss/accuracy of this epoch in the status.
        """
        with concurrency_limiter:
            epoch_start_ts = time.time()
            native_id = threading.get_native_id()

            num_samples = self.x_train.shape[0]
            num_batches = int(np.ceil(num_samples / batch_size))

            # ---- Step 1: Sample batches under data_semaphore protection ----
            with data_semaphore:
                indices = np.random.permutation(num_samples)

            # ---- Step 2: Demonstrative layer-by-layer pass to prove multi-threading ----
            demo_size = min(batch_size, 32)
            with data_semaphore:
                demo_x = self.x_train[indices[:demo_size]]
            self._demo_layer_pass(tf.convert_to_tensor(demo_x), epoch_idx)

            # ---- Step 3: Actual training on all batches of this epoch ----
            losses = []
            accuracies = []
            for b in range(num_batches):
                start_i = b * batch_size
                end_i = min(start_i + batch_size, num_samples)

                with data_semaphore:
                    batch_idx = indices[start_i:end_i]
                    x_batch = self.x_train[batch_idx]
                    y_batch = self.y_train[batch_idx]

                loss_value, acc_value = self._train_step(
                    tf.convert_to_tensor(x_batch), tf.convert_to_tensor(y_batch)
                )
                losses.append(loss_value)
                accuracies.append(acc_value)

            epoch_end_ts = time.time()

            self._log_thread_event(
                thread_type="epoch",
                name=f"EpochThread-{epoch_idx}",
                native_id=native_id,
                start_ts=epoch_start_ts,
                end_ts=epoch_end_ts,
            )

            mean_loss = float(np.mean(losses))
            mean_acc = float(np.mean(accuracies))

            with _status_lock:
                self.status["epoch_logs"].append({
                    "epoch": epoch_idx,
                    "loss": round(mean_loss, 4),
                    "accuracy": round(mean_acc, 4),
                })
                self.status["epoch_logs"].sort(key=lambda e: e["epoch"])
                self.status["epochs_done"] += 1

    # ------------------------------------------------------------------
    # Main training function (called in a background thread from app.py)
    # ------------------------------------------------------------------
    def train(self, epochs, concurrent_epochs, batch_size):
        try:
            with _status_lock:
                self.status = {
                    "state": "loading_data",
                    "message": "Preparing...",
                    "epochs_total": epochs,
                    "epochs_done": 0,
                    "epoch_logs": [],
                    "thread_log": [],
                    "test_accuracy": None,
                    "test_loss": None,
                    "error": None,
                }

            self.load_data()
            self.build_model()

            with _status_lock:
                self.status["state"] = "training"
                self.status["message"] = "Training the model..."

            # Counting semaphore to limit the number of concurrent EpochThreads
            concurrency_limiter = threading.Semaphore(max(1, int(concurrent_epochs)))

            epoch_threads = []
            for epoch_idx in range(1, epochs + 1):
                t = threading.Thread(
                    target=self._epoch_worker,
                    args=(epoch_idx, batch_size, concurrency_limiter),
                    name=f"EpochThread-{epoch_idx}",
                )
                epoch_threads.append(t)
                t.start()

            for t in epoch_threads:
                t.join()

            # Final evaluation on test data
            with _status_lock:
                self.status["state"] = "evaluating"
                self.status["message"] = "Evaluating on test data..."

            test_loss, test_acc = self.model.evaluate(
                self.x_test, self.y_test, verbose=0
            )

            with _status_lock:
                self.status["test_loss"] = round(float(test_loss), 4)
                self.status["test_accuracy"] = round(float(test_acc), 4)
                self.status["state"] = "completed"
                self.status["message"] = "Training completed successfully."

        except Exception as exc:  # noqa: BLE001
            with _status_lock:
                self.status["state"] = "error"
                self.status["error"] = str(exc)
                self.status["message"] = "Training completed successfully."

    def start_training_async(self, epochs, concurrent_epochs, batch_size):
        """
        Start training in a separate background thread so that the POST /train request
        returns immediately and the UI can track the progress via polling on /status.
        """
        if self._train_thread is not None and self._train_thread.is_alive():
            return False  # Another training process is already running

        self._train_thread = threading.Thread(
            target=self.train,
            args=(epochs, concurrent_epochs, batch_size),
            name="TrainingSupervisorThread",
            daemon=True,
        )
        self._train_thread.start()
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, image_array):
        """
        image_array: numpy array of shape (28, 28) with values 0..1
        Returns: (digit, confidence)
        """
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")

        x = image_array.astype("float32").reshape(1, 28, 28, 1)
        probs = self.model.predict(x, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        return digit, confidence, probs.tolist()

    def get_status(self):
        with _status_lock:
            # Shallow copy to prevent concurrent modification during JSON serialization
            return dict(self.status)