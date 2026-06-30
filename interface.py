import json
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import torch
import tlm_class


DATASET_FOLDER = Path("datasets_tlm_matematicas")

MODEL_PATH = Path("tlm_model.pth")
VOCAB_PATH = Path("vocab.json")
CONFIG_PATH = Path("training_config.json")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<eos>"]


# ============================================================
# TEXT PROCESSING
# ============================================================

def normalize_text(text):
    text = text.lower()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )

    text = re.sub(r"[^a-z0-9<>]+", " ", text)

    return text.strip()


def tokenize(text):
    return normalize_text(text).split()


def list_dataset_files():
    if not DATASET_FOLDER.exists():
        raise FileNotFoundError(
            f"No encuentro la carpeta {DATASET_FOLDER}.\n"
            f"Pon datasets_tlm_matematicas/ en la misma carpeta que interface.py."
        )

    files = sorted(DATASET_FOLDER.glob("*.txt"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No hay archivos .txt dentro de {DATASET_FOLDER}."
        )

    return files


def read_dataset_files(selected_dataset_name):
    files = list_dataset_files()

    if selected_dataset_name == "TODOS_LOS_DATASETS":
        selected_files = [
            file
            for file in files
            if file.name != "matematicas_dataset_completo.txt"
        ]

        if len(selected_files) == 0:
            selected_files = files

    else:
        selected_files = [
            file
            for file in files
            if file.name == selected_dataset_name
        ]

        if len(selected_files) == 0:
            raise FileNotFoundError(
                f"No encuentro el dataset seleccionado: {selected_dataset_name}"
            )

    lines = []

    for file in selected_files:
        file_lines = file.read_text(encoding="utf-8").splitlines()
        file_lines = [line.strip() for line in file_lines if line.strip()]
        lines.extend(file_lines)

    return lines, selected_files


def build_vocabulary(lines, max_vocab_size):
    counter = Counter()

    for line in lines:
        tokens = tokenize(line)

        counter.update(tokens)
        counter.update(["<eos>"])

    available_size = max_vocab_size - len(SPECIAL_TOKENS)

    most_common_words = [
        word
        for word, _ in counter.most_common(available_size)
        if word not in SPECIAL_TOKENS
    ]

    vocabulary = SPECIAL_TOKENS + most_common_words

    word_to_index = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    index_to_word = {
        index: word
        for word, index in word_to_index.items()
    }

    return word_to_index, index_to_word


def save_vocabulary(word_to_index, index_to_word):
    with open(VOCAB_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "word_to_index": word_to_index,
                "index_to_word": index_to_word
            },
            file,
            ensure_ascii=False,
            indent=4
        )


def load_vocabulary():
    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    word_to_index = data["word_to_index"]

    index_to_word = {
        int(index): word
        for index, word in data["index_to_word"].items()
    }

    return word_to_index, index_to_word


def save_training_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def load_training_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def text_to_indices(tokens, word_to_index):
    unk_index = word_to_index["<unk>"]

    return [
        word_to_index.get(token, unk_index)
        for token in tokens
    ]


def create_training_examples(lines, word_to_index, context_size):
    """
    This function creates training examples from the dataset lines.
    Each example consists of a context (a list of word indices) and a target (the
    next word index).
    It pads the context with <pad> tokens if necessary and adds an <eos> token at the end of each line.
    """

    inputs = []
    targets = []

    pad_index = word_to_index["<pad>"]
    eos_index = word_to_index["<eos>"]

    for line in lines:
        tokens = tokenize(line)
        indices = text_to_indices(tokens, word_to_index)

        sequence = [pad_index] * context_size + indices + [eos_index]

        for i in range(len(sequence) - context_size):
            context = sequence[i:i + context_size]
            target = sequence[i + context_size]

            inputs.append(context)
            targets.append(target)

    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    return x, y


def prepare_context(text, word_to_index, context_size):
    """
    This function prepares the context for prediction. 
    It takes the input text, tokenizes it, converts it to indices, and pads it if necessary.
    """
    tokens = tokenize(text)

    unk_index = word_to_index.get("<unk>", 0)
    pad_index = word_to_index.get("<pad>", 0)

    indices = [
        word_to_index.get(token, unk_index)
        for token in tokens
    ]

    context = ([pad_index] * context_size + indices)[-context_size:]

    x = torch.tensor([context], dtype=torch.long)

    return x


# ============================================================
# GRAPHICAL INTERFACE
# ============================================================

class TLMInterface(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("TLM · Tiny Language Model")
        self.geometry("960x780")
        self.minsize(900, 720)
        self.configure(bg="#111827")

        self.model = None
        self.word_to_index = None
        self.index_to_word = None

        self.current_context_size = None
        self.current_embedding_dim = None
        self.current_vocab_size = None

        self.training_thread = None
        self.is_training = False

        self.create_styles()
        self.create_widgets()
        self.load_dataset_options()

    # --------------------------------------------------------
    # STYLES
    # --------------------------------------------------------

    def create_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Main.TFrame", background="#111827")
        style.configure("Card.TFrame", background="#1f2937")

        style.configure(
            "Title.TLabel",
            background="#111827",
            foreground="#f9fafb",
            font=("Arial", 28, "bold")
        )

        style.configure(
            "Subtitle.TLabel",
            background="#111827",
            foreground="#cbd5e1",
            font=("Arial", 12)
        )

        style.configure(
            "CardTitle.TLabel",
            background="#1f2937",
            foreground="#f9fafb",
            font=("Arial", 15, "bold")
        )

        style.configure(
            "Normal.TLabel",
            background="#1f2937",
            foreground="#e5e7eb",
            font=("Arial", 12)
        )

        style.configure(
            "Muted.TLabel",
            background="#1f2937",
            foreground="#94a3b8",
            font=("Arial", 11)
        )

        style.configure(
            "Result.TLabel",
            background="#1f2937",
            foreground="#ffffff",
            font=("Arial", 26, "bold")
        )

        style.configure(
            "Accent.TButton",
            font=("Arial", 12, "bold"),
            padding=9
        )

        style.configure(
            "Secondary.TButton",
            font=("Arial", 12),
            padding=9
        )

        style.configure(
            "Horizontal.TProgressbar",
            thickness=13
        )

    # --------------------------------------------------------
    # WIDGETS
    # --------------------------------------------------------

    def create_widgets(self):
        main_frame = ttk.Frame(self, style="Main.TFrame", padding=24)
        main_frame.pack(fill="both", expand=True)

        title = ttk.Label(
            main_frame,
            text="Tiny Language Model",
            style="Title.TLabel"
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            main_frame,
            text="Entrena tu TLM y prueba la predicción de la siguiente palabra.",
            style="Subtitle.TLabel"
        )
        subtitle.pack(anchor="w", pady=(4, 18))

        # -----------------------------
        # Training card
        # -----------------------------

        training_card = ttk.Frame(main_frame, style="Card.TFrame", padding=18)
        training_card.pack(fill="x", pady=(0, 16))

        ttk.Label(
            training_card,
            text="Entrenamiento",
            style="CardTitle.TLabel"
        ).pack(anchor="w")

        dataset_frame = ttk.Frame(training_card, style="Card.TFrame")
        dataset_frame.pack(fill="x", pady=(14, 8))

        ttk.Label(
            dataset_frame,
            text="Dataset",
            style="Muted.TLabel"
        ).pack(anchor="w")

        self.dataset_var = tk.StringVar(value="TODOS_LOS_DATASETS")

        self.dataset_combo = ttk.Combobox(
            dataset_frame,
            textvariable=self.dataset_var,
            state="readonly",
            width=54,
            font=("Arial", 12)
        )
        self.dataset_combo.pack(anchor="w", pady=(4, 0))

        controls_frame = ttk.Frame(training_card, style="Card.TFrame")
        controls_frame.pack(fill="x", pady=(10, 8))

        self.epochs_var = tk.StringVar(value="60")
        self.batch_size_var = tk.StringVar(value="32")
        self.context_size_var = tk.StringVar(value="5")
        self.embedding_dim_var = tk.StringVar(value="32")
        self.max_vocab_size_var = tk.StringVar(value="500")
        self.learning_rate_var = tk.StringVar(value="0.001")

        self.create_labeled_entry(controls_frame, "epochs", self.epochs_var, 8)
        self.create_labeled_entry(controls_frame, "batch_size", self.batch_size_var, 8)
        self.create_labeled_entry(controls_frame, "context_size", self.context_size_var, 8)
        self.create_labeled_entry(controls_frame, "embedding_dim", self.embedding_dim_var, 8)
        self.create_labeled_entry(controls_frame, "max_vocab", self.max_vocab_size_var, 8)
        self.create_labeled_entry(controls_frame, "learning_rate", self.learning_rate_var, 8)

        button_frame = ttk.Frame(training_card, style="Card.TFrame")
        button_frame.pack(fill="x", pady=(10, 8))

        self.train_button = ttk.Button(
            button_frame,
            text="Entrenar modelo",
            style="Accent.TButton",
            command=self.start_training
        )
        self.train_button.pack(side="left")

        self.load_button = ttk.Button(
            button_frame,
            text="Cargar modelo guardado",
            style="Secondary.TButton",
            command=self.load_saved_model
        )
        self.load_button.pack(side="left", padx=(10, 0))

        self.training_progress = ttk.Progressbar(
            training_card,
            maximum=100,
            value=0,
            style="Horizontal.TProgressbar"
        )
        self.training_progress.pack(fill="x", pady=(8, 6))

        self.training_status = ttk.Label(
            training_card,
            text="Modelo todavía no entrenado en esta sesión.",
            style="Muted.TLabel"
        )
        self.training_status.pack(anchor="w", pady=(2, 0))

        self.metrics_label = ttk.Label(
            training_card,
            text="Coste: — · Accuracy: —",
            style="Normal.TLabel"
        )
        self.metrics_label.pack(anchor="w", pady=(8, 0))

        # -----------------------------
        # Input card
        # -----------------------------

        input_card = ttk.Frame(main_frame, style="Card.TFrame", padding=18)
        input_card.pack(fill="x", pady=(0, 16))

        ttk.Label(
            input_card,
            text="Texto de entrada",
            style="CardTitle.TLabel"
        ).pack(anchor="w")

        self.text_input = tk.Text(
            input_card,
            height=4,
            wrap="word",
            bg="#0f172a",
            fg="#f9fafb",
            insertbackground="#f9fafb",
            relief="flat",
            font=("Arial", 14),
            padx=14,
            pady=12
        )
        self.text_input.pack(fill="x", pady=(10, 12))

        self.text_input.insert("1.0", "la derivada mide el cambio")

        self.predict_button = ttk.Button(
            input_card,
            text="Predecir siguiente palabra",
            style="Accent.TButton",
            command=self.predict,
            state="disabled"
        )
        self.predict_button.pack(anchor="e")

        # -----------------------------
        # Result card
        # -----------------------------

        result_card = ttk.Frame(main_frame, style="Card.TFrame", padding=18)
        result_card.pack(fill="both", expand=True)

        ttk.Label(
            result_card,
            text="Predicción",
            style="CardTitle.TLabel"
        ).pack(anchor="w")

        self.prediction_label = ttk.Label(
            result_card,
            text="—",
            style="Result.TLabel"
        )
        self.prediction_label.pack(anchor="w", pady=(14, 4))

        self.percentage_label = ttk.Label(
            result_card,
            text="Porcentaje: —",
            style="Muted.TLabel"
        )
        self.percentage_label.pack(anchor="w", pady=(0, 16))

        ttk.Label(
            result_card,
            text="Top 5 palabras más probables",
            style="Normal.TLabel"
        ).pack(anchor="w")

        self.top_frame = ttk.Frame(result_card, style="Card.TFrame")
        self.top_frame.pack(fill="both", expand=True, pady=(10, 0))

    def create_labeled_entry(self, parent, label_text, variable, width):
        block = ttk.Frame(parent, style="Card.TFrame")
        block.pack(side="left", padx=(0, 12))

        label = ttk.Label(
            block,
            text=label_text,
            style="Muted.TLabel"
        )
        label.pack(anchor="w")

        entry = tk.Entry(
            block,
            textvariable=variable,
            width=width,
            bg="#0f172a",
            fg="#f9fafb",
            insertbackground="#f9fafb",
            relief="flat",
            font=("Arial", 12)
        )
        entry.pack(anchor="w", pady=(4, 0))

    def load_dataset_options(self):
        try:
            files = list_dataset_files()

            options = ["TODOS_LOS_DATASETS"] + [file.name for file in files]

            self.dataset_combo.configure(values=options)
            self.dataset_var.set("TODOS_LOS_DATASETS")

            self.training_status.configure(
                text=f"Datasets detectados: {len(files)} archivos .txt"
            )

        except Exception as error:
            self.training_status.configure(
                text=f"Error detectando datasets: {error}"
            )
            self.train_button.configure(state="disabled")

    # --------------------------------------------------------
    # THREAD-SAFE UI HELPERS
    # --------------------------------------------------------

    def set_training_status(self, text):
        self.after(
            0,
            lambda: self.training_status.configure(text=text)
        )

    def set_metrics(self, cost, accuracy):
        self.after(
            0,
            lambda: self.metrics_label.configure(
                text=f"Coste: {cost:.4f} · Accuracy: {accuracy:.2f}%"
            )
        )

    def set_progress(self, value):
        self.after(
            0,
            lambda: self.training_progress.configure(value=value)
        )

    def log_model_ready(self):
        self.after(0, self.on_model_ready)

    def on_model_ready(self):
        self.predict_button.configure(state="normal")
        self.train_button.configure(state="normal")
        self.load_button.configure(state="normal")
        self.is_training = False

    def show_error(self, title, error):
        self.after(
            0,
            lambda: messagebox.showerror(title, str(error))
        )

        self.after(
            0,
            lambda: self.train_button.configure(state="normal")
        )

        self.after(
            0,
            lambda: self.load_button.configure(state="normal")
        )

        self.is_training = False

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def start_training(self):
        if self.is_training:
            return

        self.is_training = True

        self.train_button.configure(state="disabled")
        self.load_button.configure(state="disabled")
        self.predict_button.configure(state="disabled")
        self.training_progress.configure(value=0)
        self.metrics_label.configure(text="Coste: — · Accuracy: —")

        self.training_thread = threading.Thread(
            target=self.training_worker,
            daemon=True
        )
        self.training_thread.start()

    def training_worker(self):
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            context_size = int(self.context_size_var.get())
            embedding_dim = int(self.embedding_dim_var.get())
            max_vocab_size = int(self.max_vocab_size_var.get())
            learning_rate = float(self.learning_rate_var.get())

            selected_dataset_name = self.dataset_var.get()

            self.set_training_status("Leyendo dataset seleccionado...")

            lines, selected_files = read_dataset_files(selected_dataset_name)

            self.set_training_status(
                f"Dataset cargado: {len(selected_files)} archivo(s)"
            )

            word_to_index, index_to_word = build_vocabulary(
                lines,
                max_vocab_size=max_vocab_size
            )

            vocab_size = len(word_to_index)

            self.set_training_status("Creando ejemplos input/target...")

            x, y = create_training_examples(
                lines,
                word_to_index,
                context_size=context_size
            )

            self.set_training_status(
                f"Inicializando tu TLM · vocab_size={vocab_size}"
            )

            model = tlm_class.TLM(
                vocab_size=vocab_size,
                context_size=context_size,
                embedding_dim=embedding_dim
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )

            model.train()

            num_examples = x.shape[0]

            final_cost = 0.0
            final_accuracy = 0.0

            for epoch in range(epochs):
                permutation = torch.randperm(num_examples)

                total_cost = 0.0
                num_batches = 0

                for start in range(0, num_examples, batch_size):
                    batch_indices = permutation[start:start + batch_size]

                    batch_x = x[batch_indices]
                    batch_y = y[batch_indices]

                    # Uses YOUR implementation.
                    batch_cost = model.trainStep(
                        input=batch_x,
                        target=batch_y,
                        epochs=1,
                        optimizer=optimizer
                    )

                    total_cost += batch_cost
                    num_batches += 1

                average_cost = total_cost / num_batches

                # Uses YOUR accuracy method.
                accuracy = model.accuracy(
                    input=x,
                    target=y,
                    batch_size=batch_size
                )

                final_cost = average_cost
                final_accuracy = accuracy

                progress_value = ((epoch + 1) / epochs) * 100
                self.set_progress(progress_value)

                self.set_training_status(
                    f"Entrenando · época {epoch + 1}/{epochs}"
                )

                self.set_metrics(
                    cost=average_cost,
                    accuracy=accuracy
                )

            torch.save(model.state_dict(), MODEL_PATH)
            save_vocabulary(word_to_index, index_to_word)

            save_training_config(
                {
                    "vocab_size": vocab_size,
                    "context_size": context_size,
                    "embedding_dim": embedding_dim,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "final_cost": final_cost,
                    "final_accuracy": final_accuracy,
                    "datasets": [file.name for file in selected_files],
                    "number_of_examples": int(num_examples)
                }
            )

            self.model = model
            self.word_to_index = word_to_index
            self.index_to_word = index_to_word

            self.current_context_size = context_size
            self.current_embedding_dim = embedding_dim
            self.current_vocab_size = vocab_size

            self.model.eval()

            self.set_training_status(
                f"Modelo guardado · vocab_size={vocab_size} · ejemplos={num_examples}"
            )

            self.log_model_ready()

        except Exception as error:
            self.show_error("Error entrenando el modelo", error)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def load_saved_model(self):
        try:
            config = load_training_config()

            context_size = int(config["context_size"])
            embedding_dim = int(config["embedding_dim"])

            self.context_size_var.set(str(context_size))
            self.embedding_dim_var.set(str(embedding_dim))

            self.word_to_index, self.index_to_word = load_vocabulary()

            vocab_size = len(self.word_to_index)

            model = tlm_class.TLM(
                vocab_size=vocab_size,
                context_size=context_size,
                embedding_dim=embedding_dim
            )

            state_dict = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()

            self.model = model

            self.current_context_size = context_size
            self.current_embedding_dim = embedding_dim
            self.current_vocab_size = vocab_size

            final_cost = config.get("final_cost", None)
            final_accuracy = config.get("final_accuracy", None)

            self.training_status.configure(
                text=f"Modelo cargado · vocab_size={vocab_size} · context_size={context_size} · embedding_dim={embedding_dim}"
            )

            if final_cost is not None and final_accuracy is not None:
                self.metrics_label.configure(
                    text=f"Coste: {final_cost:.4f} · Accuracy: {final_accuracy:.2f}%"
                )

            self.predict_button.configure(state="normal")

        except Exception as error:
            messagebox.showerror(
                "Error cargando modelo",
                str(error)
            )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def clear_top_results(self):
        for widget in self.top_frame.winfo_children():
            widget.destroy()

    def predict(self):
        if self.model is None:
            return

        text = self.text_input.get("1.0", "end").strip()

        if text == "":
            self.prediction_label.configure(text="Escribe un texto")
            self.percentage_label.configure(text="Porcentaje: —")
            self.clear_top_results()
            return

        context_size = self.current_context_size

        x = prepare_context(
            text,
            word_to_index=self.word_to_index,
            context_size=context_size
        )

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=-1)

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=5,
                dim=-1
            )

        results = []

        for probability, index in zip(top_probabilities[0], top_indices[0]):
            word = self.index_to_word[index.item()]
            percentage = probability.item() * 100

            results.append((word, percentage))

        best_word, best_percentage = results[0]

        self.prediction_label.configure(text=best_word)

        self.percentage_label.configure(
            text=f"Porcentaje: {best_percentage:.2f}%"
        )

        self.clear_top_results()

        for word, percentage in results:
            row = ttk.Frame(self.top_frame, style="Card.TFrame")
            row.pack(fill="x", pady=5)

            word_label = ttk.Label(
                row,
                text=word,
                style="Normal.TLabel",
                width=18
            )
            word_label.pack(side="left")

            progress = ttk.Progressbar(
                row,
                maximum=100,
                value=percentage,
                style="Horizontal.TProgressbar"
            )
            progress.pack(
                side="left",
                fill="x",
                expand=True,
                padx=10
            )

            percentage_label = ttk.Label(
                row,
                text=f"{percentage:.2f}%",
                style="Muted.TLabel",
                width=10
            )
            percentage_label.pack(side="right")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app = TLMInterface()
    app.mainloop()
