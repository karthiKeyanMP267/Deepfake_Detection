import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

import torch
from training.train import Trainer, set_seed
from training.evaluate import Evaluator


def run_experiment():
    print("Initializing Experiment: class_weight...")
    set_seed(42)

    train_csv = "processed/train.csv"
    val_csv = "processed/val.csv"
    test_csv = "processed/test.csv"

    batch_size = 2
    epochs = 15
    learning_rate = 1e-4
    weight_decay = 1e-5
    subsample_ratio = 0.25
    experiment_name = "class_weight"

    # Instantiate Trainer with use_sampler=False and use_class_weights=True
    trainer = Trainer(
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        subsample_ratio=subsample_ratio,
        experiment_name=experiment_name,
        use_sampler=False,
        use_class_weights=True,
        test_csv=test_csv,
    )

    # Execute training loop
    trainer.fit()

    # Verify best_model.pth checkpoint path before evaluation
    expected_checkpoint_path = os.path.join("experiments", "class_weight", "models", "best_model.pth")
    if not os.path.exists(expected_checkpoint_path):
        raise FileNotFoundError(f"Expected checkpoint not found at: {expected_checkpoint_path}")

    print("\n" + "=" * 70)
    print(f"VERIFYING CHECKPOINT PATH FOR EVALUATION: {expected_checkpoint_path}")
    print("=" * 70)

    # Run evaluation loading ONLY experiments/class_weight/models/best_model.pth
    evaluator = Evaluator(
        test_csv=test_csv,
        checkpoint_path=expected_checkpoint_path,
        experiment_name=experiment_name,
        batch_size=batch_size,
    )

    test_metrics = evaluator.evaluate()
    evaluator.print_results(test_metrics)


if __name__ == "__main__":
    run_experiment()
