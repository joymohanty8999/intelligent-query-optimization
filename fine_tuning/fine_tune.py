from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import pandas as pd
import os
import logging


def fine_tune_model(model_name, train_data, save_path, batch_size=32, epochs=1):
    """
    Fine-tunes a SentenceTransformer model on a dataset, handling missing data gracefully.
    """
    # Load the model
    model = SentenceTransformer(model_name)

    # Unpack train_data
    queries, qrels, collection = train_data

    # Prepare training examples
    train_examples = []
    for _, row in qrels.iterrows():
        # Get the query and passage; use default values if not found
        query = queries.loc[queries['QueryID'] == row['QueryID'], 'Query']
        passage = collection.loc[collection['PassageID'] == row['PassageID'], 'Passage']

        query_text = query.values[0] if not query.empty else "Default query text"
        passage_text = passage.values[0] if not passage.empty else "Default passage text"

        train_examples.append(InputExample(texts=[query_text, passage_text], label=float(row['Relevance'])))

    print(f"Prepared {len(train_examples)} training examples (missing data handled).")

    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # Define loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        show_progress_bar=True
    )

    # Save the fine-tuned model
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
    print(f"Fine-tuned model saved at {save_path}")