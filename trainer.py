import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, precision_score, f1_score, average_precision_score
from tqdm import tqdm
import wandb
import os
import warnings
warnings.filterwarnings('ignore')

class NewsDataModule(pl.LightningDataModule):
    def __init__(self, ccnews_size=1500, c4_size=13500, batch_size=32, embedding_model='all-MiniLM-L6-v2'):
        super().__init__()
        self.ccnews_size = ccnews_size
        self.c4_size = c4_size
        self.batch_size = batch_size
        self.embedding_model = embedding_model

    def prepare_data(self):
        """Download and prepare datasets to train/test/val"""
        char_limit = 50

        # TODO: Shuffle raw data?

        print(f"Loading CC-News dataset w/ at least {char_limit} characters...")
        ccnews_dataset = load_dataset("stanford-oval/ccnews", split="train", streaming=True, token=os.environ["HF_TOKEN"])

        ccnews_texts = []
        for i, example in enumerate(tqdm(ccnews_dataset, desc="Loading CC-News")):
            if i >= self.ccnews_size:
                break
            text = example.get('plain_text')
            if text and len(text.strip()) > char_limit:
                ccnews_texts.append(text[:512])

        print("Loading C4 dataset...")
        c4_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, token=os.environ["HF_TOKEN"])

        c4_texts = []
        for i, example in enumerate(tqdm(c4_dataset, desc="Loading C4")):
            if i >= self.c4_size:
                break
            text = example.get('text', '')
            if text and len(text.strip()) > 50:
                c4_texts.append(text[:512])

        # Init dfs
        ccnews_df = pd.DataFrame({'text': ccnews_texts, 'label': 1})
        c4_df = pd.DataFrame({'text': c4_texts, 'label': 0})

        # Splits
        ccnews_shuffled = ccnews_df.sample(frac=1, random_state=42).reset_index(drop=True)
        ccnews_train_size = int(0.6 * len(ccnews_shuffled))
        ccnews_val_size = int(0.2 * len(ccnews_shuffled))

        ccnews_train = ccnews_shuffled[:ccnews_train_size]
        ccnews_val = ccnews_shuffled[ccnews_train_size:ccnews_train_size + ccnews_val_size]
        ccnews_test = ccnews_shuffled[ccnews_train_size + ccnews_val_size:]

        c4_shuffled = c4_df.sample(frac=1, random_state=42).reset_index(drop=True)
        c4_train_size = int(0.6 * len(c4_shuffled))
        c4_val_size = int(0.2 * len(c4_shuffled))

        c4_train = c4_shuffled[:c4_train_size]
        c4_val = c4_shuffled[c4_train_size:c4_train_size + c4_val_size]
        c4_test = c4_shuffled[c4_train_size + c4_val_size:]

        # Create balanced train set
        min_train_size = min(len(ccnews_train), len(c4_train))
        self.train_df = pd.concat([
            ccnews_train[:min_train_size],
            c4_train[:min_train_size]
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Create imbalanced val set (30% news, 70% non-news) so we can mimic test a bit
        val_ccnews_size = int(0.3 * (len(ccnews_val) + len(c4_val)))
        val_c4_size = int(0.7 * (len(ccnews_val) + len(c4_val)))

        self.val_df = pd.concat([
            ccnews_val[:val_ccnews_size],
            c4_val[:val_c4_size]
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Create imbalanced test test (10% news, 90% non-news)
        test_ccnews_size = int(0.1 * (len(ccnews_test) + len(c4_test)))
        test_c4_size = int(0.9 * (len(ccnews_test) + len(c4_test)))

        self.test_df = pd.concat([
            ccnews_test[:test_ccnews_size],
            c4_test[:test_c4_size]
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Training set: {len(self.train_df)} samples (balanced)")
        print(f"Validation set: {len(self.val_df)} samples (slightly imbalanced)")
        print(f"Test set: {len(self.test_df)} samples (10-90)")

    def setup(self, stage=None):
        """Set up datasets for different stages"""
        self.embedder = SentenceTransformer(self.embedding_model)

        if stage == 'fit':
            self.train_dataset = DatasetEmbedded(
                self.train_df['text'].tolist(),
                self.train_df['label'].values,
                self.embedder
            )
            self.val_dataset = DatasetEmbedded(
                self.val_df['text'].tolist(),
                self.val_df['label'].values,
                self.embedder
            )

        if stage == 'test':
            self.test_dataset = DatasetEmbedded(
                self.test_df['text'].tolist(),
                self.test_df['label'].values,
                self.embedder
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)


class DatasetEmbedded(Dataset):
    """Simple dataset that also returns embedding representation"""
    def __init__(self, texts, labels, embedder):
        self.texts = texts
        self.labels = labels
        self.embedder = embedder

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        embedding = self.embedder.encode([text])[0]
        return {
            'embedding': torch.tensor(embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


class FocalLoss(nn.Module):
    """Focal loss can help weigh hard examples harder"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class NewsClassifierLightning(pl.LightningModule):
    def __init__(self, embedding_dim=384, hidden_dim=256, num_classes=2, learning_rate=1e-3, use_focal_loss=True):
        super().__init__()
        self.save_hyperparameters()

        # Baseline: MLP with some layers and dropouts
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Define loss fn
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=2, gamma=2)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss()

        # For storing predictions
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # For storing best prob threshold from val set
        self.best_threshold = 0.5


    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        embeddings = batch['embedding']
        labels = batch['label']

        logits = self(embeddings)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        embeddings = batch['embedding']
        labels = batch['label']

        logits = self(embeddings)
        loss = self.criterion(logits, labels)

        # Calculate acc
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        probs = F.softmax(logits, dim=1)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        # Store for epoch_end calculations
        self.validation_step_outputs.append({
            'preds': preds.cpu(),
            'labels': labels.cpu(),
            'probs': probs[:, 1].cpu()  # Prob of 'news' class
        })

        return loss

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
            all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])

            # Calculate metrics
            precision, recall, thresholds = precision_recall_curve(all_labels.numpy(), all_probs.numpy(), pos_label=1)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

            # 1. average precision achieved at each threshold
            ap = average_precision_score(all_labels.numpy(), all_probs.numpy(), pos_label=1)
            self.log('val_news_average_precision', ap, on_epoch=True, prog_bar=True)

            # 2. news class precision and F1 (threshold=0.5 here)
            news_precision = precision_score(all_labels.numpy(), all_preds.numpy(), pos_label=1, zero_division=0)
            news_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), pos_label=1, zero_division=0)
            self.log('val_news_precision', news_precision, on_epoch=True, prog_bar=True)
            self.log('val_news_f1', news_f1, on_epoch=True, prog_bar=True)

            # Additionally, tune the best prob threshold
            beta = 0.5 # (favors precision 4x more than recall)
            f05_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)
            best_threshold_idx = np.argmax(f05_scores)
            self.best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
            best_f05 = f05_scores[best_threshold_idx] if best_threshold_idx < len(f05_scores) else 0.0
            self.log('val_news_f05', best_f05, on_epoch=True, prog_bar=True)
            self.log('val_best_threshold', self.best_threshold, on_epoch=True, prog_bar=True)

            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        embeddings = batch['embedding']
        labels = batch['label']
        texts = batch['text']

        logits = self(embeddings)
        loss = self.criterion(logits, labels)

        # Calculate acc
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Get probs
        probs = F.softmax(logits, dim=1)

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)

        # Store for epoch_end calculations
        self.test_step_outputs.append({
            'preds': preds.cpu(),
            'labels': labels.cpu(),
            'probs': probs[:, 1].cpu(),
            'texts': texts
        })

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
            all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
            all_texts = []
            for x in self.test_step_outputs:
                all_texts.extend(x['texts'])

            # Calculate metrics
            # 1. average precisionachieved at each threshold
            ap = average_precision_score(all_labels.numpy(), all_probs.numpy(), pos_label=1)
            self.log('test_news_average_precision', ap, on_epoch=True, prog_bar=True)

            # 2. news class precision and F1
            news_precision = precision_score(all_labels.numpy(), all_preds.numpy(), pos_label=1, zero_division=0)
            news_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), pos_label=1, zero_division=0)
            self.log('test_news_precision', news_precision, on_epoch=True)
            self.log('test_news_f1', news_f1, on_epoch=True)

            # 3. f0.5
            beta = 0.5
            news_recall = (news_f1 * news_precision) / (2 * news_precision - news_f1 + 1e-8) if news_precision > 0 and news_f1 > 0 else 0.0
            news_f05 = (1 + beta**2) * (news_precision * news_recall) / (beta**2 * news_precision + news_recall + 1e-8) if news_precision > 0 and news_recall > 0 else 0.0
            self.log('test_news_f05', news_f05, on_epoch=True)

            # 3. metrics with tuned threshold
            preds_optimized = (all_probs >= self.best_threshold).int()
            news_precision_optimized = precision_score(all_labels.numpy(), preds_optimized.numpy(), pos_label=1, zero_division=0)
            news_f1_optimized = f1_score(all_labels.numpy(), preds_optimized.numpy(), pos_label=1, zero_division=0)
            news_recall_optimized = (news_f1_optimized * news_precision_optimized) / (2 * news_precision_optimized - news_f1_optimized + 1e-8) if news_precision_optimized > 0 and news_f1_optimized > 0 else 0.0
            news_f05_optimized = (1 + beta**2) * (news_precision_optimized * news_recall_optimized) / (beta**2 * news_precision_optimized + news_recall_optimized + 1e-8) if news_precision_optimized > 0 and news_recall_optimized > 0 else 0.0

            self.log('test_news_precision_optimized', news_precision_optimized, on_epoch=True)
            self.log('test_news_f1_optimized', news_f1_optimized, on_epoch=True)
            self.log('test_news_f05_optimized', news_f05_optimized, on_epoch=True)

            # Print some results
            print("\nTest Results:")
            print(f"Test average_precision: {ap:.4f}")
            print("\nClassification Report:")
            print(classification_report(all_labels.numpy(), all_preds.numpy(),
                                      target_names=['Non-News', 'News']))

            print(f"\nClassification Report (threshold={self.best_threshold:.4f}):")
            print(classification_report(all_labels.numpy(), preds_optimized.numpy(),
                                  target_names=['Non-News', 'News']))

            # Also log a table of test samples with wandb
            preds_np = all_preds.numpy()
            labels_np = all_labels.numpy()
            probs_np = all_probs.numpy()
            preds_optimized_np = preds_optimized.numpy()

            # Create table
            table = wandb.Table(columns=["text", "news_prob", "pred", "pred_optimized", "gt"])
            for idx in range(len(all_texts)):
                label_name = "news" if labels_np[idx] == 1 else "c4"
                pred_name = "news" if preds_np[idx] == 1 else "c4"
                pred_optimized_name = "news" if preds_optimized_np[idx] == 1 else "c4"
                table.add_data(
                    all_texts[idx],
                    float(probs_np[idx]),
                    pred_name,
                    pred_optimized_name,
                    label_name,
                )
            wandb.log({f"test_samples_{wandb.run.name}": table})

            self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train news classifier')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2',
                        choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'google/embeddinggemma-300m'],
                        help='Embedding model to use')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--focal', action='store_true', help='Use focal loss or not')
    args = parser.parse_args()

    # Set run name
    run_name = f"{args.embedding_model.replace('/', '-')}-lr{args.lr}-bs{args.bs}-focal{args.focal}"

    # Initialize wandb
    wandb.init(
        # mode="disabled", # disable while debuging
        project="ccnews-classification",
        name=run_name,
        config={
            "embedding_model": args.embedding_model,
            "hidden_dim": 256,
            "learning_rate": args.lr,
            "batch_size": args.bs,
            "max_epochs": 10,
            "ccnews_size": 5000,
            "c4_size": 13500,
            "use_focal_loss": args.focal
        }
    )

    config = wandb.config

    # Initialize data module
    data_module = NewsDataModule(
        ccnews_size=config.ccnews_size,
        c4_size=config.c4_size,
        batch_size=config.batch_size,
        embedding_model=config.embedding_model,
    )

    # Prepare data
    data_module.prepare_data() # For PL, we write df's
    data_module.setup()

    # Initialize model
    embedding_dim = SentenceTransformer(config.embedding_model).get_sentence_embedding_dimension()
    model = NewsClassifierLightning(
        embedding_dim=embedding_dim,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        use_focal_loss=config.use_focal_loss,
    )

    # Initialize logger and callbacks
    wandb_logger = WandbLogger(project="news-classification")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best-checkpoint-{epoch:02d}-{val_loss:.3f}'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=3,
        verbose=True
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        devices='auto',
        # gradient_clip_val=0.5  # Saw some spiky loss so try gradient clips
    )

    print(f"\nClassifier parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("Training model...")
    trainer.fit(model, data_module)

    # Test model
    print("Performing on test sets...")
    trainer.test(model, data_module, ckpt_path='best')

    wandb.finish()
