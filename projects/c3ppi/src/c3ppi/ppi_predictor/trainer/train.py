import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve

from c3ppi.ppi_predictor.utils.config import MODEL_REGISTRY
from c3ppi.ppi_predictor.utils.logger import setup_logger
from c3ppi.ppi_predictor.models.C3BCA import PPI_Model, FocalLoss, get_ppi_data_loader

logger = setup_logger()
warnings.filterwarnings("ignore")

class Trainer:
    """
    Trainer class to handle the training and validation of the PPI prediction model.
    """

    def __init__(self, 
                 train_data_path: str,
                 val_data_path: str,
                 embeddings_dir: str,
                 save_name: str,
                 epochs: int = 100,
                 batch_size: int = 256, 
                 patience: int = 3):
        
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.embeddings_dir = embeddings_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.save_path = os.path.join(MODEL_REGISTRY, save_name)

    def run(self):
        """
        Main method to run training and validation.
        """

        # Check if best_model.pt exists
        if os.path.isfile(os.path.join(self.save_path, "best_model.pt")):
            logger.info("Best model already exists. Skipping training.")
            return

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Load data
        train_loader = get_ppi_data_loader(self.train_data_path, 
                                       self.embeddings_dir,
                                       batch_size=self.batch_size, 
                                       shuffle=True, 
                                       num_workers=0, 
                                       pin_memory=False)

        val_loader = get_ppi_data_loader(self.val_data_path, 
                                     self.embeddings_dir,
                                     batch_size=self.batch_size, 
                                     shuffle=False, 
                                     num_workers=0, 
                                     pin_memory=False)

        logger.info("Train data points: %d", len(train_loader.dataset))
        logger.info("Val data points: %d", len(val_loader.dataset))
        logger.info("Device: %s", self.device)


        # Initialize model, loss function, and optimizer
        self.model = PPI_Model().to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr= 0.001)

        # Define scheduler: reduce learning rate when a metric has stopped improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # Loss
        criterion = FocalLoss()

        # Early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            logger.info("Epoch %d/%d", epoch, self.epochs)

            # Training phase
            train_loss = self.train_one_epoch(train_loader, criterion, optimizer)
            logger.info("Train Loss: %.4f", train_loss)

            # Validation phase
            val_loss = self.validate(val_loader, criterion)
            logger.info("Val Loss: %.4f", val_loss)

            scheduler.step(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pt"))
                logger.info("Model saved to %s", os.path.join(self.save_path, "best_model.pt"))
            else:
                patience_counter += 1
                logger.info("Early stopping patience counter: %d/%d", patience_counter, self.patience)

            if patience_counter >= self.patience:
                logger.info("Early stopping triggered.")
                break

        logger.info("----------------------------------------")


    def train_one_epoch(self, 
                         loader: torch.utils.data.DataLoader, 
                         criterion: nn.Module, 
                         optimizer: optim.Optimizer) -> float:
        """
        Train the model for one epoch.

        Args:
            loader (torch.utils.data.DataLoader): The data loader for training data.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for data in tqdm(loader, desc="Training", leave=False):
            p1, p2, labels = data
            p1, p2, labels = p1.to(self.device).float(), p2.to(self.device).float(), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(p1, p2)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            
        return total_loss / len(loader.dataset)

    def validate(self, 
                  loader: torch.utils.data.DataLoader, 
                  criterion: nn.Module) -> float:
        """
        Validate the model on the validation dataset.

        Args:
            loader (torch.utils.data.DataLoader): The data loader for validation data.
            criterion (nn.Module): The loss function.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for data in tqdm(loader, desc="Validating", leave=False):
                p1, p2, labels = data
                p1, p2, labels = p1.to(self.device).float(), p2.to(self.device).float(), labels.to(self.device)

                outputs = self.model(p1, p2)
                loss = criterion(outputs, labels.unsqueeze(1))

                total_loss += loss.item() * labels.size(0)

                # Store predictions and labels for metrics calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(torch.sigmoid(outputs.cpu()).numpy())
                
                # Compute detailed metrics
        self.report_metrics(all_labels, all_predictions)

        return total_loss / len(loader.dataset)

    @staticmethod
    def report_metrics(true_labels: list, predicted_probs: list, threshold: float = 0.5) -> None:
        """
        Generate and log a detailed metrics report tailored for imbalanced protein-protein interaction prediction.

        Args:
            true_labels (list): Ground truth labels (0 or 1).
            predicted_probs (list): Model predictions (probabilities of the positive class).
            threshold (float): Threshold for converting probabilities to binary predictions.

        Returns:
            None
        """
        # Convert probabilities to binary predictions based on the threshold
        predicted_labels = (np.array(predicted_probs) >= threshold).astype(int)

        # Core Metrics for Imbalanced Classification
        precision_macro = precision_score(true_labels, predicted_labels, average="macro")
        recall_macro = recall_score(true_labels, predicted_labels, average="macro")
        f1_macro = f1_score(true_labels, predicted_labels, average="macro")

        precision_binary = precision_score(true_labels, predicted_labels, average='binary') #for the interacting class
        recall_binary = recall_score(true_labels, predicted_labels, average='binary') #for the interacting class
        f1_binary = f1_score(true_labels, predicted_labels, average='binary') #for the interacting class

        mcc = matthews_corrcoef(true_labels, predicted_labels)
        balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

        # Curve-based Metrics
        ap = average_precision_score(true_labels, predicted_probs)
        try:
            auroc = roc_auc_score(true_labels, predicted_probs)
        except ValueError:
            logger.warning("Only one class present in y_true. ROC AUC score is not defined in that case.")
            auroc = np.nan

        # Precision-Recall Curve and Specific Recall@Precision
        precision, recall, thresholds = precision_recall_curve(true_labels, predicted_probs)
        recall_at_precision_50 = recall[np.argmin(np.abs(precision - 0.5))]

        logger.info("Macro-averaged Metrics:")
        logger.info("  Precision (Macro): %.4f", precision_macro)
        logger.info("  Recall (Macro): %.4f", recall_macro)
        logger.info("  F1-Score (Macro): %.4f", f1_macro)
        logger.info("Interacting Class Metrics:")
        logger.info("  Precision (Interacting): %.4f", precision_binary)
        logger.info("  Recall (Interacting): %.4f", recall_binary)
        logger.info("  F1-Score (Interacting): %.4f", f1_binary)
        logger.info("Imbalance-Aware Metrics:")
        logger.info("  Matthews Correlation Coefficient (MCC): %.4f", mcc)
        logger.info("  Balanced Accuracy: %.4f", balanced_acc)
        logger.info("  Average Precision (AP): %.4f", ap)
        logger.info("  Area Under ROC Curve (AUROC): %.4f", auroc)
        logger.info("  Recall@Precision=0.5: %.4f", recall_at_precision_50)
