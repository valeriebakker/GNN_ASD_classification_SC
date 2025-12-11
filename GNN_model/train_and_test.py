import os
import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import datetime

import utils as Reader
from GNNModel import GNNmodel
from metrics import auc, prf, specificity, npv, mean_std, plot_accuracies, plot_losses, plot_cv_results

def train_gnn_cv(args, dev_features, dev_y_true, pheno_affinity_matrix, n_folds, device=None):
    """
    Train and evaluate the GNN using k-fold cross-validation on the development set.

    Inputs:
        args: hyperparameters.
        dev_features: SC feature matrix for all development subjects (N x F)
        dev_y_true: ground-truth labels for development subjects (N,)
        pheno_affinity_matrix: phenotypic similarity matrix (N x N) or None
        n_folds: number of cross-validation folds

    Outputs:
        model: final model object (from the last fold)
        fold_results: list of dictionaries containing metrics for each fold
        mean_results: fold-averaged metrics with mean ± std
        results_dir: path where plots and results were saved
    """

    if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

    # Create unique result directory for this run
    my_path = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(my_path, f"Results/modelresults_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    num_nodes = len(dev_y_true)  # Total number of nodes/subjects (train set)

    # Stratified CV to obtain similar percentage of each class in each of the folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)  
    cv_splits = list(skf.split(dev_features, dev_y_true))

    fold_results = []

    # K-Fold cross-validation
    for fold in range(n_folds):
        print(f"Training on fold {fold + 1}/{n_folds}")

        train_idx, val_idx = cv_splits[fold]

        # Feature selection on all development subjects,
        # but using only train_idx internally to decide which features to keep
        node_ftr_all, selected_idx = Reader.feature_selection(
            dev_features,
            dev_y_true,
            train_idx,
            args.node_ftr_dim)
        
        # Combine feature similarity (RBF) with phenotypic similarity for total affinity graph.
        # If pheno_affinity_matrix = None, only RBF similarity matrix will be used.
        edge_index, edge_attr, aff_graph = Reader.compute_total_affinity_graph(
                        node_ftr_all, pheno_affinity_matrix,
                        graph_method=args.graph_method,
                        affinity_threshold=args.affinity_threshold,
                        k=args.k)

        # isolated, degrees = Reader.check_graph_connectivity(edge_index, aff_graph.shape[0], plot=True)

        # Convert the variables to tensors
        node_ftr_all = torch.tensor(node_ftr_all, dtype=torch.float32)
        y_all = torch.tensor(dev_y_true, dtype=torch.long)
        num_nodes = node_ftr_all.shape[0]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_idx] = True

        # Create a new copy of the Data object with updated attributes
        data_in = Data( x=node_ftr_all,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y_all,
                        train_mask=train_mask,
                        val_mask=val_mask)

        print(f'Dataset: {data_in}:')

        # Initialize variables to save each metric values
        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        best_train_acc = best_val_acc = 0.0
        best_train_loss = best_val_loss = float('inf')
        best_model = None
        best_epoch = 0
        best_auc = best_spe = best_npv = 0.0
        best_prfs = [0.0, 0.0, 0.0]  # [precision, recall, f1]
        val_aucs = []

        patience = args.patience    # Early stopping criterium
        epochs_no_improve = 0       # Counter

        print("Number of training samples %d" % int(data_in.train_mask.sum().item()))
        print("Start training...\r\n")

        # Create a new instance of the model for the respective fold
        model = GNNmodel(num_features=data_in.num_node_features,
                         edgenet_input_dim=data_in.edge_attr.size(-1),
                         out_dim=args.num_classes,
                         args=args
                         ).to(device)

        # Updates model weights using adaptive learning rates
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # LR scheduler: reduces LR by factor 0.7 when validation loss stops improving
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)

        ############
        # # Below is a sanity check for using class weights on the data.

        # # Class weights over full dev set using inverse-frequency weights
        # class_counts = np.bincount(dev_y_true.astype(int))
        # class_weights = class_counts.sum() / (2.0 * class_counts)
        # class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

        # classes = np.array([0,1])
        # weights = compute_class_weight(
        #     class_weight='balanced',
        #     classes=classes,
        #     y=dev_y_true)
        # weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Cross-entropy loss for binary (ASD vs TDC) classification
        criterion = torch.nn.CrossEntropyLoss()

        # Check if batching (neighborhood sampling) should be performed
        if args.use_batching:
            train_loader = NeighborLoader(data_in, input_nodes=data_in.train_mask, num_neighbors=args.num_neighbors, 
                                      shuffle=True, batch_size=args.batch_size)
            val_loader = NeighborLoader(data_in, input_nodes=data_in.val_mask, num_neighbors=[-1],
                                        batch_size=args.batch_size)
            
        # No batching
        else:
            x = data_in.x.to(device)
            edge_index = data_in.edge_index.to(device)
            edge_attr = data_in.edge_attr.to(device)
            y = data_in.y.to(device)
            train_mask = data_in.train_mask.to(device)
            val_mask = data_in.val_mask.to(device)

        for epoch in range(args.epochs):
            model.current_epoch = epoch

            # Check again for batching in the epoch loop:
            if args.use_batching:
                ##### TRAIN #####
                model.train()
                total_examples = sum_train_losses = total_corrects = 0
                with torch.set_grad_enabled(True):
                    for batch in train_loader:
                        optimizer.zero_grad()
                        batch = batch.to(device)

                        y_batch = batch.y[:batch.batch_size]  # Select the subject labels for the respective batch

                        logits = model(batch.x, batch.edge_index, batch.edge_attr)          # Perform a single forward pass
                        batch_train_loss = criterion(logits[:batch.batch_size], y_batch)    # Compute the loss solely based on the seed nodes
                        batch_train_loss.backward()     # Derive gradients
                        optimizer.step()                # Update parameters based on gradients

                        batch_train_pred = logits[:batch.batch_size].argmax(dim=-1)         # Model's predictions for subjects in the respective batch

                        total_corrects += int((batch_train_pred == y_batch).sum())          # Counting how many correct predictions the model performed
                        total_examples += batch.batch_size                                  # Number of samples seen in training
                        sum_train_losses += float(batch_train_loss) * batch.batch_size
                
                train_loss = sum_train_losses / total_examples
                train_losses.append(train_loss)
                train_acc = total_corrects / total_examples     # Compute the accuracy for the training nodes
                train_accs.append(train_acc)

                ##### VALIDATION #####
                model.eval()
                val_pred = []       # Predicted labels for all validation nodes
                y_true_val = []     # True labels for all validation nodes
                val_probs_pos = []  # Probability for the positive class (needed for AUC)
                total_examples = sum_val_losses = total_corrects = 0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        logits = model(batch.x, 
                                    batch.edge_index, 
                                    batch.edge_attr)            # Logits for all nodes in subgraph (num_nodes, 2)
                        y_batch = batch.y[:batch.batch_size]    # Take labels of seed nodes only
                        batch_val_loss = criterion(logits[:batch.batch_size], y_batch)

                        batch_val_probs = torch.softmax(logits[:batch.batch_size], dim=1) # Class probabilities
                        batch_val_probs_pos = batch_val_probs[:, 0]                       # Positive (ASD) class probability (label 0 = ASD)

                        batch_val_pred = logits[:batch.batch_size].argmax(dim=1)    # Threshold at 0.5 for predictions

                        total_corrects += int((batch_val_pred == y_batch).sum())    # Counting how many correct predictions the model performed
                        total_examples += batch.batch_size                          # Number of samples seen in validation
                        sum_val_losses += float(batch_val_loss) * batch.batch_size

                        val_pred.extend(batch_val_pred.detach().cpu().numpy().tolist())            # for PRF, specificity, etc.
                        val_probs_pos.extend(batch_val_probs_pos.detach().cpu().numpy().tolist())  # for AUC
                        y_true_val.extend(y_batch.detach().cpu().numpy().tolist())

                val_loss = sum_val_losses / total_examples
                val_losses.append(val_loss)
                val_acc = total_corrects / total_examples
                val_accs.append(val_acc)

            # No batching:
            else:
                ##### TRAIN #####
                model.train()
                optimizer.zero_grad()

                logits = model(x, edge_index, edge_attr)
                logits_train = logits[train_mask]
                y_train = y[train_mask]

                train_loss = criterion(logits_train, y_train)
                train_loss.backward()
                optimizer.step()
                train_loss = float(train_loss.item())

                train_pred = logits_train.argmax(dim=-1)
                train_acc = float((train_pred == y_train).float().mean().item())

                train_losses.append(train_loss)
                train_accs.append(train_acc)

                ##### VALIDATION #####
                model.eval()
                with torch.no_grad():
                    logits = model(x, edge_index, edge_attr)
                    logits_val = logits[val_mask]
                    y_val = y[val_mask]

                    val_loss = criterion(logits_val, y_val)
                    val_loss = float(val_loss.item())
                    val_losses.append(val_loss)

                    val_probs = torch.softmax(logits_val, dim=1)
                    val_probs_pos = val_probs[:, 0]  # ASD = class 0

                    val_pred = logits_val.argmax(dim=1)

                    val_acc = float((val_pred == y_val).float().mean().item())
                    val_accs.append(val_acc)

                    val_pred = val_pred.detach().cpu().numpy().tolist()
                    val_probs_pos = val_probs_pos.detach().cpu().numpy().tolist()
                    y_true_val = y_val.detach().cpu().numpy().tolist()
        
            # Calculate the other metrics
            auc_score = auc(np.column_stack([1 - np.array(val_probs_pos),
                            np.array(val_probs_pos)]),
                            y_true_val, 
                            is_logit=False)
            
            val_aucs.append(auc_score)
            
            precision, recall, f1 = prf(val_pred, y_true_val)

            # print(f'Epoch: {epoch:>3}, Train Loss: {train_loss:.3f}, '
            #     f'Train Acc: {train_acc * 100:>5.2f}%, '
            #     f'Val Loss: {val_loss:.2f}, '
            #     f'Val Acc: {val_acc * 100:.2f}%, '
            #     f'Val AUC: {auc_score:.2f}')
            
            scheduler.step(val_loss)

            min_epochs = 50

            # Do not start early stopping before min_epochs
            if epoch < min_epochs:
                continue

            # Track early stopping on validation loss
            if val_loss < best_val_loss:
                best_auc = auc_score
                best_train_loss = train_loss
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_val_acc = val_acc
                best_prfs = [precision, recall, f1]
                best_spe = specificity(val_pred, y_true_val)
                best_npv = npv(val_pred, y_true_val)
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch >= 200 and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\n => Fold {fold + 1} best AUC {best_auc:.5f} at epoch {best_epoch}")

        # Save intermediate fold results
        fold_res = {
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'best_train_loss': float(best_train_loss),
            'best_val_loss': float(best_val_loss),
            'best_train_acc': float(best_train_acc),
            'best_val_acc': float(best_val_acc),
            'best_auc': float(best_auc),
            'best_precision': float(best_prfs[0]),
            'best_recall': float(best_prfs[1]),
            'best_f1': float(best_prfs[2]),            
            'specificity': float(best_spe),
            'npv': float(best_npv)}

        fold_results.append(fold_res)

        if args.ckpt_path:
            ckpt_dir = os.path.join(results_dir, args.ckpt_path)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(best_model, os.path.join(ckpt_dir, f'best_model_fold{fold+1}.pth'))

        # Plot training and validation accuracies and losses of each fold
        plot_accuracies(train_accs, val_accs, results_dir, fold=fold)
        plot_losses(train_losses, val_losses, results_dir, fold=fold)

    # Plot CV results
    plot_cv_results(fold_results, os.path.join(results_dir, 'cv_results.png'))

    mean_results = {'train_loss': mean_std([r['best_train_loss'] for r in fold_results]),
                    'val_loss': mean_std([r['best_val_loss'] for r in fold_results]),
                    'train_accuracy': mean_std([r['best_train_acc'] for r in fold_results]),
                    'val_accuracy': mean_std([r['best_val_acc'] for r in fold_results]),
                    'auc': mean_std([r['best_auc'] for r in fold_results]),
                    'precision': mean_std([r['best_precision'] for r in fold_results]),
                    'recall': mean_std([r['best_recall'] for r in fold_results]),
                    'f1_score': mean_std([r['best_f1'] for r in fold_results]),
                    'specificity': mean_std([r['specificity'] for r in fold_results]),
                    'npv': mean_std([r['npv'] for r in fold_results])}
        
    return model, fold_results, mean_results, results_dir


def refit_final_model(args, dev_features, dev_y_true, pheno_affinity_matrix, results_dir, device):
    """
    Refit the final GNN model on the full development set using the best hyperparameters.

    Inputs:
        args: hyperparameters.
        dev_features: SC feature matrix for all development subjects.
        dev_y_true: corresponding ASD/TDC labels.
        pheno_affinity_matrix: phenotypic similarity (dev x dev).
        results_dir: folder to save plots.

    Outputs:
        model: trained GNN model on full development set.
        node_ftr: reduced feature matrix after feature selection.
        selected_idx: indices of selected SC features.
        final_model_results: final training loss and accuracy.
    """

    # Feature selection on ALL development data
    node_ftr, selected_idx = Reader.feature_selection(
        dev_features, dev_y_true,
        np.arange(len(dev_y_true)),
        args.node_ftr_dim)

    # 2. Build population graph on ALL development data
    edge_index, edge_attr, aff_graph = Reader.compute_total_affinity_graph(
        node_ftr, pheno_affinity_matrix,
        graph_method=args.graph_method,
        affinity_threshold=args.affinity_threshold,
        k=args.k)

    x = torch.tensor(node_ftr, dtype=torch.float32, device=device)
    y = torch.tensor(dev_y_true, dtype=torch.long, device=device)

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    data_in = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Store training curves
    train_losses = []
    train_accs = []

    # Create model
    model = GNNmodel(
        num_features=data_in.num_node_features,
        edgenet_input_dim=data_in.edge_attr.size(-1),
        out_dim=args.num_classes,
        args=args
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)
    criterion = torch.nn.CrossEntropyLoss()

    # Batching option
    if args.use_batching:
        train_loader = NeighborLoader(
            data_in,
            input_nodes=torch.arange(len(data_in.y)),  # all nodes
            num_neighbors=args.num_neighbors,
            shuffle=True,
            batch_size=args.batch_size
        )

    for epoch in range(args.epochs):
        model.train()

        # With batching
        if args.use_batching:
            total_loss = 0.0
            total_correct = 0
            total_examples = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                y_batch = batch.y[:batch.batch_size]

                loss = criterion(logits[:batch.batch_size], y_batch)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * batch.batch_size
                preds = logits[:batch.batch_size].argmax(dim=1)
                total_correct += int((preds == y_batch).sum())
                total_examples += batch.batch_size

            train_loss = total_loss / total_examples
            train_acc = total_correct / total_examples

        # No batching:
        else:
            optimizer.zero_grad()

            logits = model(data_in.x, data_in.edge_index, data_in.edge_attr)
            loss = criterion(logits, data_in.y)
            loss.backward()
            optimizer.step()

            # Accuracy
            preds = logits.argmax(dim=1)
            train_acc = (preds == data_in.y).float().mean().item()

            train_loss = float(loss.item())

            scheduler.step(train_loss)

        # Save curves
        train_losses.append(train_loss)
        train_accs.append(train_acc)

    # Plot training loss
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Final refit training loss")
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(results_dir, "final_refit_train_loss.png"))
    plt.close()

    # Plot training accuracy
    plt.figure()
    plt.plot(train_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.title("Final refit training accuracy")
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(results_dir, "final_refit_train_accuracy.png"))
    plt.close()

    # Prepare output
    final_model_results = {
        "epochs_trained": args.epochs,
        "final_train_loss": float(train_loss),
        "final_train_accuracy": float(train_acc),
    }

    model.eval()

    return model, node_ftr, selected_idx, final_model_results


def test_gnn_inductive(args, model, dev_features, selected_idx, pheno_affinity_dev, test_features,
    test_labels, pheno_affinity_test_to_dev, device=None, pos_class_index=0):
    """
    Inductive testing:
    For each test subject, an augmented graph (development graph + 1 test node) is built.
    The test node only connects to the development nodes (no test-test edges).
    Run the model, take the logits for that single test node, collect metrics.

    Inputs:
        args: hyperparameters.
        model: trained GNN model (from refit_final_model).
        dev_features: reduced SC features for development subjects.
        selected_idx: feature indices selected during training.
        pheno_affinity_dev: phenotypic similarity (dev × dev).
        test_features: full SC features for test subjects.
        test_labels: true ASD/TDC labels for test subjects.
        pheno_affinity_test_to_dev: phenotypic similarity (test × dev).
        device: computation device ('cpu' or 'cuda').
        pos_class_index: index of the ASD class (0).

    Outputs:
        final_test_results: dict with all test metrics and predicted probabilities.

    """

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Apply the exact same feature selection used in training
    dev_ftr = np.asarray(dev_features)      # Already reduced to args.node_ftr_dim features
    test_ftr = np.asarray(test_features)    # Still contain 7140 features

    test_ftr = test_ftr[:, selected_idx]
    
    num_nodes_dev = dev_ftr.shape[0]
    num_nodes_test = test_ftr.shape[0]

    # Sanity checks for phenotypic matrices
    pheno_aff_matrix_dev = np.asarray(pheno_affinity_dev)
    cross_aff_matrix = np.asarray(pheno_affinity_test_to_dev)

    assert pheno_aff_matrix_dev.shape == (num_nodes_dev, num_nodes_dev), \
        f"pheno_aff_matrix_dev must be ({num_nodes_dev},{num_nodes_dev})"
    assert cross_aff_matrix.shape == (num_nodes_test, num_nodes_dev), \
        f"pheno_affinity_test_to_dev must be ({num_nodes_test},{num_nodes_dev})"

    # Collect results
    y_true = np.asarray(test_labels).astype(int)
    prob_pos_list = []   # probabilities for positive class (ASD = class 0)
    pred_list = []       # argmax predictions for each test subject

    with torch.no_grad():
        for subj in range(num_nodes_test):
            # Build augmented features: development set + one test subject
            features_aug = np.vstack([dev_ftr, test_ftr[subj:subj+1, :]]) 

            if args.use_pheno_data:
                cross_row = cross_aff_matrix[subj, :].reshape(1, num_nodes_dev)
                cross_col = cross_aff_matrix[subj, :].reshape(num_nodes_dev, 1)

                pheno_aug = np.zeros((num_nodes_dev + 1, num_nodes_dev + 1), dtype=float)
                pheno_aug[:num_nodes_dev, :num_nodes_dev] = pheno_aff_matrix_dev
                pheno_aug[:num_nodes_dev, num_nodes_dev] = cross_col.ravel()
                pheno_aug[num_nodes_dev, :num_nodes_dev] = cross_row.ravel()

                # Self-similarity for the single test node
                pheno_aug[num_nodes_dev, num_nodes_dev] = 3.0  

            else:
                pheno_aug = np.ones((num_nodes_dev + 1, num_nodes_dev + 1), dtype=float)

            # Compute total affinity graph (RBF * pheno)
            edge_index, edge_attr, aff_matrix_aug = Reader.compute_total_affinity_graph(
                features_aug, 
                pheno_aug, 
                graph_method=args.graph_method, 
                affinity_threshold=args.affinity_threshold,
                k=args.k)

            x_tensor = torch.tensor(features_aug, dtype=torch.float32)
            y_tensor = torch.zeros((num_nodes_dev + 1,), dtype=torch.long)
            test_mask = torch.zeros((num_nodes_dev + 1,), dtype=torch.bool)
            test_mask[-1] = True  # Only the last node is the test subject

            data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr,
                y=y_tensor, test_mask=test_mask)

            # Forward pass on the augmented graph
            logits = model(data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device))

            # Take the output for the single test node (last index)
            logits_test = logits[-1].unsqueeze(0)
            probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()[0]
            pred_test = int(np.argmax(probs_test))

            prob_pos_list.append(float(probs_test[pos_class_index]))
            pred_list.append(pred_test)

    # Metrics on the full test set
    precision, recall, f1 = prf(pred_list, y_true)

    if args.num_classes == 2:
        # We already collected prob_pos_list as P(class 0)
        probs_2col = np.column_stack([1 - np.array(prob_pos_list),
                                    np.array(prob_pos_list)])
        auc_score = auc(probs_2col, y_true, is_logit=False)
        spe = specificity(pred_list, y_true)
        npv_score = npv(pred_list, y_true)
    else:
        auc_score = None
        spe = None
        npv_score = None

    test_acc = float(np.mean(np.array(pred_list) == y_true))

    print("\n=== Inductive test results ===")
    print(f"Test Accuracy:         {test_acc:.4f}")
    if auc_score is not None:
        print(f"Test AUC:              {auc_score:.4f}")
    print(f"Test F1-score:         {f1:.4f}")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall (Sensitivity):  {recall:.4f}")
    if spe is not None:
        print(f"Specificity:           {spe:.4f}")
    if npv_score is not None:
        print(f"NPV:                   {npv_score:.4f}")

    final_test_results = {
        'test_accuracy': test_acc,
        'test_auc': auc_score,
        'test_f1': f1,
        'test_precision': precision,
        'test_recall': recall,
        'test_specificity': spe,
        'test_npv': npv_score,
        'y_true': y_true.tolist(),
        'y_pred': pred_list,
        'prob_pos': prob_pos_list}

    return final_test_results
