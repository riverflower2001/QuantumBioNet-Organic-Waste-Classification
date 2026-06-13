"""
Quantum BioNet 2.0
This repository accompanies the published paper and is provided for
academic inspection and educational purposes.

NOTE:
- Google Colab-specific code has been removed.
- Dataset paths and pretrained weights are intentionally omitted.
- Additional resources are required to reproduce the published results.
"""

# ---- Section ----

# 9️⃣ Training Loop (updated with stage_name)
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, scheduler=None, stage_name=""):
    print(f"\n🟢 Starting training: {stage_name}")

    model.to(device)
    history = {'train_acc': [], 'val_acc': []}
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        n_train, n_val = 0, 0
        correct_train, correct_val = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if out.shape[1] == 1:
                y = y.float().unsqueeze(1)
                loss = criterion(out, y)
                preds = (torch.sigmoid(out) > 0.5).long()
            else:
                loss = criterion(out, y)
                preds = torch.argmax(out, 1)
            loss.backward()
            optimizer.step()
            correct_train += (preds.flatten() == y.flatten()).sum().item()
            n_train += y.size(0)

        acc_train = correct_train / n_train
        model.eval()

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if out.shape[1] == 1:
                    y = y.float().unsqueeze(1)
                    preds = (torch.sigmoid(out) > 0.5).long()
                else:
                    preds = torch.argmax(out, 1)
                correct_val += (preds.flatten() == y.flatten()).sum().item()
                n_val += y.size(0)

        acc_val = correct_val / n_val
        history['train_acc'].append(acc_train)
        history['val_acc'].append(acc_val)

        if scheduler:
            scheduler.step(acc_val)

        print(f"[{stage_name}] Epoch {epoch + 1}/{num_epochs} | Train Acc: {acc_train:.4f} | Val Acc: {acc_val:.4f}")

        if acc_val >= best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), f"PRETRAINED_WEIGHTS_NOT_INCLUDED")  # Save best

    return history

# ---- Section ----

# 1️⃣1️⃣ Training (with best-practice hyperparameters)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Stage 1 (Binary Classification)
model_stage1 = HybridNet(num_classes=1, stage=1)
optimizer1 = torch.optim.Adam(model_stage1.parameters(), lr=5e-4, weight_decay=1e-4)
criterion1 = nn.BCEWithLogitsLoss(pos_weight=class_weights1[1].to(device))
train_model(model_stage1, train_loader1, val_loader1, criterion1, optimizer1, device, num_epochs, stage_name="Stage 1 (Binary)")

# Stage 2 (Multiclass Classification)
model_stage2 = HybridNet(num_classes=len(class_names2), stage=2)
optimizer2 = torch.optim.Adam(model_stage2.parameters(), lr=5e-4, weight_decay=1e-4)
criterion2 = nn.CrossEntropyLoss(weight=class_weights2.to(device))
train_model(model_stage2, train_loader2, val_loader2, criterion2, optimizer2, device, num_epochs, stage_name="Stage 2 (Multiclass)")

